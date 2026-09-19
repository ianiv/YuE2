""""Ask Claude" for the Create / Cover / Hum forms: a prompt in, form fields out.

Two providers, picked by the ``assist_provider`` setting: the ``claude`` CLI (Claude Code's login,
nothing to configure) or the Messages API with an API key (Settings or ``ANTHROPIC_API_KEY``). Both
are forced to answer with one JSON object matching ``SCHEMA`` — the CLI through ``--json-schema``,
the API through a forced ``tool_use`` — so the UI never has to parse prose. The system prompt lives
in ``assist_prompt.md`` next to this file (an adaptation of the ``yue2-prompt`` skill).

Pure Python (stdlib + pydantic); never imports mlx. Every network/subprocess call goes through
``run_cli`` / ``_post_json`` so tests can replace them.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

log = logging.getLogger("yue2_studio.assist")

SYSTEM_PROMPT = Path(__file__).with_name("assist_prompt.md").read_text(encoding="utf-8")

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"
API_DEFAULT_MODEL = "claude-sonnet-5"
API_MAX_TOKENS = 4096
CLI_TIMEOUT = 180
CLI_STRIPPED_ENV = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")
API_TIMEOUT = 120
TOOL_NAME = "set_song_fields"
TOOL_DESCRIPTION = ("Fill the YuE2 Studio song form. Only include the fields that should change; leave "
                    "the others out.")

PAGES = ("create", "cover", "hum")
Page = Literal["create", "cover", "hum"]
Provider = Literal["cli", "api"]
FIELD_KEYS = ("title", "style", "lyrics", "cot", "cfg_scale")
LIMITS = {"title": 200, "style": 2000, "lyrics": 20000, "notes": 500}

NO_CLI = "claude CLI not found on PATH"
NO_KEY = "no API key: add one in Settings or set ANTHROPIC_API_KEY"
OFF = "assist is turned off in Settings"

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "description": "Song title, short, no quotes."},
        "style": {"type": "string", "description": "Style tags or a prose paragraph (language, genre, "
                  "vocal, instrumentation, mood, tempo)."},
        "lyrics": {"type": "string", "description": "Lyrics with [Intro]/[Verse]/[Chorus]... section tags, "
                   "a blank line between sections; instrumental = tags without lines."},
        "cot": {"type": "string", "enum": ["full", "melody", "off"],
                "description": "Generation mode; full unless there is a reason. Create page only."},
        "cfg_scale": {"type": ["number", "null"], "description": "Omit unless it should change: 1.5-3 for "
                      "stronger adherence, null means back to the engine default. Create page only."},
        "notes": {"type": "string", "description": "One line: 'if it comes out X, change Y'."},
    },
    "additionalProperties": False,
}


class AssistError(Exception):
    """The provider ran but did not produce usable fields; the API maps it to 502 ``assist_failed``."""


class AssistUnavailable(Exception):
    """No provider can run with the current settings; the API maps it to 503 ``assist_unavailable``."""

    def __init__(self, reasons: list[str]):
        super().__init__("; ".join(reasons))
        self.reasons = reasons


class Fields(BaseModel):
    """What Claude may return. Unknown keys are ignored; strings are stripped before the length checks."""

    model_config = ConfigDict(extra="ignore")
    title: str | None = Field(default=None, max_length=LIMITS["title"])
    style: str | None = Field(default=None, max_length=LIMITS["style"])
    lyrics: str | None = Field(default=None, max_length=LIMITS["lyrics"])
    cot: Literal["full", "melody", "off"] | None = None
    cfg_scale: float | None = Field(default=None, ge=0, le=20)
    notes: str | None = Field(default=None, max_length=LIMITS["notes"])

    @field_validator("title", "style", "lyrics", "notes", "cot", mode="before")
    @classmethod
    def _strip(cls, v):
        if isinstance(v, str):
            v = v.strip()
            return v or None  # an empty string means "nothing to say", not a value
        return v

    @field_validator("cfg_scale", mode="before")
    @classmethod
    def _number(cls, v):
        if isinstance(v, bool):
            raise ValueError("must be a number or null")
        return v


def clean(raw: Any, page: str) -> dict:
    """Validate a provider's JSON object into a dict of the present fields (``notes`` included).

    Empty strings and nulls are dropped (except an explicit ``cfg_scale: null`` on create, see below);
    on cover/hum the mode/CFG do not exist in the form, so they are dropped even if Claude filled them in.
    """
    if not isinstance(raw, dict):
        raise AssistError("Claude returned no structured output")
    try:
        fields = Fields.model_validate(raw)
    except ValidationError as error:
        parts = [f"{'.'.join(str(x) for x in e.get('loc', ()))}: {e.get('msg')}" for e in error.errors()]
        raise AssistError("Claude returned unusable fields: " + "; ".join(parts)) from None
    out = {k: v for k, v in fields.model_dump().items() if v is not None}
    if page == "create":
        # an explicit null is an answer ("back to the engine default"), so Refine can reset the CFG
        if "cfg_scale" in raw and raw["cfg_scale"] is None:
            out["cfg_scale"] = None
    else:
        out.pop("cot", None)
        out.pop("cfg_scale", None)
    return out


def build_user_message(prompt: str, page: str, context: dict | None = None) -> str:
    """``Page:`` line, optional ``Current form:`` fenced JSON (only the known keys, stringified and
    capped at the field limits so a pasted novel cannot blow up the request), then ``Request:``."""
    parts = [f"Page: {page}"]
    current = {}
    for key in FIELD_KEYS:
        value = (context or {}).get(key)
        if value is None or value == "":
            continue
        text = str(value)
        limit = LIMITS.get(key, 200)
        current[key] = text if len(text) <= limit else text[:limit] + "…"
    if current:
        parts.append("Current form:\n```json\n" + json.dumps(current, ensure_ascii=False, indent=1) + "\n```")
    parts.append(f"Request: {prompt.strip()}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------------------------
# provider resolution
# ---------------------------------------------------------------------------------------------


def cli_path() -> str | None:
    return shutil.which("claude")


def api_key(settings: dict) -> str | None:
    """The key from Settings, else ``ANTHROPIC_API_KEY``; ``None`` when neither is set."""
    key = (settings.get("anthropic_api_key") or "").strip() or os.environ.get("ANTHROPIC_API_KEY", "").strip()
    return key or None


@dataclass(frozen=True)
class Resolved:
    provider: Provider | None
    model: str | None
    reasons: list[str] = field(default_factory=list)
    cli: bool = False  # the claude CLI is on PATH
    key: bool = False  # an API key is set (Settings or env)


def resolve(settings: dict) -> Resolved:
    """Which provider a request would use: ``auto`` prefers the CLI (no key to manage) over the API.

    The model is ``assist_model`` when set, else the provider default (the CLI picks its own; the
    API needs one).
    """
    mode = settings.get("assist_provider", "auto")
    model = (settings.get("assist_model") or "").strip() or None
    have_cli, have_key = cli_path() is not None, api_key(settings) is not None
    if mode == "off":
        return Resolved(None, None, [OFF], have_cli, have_key)
    if mode in ("auto", "cli") and have_cli:
        return Resolved("cli", model, [], have_cli, have_key)
    if mode in ("auto", "api") and have_key:
        return Resolved("api", model or API_DEFAULT_MODEL, [], have_cli, have_key)
    reasons = []
    if mode in ("auto", "cli"):
        reasons.append(NO_CLI)
    if mode in ("auto", "api"):
        reasons.append(NO_KEY)
    return Resolved(None, None, reasons, have_cli, have_key)


def status(settings: dict) -> dict:
    """``Status.assist`` for ``GET /api/status``."""
    resolved = resolve(settings)
    return {"provider": resolved.provider, "cli": resolved.cli, "api_key": resolved.key,
            "model": resolved.model, "reasons": resolved.reasons}


# ---------------------------------------------------------------------------------------------
# providers
# ---------------------------------------------------------------------------------------------


def _tail(text: str, limit: int = 400) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else "…" + text[-limit:]


def run_cli(system: str, user: str, model: str | None = None,
            timeout: float = CLI_TIMEOUT) -> tuple[dict, str | None]:
    """One ``claude -p`` call; returns ``(structured_output, model_used)``.

    ``--tools ""``, ``--strict-mcp-config`` (with no ``--mcp-config`` = zero MCP servers),
    ``--disable-slash-commands`` and ``--no-session-persistence`` keep it a plain completion that does
    not load the user's servers, skills or hooks for a form fill; the user message goes on stdin so it
    is never subject to argv limits. The call runs in a throwaway directory so the CLI does not pick up
    any ``CLAUDE.md`` from the repo or home, and without ``ANTHROPIC_API_KEY`` / ``ANTHROPIC_AUTH_TOKEN``
    so it bills the Claude Code login rather than a key that happens to be exported. Never pass
    ``--bare``: it skips the keychain read and every call comes back "Not logged in".
    """
    exe = cli_path()
    if exe is None:
        raise AssistError(NO_CLI)
    argv = [exe, "-p", "--tools", "", "--strict-mcp-config", "--disable-slash-commands",
            "--no-session-persistence", "--output-format", "json", "--json-schema", json.dumps(SCHEMA),
            "--system-prompt", system]
    if model:
        argv += ["--model", model]
    env = {k: v for k, v in os.environ.items() if k not in CLI_STRIPPED_ENV}
    # flags only: the system prompt and schema bodies would swamp the log line
    flags = [a for a in argv[1:] if a.startswith("-") or a in ("json", model)]
    log.info("assist: asking claude via cli: %s %s (timeout %.0fs, %d-char message)", Path(exe).name,
             " ".join(flags), timeout, len(user))
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="yue2-assist-") as cwd:
        try:
            proc = subprocess.run(argv, input=user, capture_output=True, text=True, timeout=timeout, cwd=cwd,
                                  env=env)
        except subprocess.TimeoutExpired:
            log.warning("assist: claude cli timed out after %.0fs", timeout)
            raise AssistError(f"claude CLI timed out after {timeout:.0f}s") from None
        except OSError as error:
            log.warning("assist: could not run the claude cli: %s", error)
            raise AssistError(f"could not run the claude CLI: {error}") from None
    elapsed = time.monotonic() - started
    log.debug("assist: claude cli exit %d after %.1fs, stdout tail: %s", proc.returncode, elapsed,
              _tail(proc.stdout))
    try:
        payload = json.loads(proc.stdout)
        if not isinstance(payload, dict):
            raise ValueError
    except ValueError:
        detail = _tail(proc.stderr) or _tail(proc.stdout) or f"exit status {proc.returncode}"
        log.warning("assist: claude cli failed (exit %d after %.1fs): %s", proc.returncode, elapsed, detail)
        raise AssistError(f"claude CLI failed: {detail}") from None
    if payload.get("is_error"):
        detail = _tail(str(payload.get("result") or "")) or "unknown error"
        log.warning("assist: claude cli reported an error after %.1fs: %s", elapsed, detail)
        raise AssistError(f"claude CLI: {detail}")
    output = payload.get("structured_output")
    if isinstance(output, str):
        try:
            output = json.loads(output)
        except ValueError:
            output = None
    if not isinstance(output, dict):
        log.warning("assist: claude cli answered after %.1fs without structured output", elapsed)
        raise AssistError("Claude returned no structured output")
    usage = payload.get("modelUsage")
    used = next(iter(usage), None) if isinstance(usage, dict) and usage else None
    cost, tokens = payload.get("total_cost_usd"), payload.get("usage")
    if cost is not None or tokens:
        cost_text = f"${cost:.4f}" if isinstance(cost, int | float) else "?"
        in_tok = tokens.get("input_tokens") if isinstance(tokens, dict) else None
        out_tok = tokens.get("output_tokens") if isinstance(tokens, dict) else None
        log.info("assist: claude cli usage: cost=%s input_tokens=%s output_tokens=%s", cost_text, in_tok,
                 out_tok)
    return output, used or model


def _post_json(url: str, headers: dict, body: dict, timeout: float) -> dict:
    """The one network call of the API provider (tests replace it)."""
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _http_error_message(error: urllib.error.HTTPError) -> str:
    if error.code == 401:
        return "API key rejected"
    if error.code == 429:
        return "rate limited by the Anthropic API; try again in a moment"
    try:
        detail = json.loads(error.read().decode("utf-8"))
        message = detail["error"]["message"]
    except (ValueError, KeyError, TypeError, AttributeError, OSError):
        message = error.reason
    return f"Anthropic API error {error.code}: {message}"


def run_api(system: str, user: str, model: str, key: str,
            timeout: float = API_TIMEOUT) -> tuple[dict, str | None]:
    """One Messages API call with a forced ``tool_use`` so the answer is exactly ``SCHEMA``."""
    headers = {"x-api-key": key, "anthropic-version": API_VERSION, "content-type": "application/json"}
    body = {
        "model": model,
        "max_tokens": API_MAX_TOKENS,
        "system": system,
        "messages": [{"role": "user", "content": user}],
        "tools": [{"name": TOOL_NAME, "description": TOOL_DESCRIPTION, "input_schema": SCHEMA}],
        "tool_choice": {"type": "tool", "name": TOOL_NAME},
    }
    log.info("assist: asking claude via api: POST %s model=%s (timeout %.0fs, %d-char message)", API_URL,
             model, timeout, len(user))
    started = time.monotonic()
    try:
        payload = _post_json(API_URL, headers, body, timeout)
    except urllib.error.HTTPError as error:
        message = _http_error_message(error)
        log.warning("assist: anthropic api returned HTTP %d after %.1fs: %s", error.code,
                    time.monotonic() - started, message)
        raise AssistError(message) from None
    except TimeoutError:
        log.warning("assist: anthropic api timed out after %.0fs", timeout)
        raise AssistError(f"Anthropic API timed out after {timeout:.0f}s") from None
    except urllib.error.URLError as error:
        if isinstance(error.reason, TimeoutError):
            log.warning("assist: anthropic api timed out after %.0fs", timeout)
            raise AssistError(f"Anthropic API timed out after {timeout:.0f}s") from None
        log.warning("assist: could not reach the anthropic api: %s", error.reason)
        raise AssistError(f"could not reach the Anthropic API: {error.reason}") from None
    except (OSError, ValueError) as error:
        log.warning("assist: anthropic api request failed: %s", error)
        raise AssistError(f"Anthropic API request failed: {error}") from None
    elapsed = time.monotonic() - started
    usage = payload.get("usage")
    if isinstance(usage, dict):
        log.info("assist: anthropic api usage after %.1fs: input_tokens=%s output_tokens=%s", elapsed,
                 usage.get("input_tokens"), usage.get("output_tokens"))
    if payload.get("stop_reason") == "refusal":
        log.warning("assist: claude declined the request (stop_reason=refusal) after %.1fs", elapsed)
        raise AssistError("Claude declined this request")
    for block in payload.get("content") or []:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            if isinstance(block.get("input"), dict):
                return block["input"], payload.get("model") or model
    log.warning("assist: anthropic api answered after %.1fs without a tool_use block (stop_reason=%s)",
                elapsed, payload.get("stop_reason"))
    raise AssistError("Claude returned no structured output")


# ---------------------------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------------------------


@dataclass
class AssistResult:
    fields: dict
    notes: str | None
    provider: Provider
    model: str | None
    seconds: float

    def to_api(self) -> dict:
        return {"fields": self.fields, "notes": self.notes, "provider": self.provider, "model": self.model,
                "seconds": self.seconds}


def assist(settings: dict, *, prompt: str, page: str = "create", context: dict | None = None) -> AssistResult:
    """Resolve the provider, run it and return cleaned fields. Blocking: call it off the event loop."""
    resolved = resolve(settings)
    context_keys = sorted(k for k in FIELD_KEYS if (context or {}).get(k) not in (None, ""))
    provider = resolved.provider or f"unavailable ({'; '.join(resolved.reasons)})"
    log.info("assist: request page=%s prompt=%d chars context=%s provider=%s model=%s", page, len(prompt),
             ",".join(context_keys) or "none", provider, resolved.model or "default")
    log.debug("assist: prompt starts: %r", prompt[:80])
    if resolved.provider is None:
        raise AssistUnavailable(resolved.reasons)
    user = build_user_message(prompt, page, context)
    started = time.monotonic()
    try:
        if resolved.provider == "cli":
            raw, used = run_cli(SYSTEM_PROMPT, user, resolved.model)
        else:
            raw, used = run_api(SYSTEM_PROMPT, user, resolved.model, api_key(settings))
        fields = clean(raw, page)
    except AssistError as error:
        log.warning("assist: failed via %s after %.1fs: %s", resolved.provider, time.monotonic() - started,
                    error)
        raise
    notes = fields.pop("notes", None)
    seconds = round(time.monotonic() - started, 2)
    log.info("assist: claude answered via %s in %.1fs: fields=[%s] notes=%s model=%s", resolved.provider,
             seconds, ", ".join(fields), "yes" if notes else "no", used or resolved.model or "default")
    return AssistResult(fields=fields, notes=notes, provider=resolved.provider, model=used or resolved.model,
                        seconds=seconds)


def test(settings: dict) -> AssistResult:
    """The Settings page's "Test" button: the smallest real round trip through the resolved provider."""
    return assist(settings, prompt="Connection test: reply with the title 'ok' and nothing else.")
