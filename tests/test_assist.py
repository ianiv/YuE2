"""``yue2_studio.assist`` without a network or a ``claude`` binary: ``subprocess.run`` / ``_post_json``
are replaced, ``shutil.which`` is pinned so the resolve() matrix does not depend on the host."""

import io
import json
import logging
import os
import subprocess
import urllib.error
from email.message import Message
from pathlib import Path

import pytest

from yue2_studio import assist

SYSTEM = "system text"
REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def no_env_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


@pytest.fixture
def cli(monkeypatch):
    monkeypatch.setattr(assist, "cli_path", lambda: "/stub/claude")


@pytest.fixture
def no_cli(monkeypatch):
    monkeypatch.setattr(assist, "cli_path", lambda: None)


# -- prompt / message ---------------------------------------------------------------------------


def test_system_prompt_ships_with_the_package():
    text = assist.SYSTEM_PROMPT
    assert "Output contract" in text and "Page rules" in text and "Refinement rule" in text
    assert "[Intro]" in text and "88 BPM" in text  # the skill's concrete examples survived
    assert "## 5. Output formats" not in text and "## 6. Workflow" not in text  # skill-only sections dropped


def test_build_user_message_without_context():
    assert assist.build_user_message("  a song  ", "create") == "Page: create\n\nRequest: a song"
    assert assist.build_user_message("x", "hum", {}) == "Page: hum\n\nRequest: x"
    assert assist.build_user_message("x", "hum", {"title": "", "style": None}) == "Page: hum\n\nRequest: x"


def test_build_user_message_with_context_is_fenced_and_truncated():
    msg = assist.build_user_message("make it sadder", "create",
                                    {"title": "T", "style": "s" * 2500, "cfg_scale": 2, "cot": "off",
                                     "lyrics": "[Verse]\nla", "junk": "ignored"})
    head, fenced, tail = msg.split("\n\n")
    assert head == "Page: create" and tail == "Request: make it sadder"
    assert fenced.startswith("Current form:\n```json\n") and fenced.endswith("\n```")
    current = json.loads(fenced[len("Current form:\n```json\n"):-len("\n```")])
    assert set(current) == {"title", "style", "cfg_scale", "cot", "lyrics"}
    assert current["cfg_scale"] == "2" and current["cot"] == "off"  # stringified
    assert len(current["style"]) == 2001 and current["style"].endswith("…")


# -- clean ---------------------------------------------------------------------------------------


def test_clean_strips_drops_empty_and_ignores_unknown():
    out = assist.clean({"title": "  T ", "style": "", "lyrics": None, "cot": "melody", "cfg_scale": 2,
                        "notes": " n ", "extra": 1}, "create")
    assert out == {"title": "T", "cot": "melody", "cfg_scale": 2.0, "notes": "n"}


def test_clean_keeps_explicit_cfg_null_on_create_only():
    assert assist.clean({"title": "T", "cfg_scale": None}, "create") == {"title": "T", "cfg_scale": None}
    assert assist.clean({"title": "T", "cot": None}, "create") == {"title": "T"}  # other nulls still drop
    assert assist.clean({"title": "T"}, "create") == {"title": "T"}  # absent stays absent
    assert assist.clean({"title": "T", "cfg_scale": None}, "cover") == {"title": "T"}


@pytest.mark.parametrize("page", ["cover", "hum"])
def test_clean_drops_mode_and_cfg_off_the_create_page(page):
    out = assist.clean({"title": "T", "cot": "full", "cfg_scale": 1.5, "lyrics": "[Verse]\nla"}, page)
    assert out == {"title": "T", "lyrics": "[Verse]\nla"}


@pytest.mark.parametrize("raw", [
    {"cot": "fast"}, {"cfg_scale": 21}, {"cfg_scale": -1}, {"cfg_scale": True}, {"title": "t" * 201},
    {"notes": "n" * 501}, {"style": 3}, "not an object", None, [],
])
def test_clean_rejects_unusable_output(raw):
    with pytest.raises(assist.AssistError):
        assist.clean(raw, "create")


# -- resolve -------------------------------------------------------------------------------------


@pytest.mark.parametrize(("mode", "has_cli", "key", "provider", "model", "reasons"), [
    ("auto", True, "", "cli", None, []),
    ("auto", True, "sk", "cli", None, []),
    ("auto", False, "sk", "api", assist.API_DEFAULT_MODEL, []),
    ("auto", False, "", None, None, [assist.NO_CLI, assist.NO_KEY]),
    ("cli", True, "sk", "cli", None, []),
    ("cli", False, "sk", None, None, [assist.NO_CLI]),
    ("api", True, "sk", "api", assist.API_DEFAULT_MODEL, []),
    ("api", True, "", None, None, [assist.NO_KEY]),
    ("off", True, "sk", None, None, [assist.OFF]),
])
def test_resolve_matrix(monkeypatch, no_env_key, mode, has_cli, key, provider, model, reasons):
    monkeypatch.setattr(assist, "cli_path", lambda: "/stub/claude" if has_cli else None)
    settings = {"assist_provider": mode, "assist_model": "", "anthropic_api_key": key}
    resolved = assist.resolve(settings)
    assert (resolved.provider, resolved.model, resolved.reasons) == (provider, model, reasons)
    status = assist.status(settings)
    assert status == {"provider": provider, "cli": has_cli, "api_key": bool(key), "model": model,
                      "reasons": reasons}


def test_resolve_model_override_and_env_key(monkeypatch, no_cli):
    monkeypatch.setenv("ANTHROPIC_API_KEY", " sk-env ")
    resolved = assist.resolve({"assist_provider": "auto", "assist_model": " claude-opus-5 ",
                               "anthropic_api_key": ""})
    assert (resolved.provider, resolved.model) == ("api", "claude-opus-5")
    assert assist.api_key({"anthropic_api_key": ""}) == "sk-env"
    assert assist.api_key({"anthropic_api_key": "sk-settings"}) == "sk-settings"  # Settings wins over env
    monkeypatch.setattr(assist, "cli_path", lambda: "/stub/claude")
    assert assist.resolve({"assist_provider": "cli", "assist_model": "haiku"}).model == "haiku"


def test_defaults_when_settings_lack_assist_keys(no_cli, no_env_key):
    assert assist.resolve({}).reasons == [assist.NO_CLI, assist.NO_KEY]
    assert assist.api_key({}) is None


# -- run_cli -------------------------------------------------------------------------------------


def cli_payload(**overrides):
    payload = {"type": "result", "subtype": "success", "is_error": False, "result": "done",
               "structured_output": {"title": "Ok", "style": "pop", "cot": "full", "notes": "n"},
               "modelUsage": {"claude-haiku-4-5-20251001": {"inputTokens": 1}}}
    payload.update(overrides)
    return payload


def fake_run(monkeypatch, *, stdout="", stderr="", returncode=0, raise_=None, calls=None):
    def run(argv, **kwargs):
        if calls is not None:
            calls.append((argv, kwargs))
        if raise_ is not None:
            raise raise_
        return subprocess.CompletedProcess(argv, returncode, stdout, stderr)

    monkeypatch.setattr(subprocess, "run", run)


def test_run_cli_success_argv_and_stdin(monkeypatch, cli):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-exported")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "tok")
    calls = []
    fake_run(monkeypatch, stdout=json.dumps(cli_payload()), calls=calls)
    fields, model = assist.run_cli(SYSTEM, "Page: create\n\nRequest: x", model="haiku", timeout=7)
    assert fields == {"title": "Ok", "style": "pop", "cot": "full", "notes": "n"}
    assert model == "claude-haiku-4-5-20251001"
    (argv, kwargs), = calls
    assert argv[0] == "/stub/claude" and "-p" in argv and "--bare" not in argv
    assert argv[argv.index("--tools") + 1] == ""  # no tools at all
    assert "--no-session-persistence" in argv
    assert "--strict-mcp-config" in argv and "--mcp-config" not in argv  # zero MCP servers
    assert "--disable-slash-commands" in argv  # no skills/hooks for a form fill
    assert "ANTHROPIC_API_KEY" not in kwargs["env"] and "ANTHROPIC_AUTH_TOKEN" not in kwargs["env"]
    assert kwargs["env"].get("PATH") == os.environ.get("PATH")  # the rest of the environment is intact
    assert argv[argv.index("--output-format") + 1] == "json"
    assert json.loads(argv[argv.index("--json-schema") + 1]) == assist.SCHEMA
    assert argv[argv.index("--system-prompt") + 1] == SYSTEM
    assert argv[argv.index("--model") + 1] == "haiku"
    assert kwargs["input"] == "Page: create\n\nRequest: x" and kwargs["text"] is True
    assert kwargs["timeout"] == 7 and kwargs["capture_output"] is True
    # never the repo or cwd: a throwaway dir keeps CLAUDE.md auto-discovery out of the request
    assert Path(kwargs["cwd"]).is_absolute() and not Path(kwargs["cwd"]).is_relative_to(REPO)


def test_run_cli_without_model_flag(monkeypatch, cli):
    calls = []
    fake_run(monkeypatch, stdout=json.dumps(cli_payload(modelUsage={})), calls=calls)
    fields, model = assist.run_cli(SYSTEM, "x")
    assert "--model" not in calls[0][0] and model is None


def test_run_cli_not_logged_in(monkeypatch, cli):
    payload = cli_payload(is_error=True, result="Not logged in · Please run /login", structured_output=None)
    fake_run(monkeypatch, stdout=json.dumps(payload), returncode=1)
    with pytest.raises(assist.AssistError, match="Not logged in"):
        assist.run_cli(SYSTEM, "x")
    fake_run(monkeypatch, stdout=json.dumps(cli_payload(is_error=True, result="e" * 5000)), returncode=1)
    with pytest.raises(assist.AssistError) as info:
        assist.run_cli(SYSTEM, "x")
    assert len(str(info.value)) < 500  # a huge CLI error is cut to a tail, not forwarded whole


def test_run_cli_missing_structured_output(monkeypatch, cli):
    fake_run(monkeypatch, stdout=json.dumps(cli_payload(structured_output=None)))
    with pytest.raises(assist.AssistError, match="no structured output"):
        assist.run_cli(SYSTEM, "x")
    fake_run(monkeypatch, stdout=json.dumps(cli_payload(structured_output="{\"title\": \"S\"}")))
    assert assist.run_cli(SYSTEM, "x")[0] == {"title": "S"}  # a JSON string is tolerated


def test_run_cli_non_json_output(monkeypatch, cli):
    fake_run(monkeypatch, stdout="garbage", stderr="boom: something broke", returncode=2)
    with pytest.raises(assist.AssistError, match="something broke"):
        assist.run_cli(SYSTEM, "x")
    fake_run(monkeypatch, stdout="[1, 2]", returncode=0)
    with pytest.raises(assist.AssistError, match="claude CLI failed"):
        assist.run_cli(SYSTEM, "x")


def test_run_cli_timeout_and_missing_binary(monkeypatch, cli):
    fake_run(monkeypatch, raise_=subprocess.TimeoutExpired(["claude"], 5))
    with pytest.raises(assist.AssistError, match="timed out after 5s"):
        assist.run_cli(SYSTEM, "x", timeout=5)
    fake_run(monkeypatch, raise_=OSError("exec format error"))
    with pytest.raises(assist.AssistError, match="could not run"):
        assist.run_cli(SYSTEM, "x")
    monkeypatch.setattr(assist, "cli_path", lambda: None)
    with pytest.raises(assist.AssistError, match=assist.NO_CLI):
        assist.run_cli(SYSTEM, "x")


# -- run_api -------------------------------------------------------------------------------------


def api_payload(**overrides):
    payload = {"id": "msg_1", "model": "claude-sonnet-5-20260101", "stop_reason": "tool_use",
               "content": [{"type": "text", "text": "Here you go."},
                           {"type": "tool_use", "id": "tu_1", "name": assist.TOOL_NAME,
                            "input": {"title": "Ok", "lyrics": "[Verse]\nla"}}]}
    payload.update(overrides)
    return payload


def http_error(code: int, body: dict | None = None) -> urllib.error.HTTPError:
    raw = json.dumps(body or {"error": {"type": "x", "message": f"http {code}"}}).encode()
    return urllib.error.HTTPError(assist.API_URL, code, "err", Message(), io.BytesIO(raw))


def fake_post(monkeypatch, *, payload=None, raise_=None, calls=None):
    def post(url, headers, body, timeout):
        if calls is not None:
            calls.append((url, headers, body, timeout))
        if raise_ is not None:
            raise raise_
        return payload

    monkeypatch.setattr(assist, "_post_json", post)


def test_run_api_success_request_shape(monkeypatch):
    calls = []
    fake_post(monkeypatch, payload=api_payload(), calls=calls)
    fields, model = assist.run_api(SYSTEM, "Request: x", "claude-sonnet-5", "sk-test", timeout=9)
    assert fields == {"title": "Ok", "lyrics": "[Verse]\nla"} and model == "claude-sonnet-5-20260101"
    (url, headers, body, timeout), = calls
    assert url == "https://api.anthropic.com/v1/messages" and timeout == 9
    assert headers == {"x-api-key": "sk-test", "anthropic-version": "2023-06-01",
                       "content-type": "application/json"}
    assert body["model"] == "claude-sonnet-5" and body["max_tokens"] == 4096 and body["system"] == SYSTEM
    assert body["messages"] == [{"role": "user", "content": "Request: x"}]
    assert body["tools"] == [{"name": "set_song_fields", "description": assist.TOOL_DESCRIPTION,
                              "input_schema": assist.SCHEMA}]
    assert body["tool_choice"] == {"type": "tool", "name": "set_song_fields"}


def test_run_api_no_tool_use_or_refusal(monkeypatch):
    fake_post(monkeypatch, payload=api_payload(content=[{"type": "text", "text": "nope"}],
                                               stop_reason="end_turn"))
    with pytest.raises(assist.AssistError, match="no structured output"):
        assist.run_api(SYSTEM, "x", "m", "k")
    fake_post(monkeypatch, payload=api_payload(stop_reason="refusal", content=[]))
    with pytest.raises(assist.AssistError, match="declined"):
        assist.run_api(SYSTEM, "x", "m", "k")
    fake_post(monkeypatch, payload=api_payload(model=None))
    assert assist.run_api(SYSTEM, "x", "m", "k")[1] == "m"  # falls back to the requested model


@pytest.mark.parametrize(("error", "match"), [
    (http_error(401), "API key rejected"),
    (http_error(429), "rate limited"),
    (http_error(400, {"error": {"message": "model: not found"}}), "API error 400: model: not found"),
    (http_error(529, {"oops": 1}), "Anthropic API error 529"),
    (urllib.error.URLError("nodename nor servname provided"), "could not reach the Anthropic API"),
    (urllib.error.URLError(TimeoutError()), "timed out after 3s"),
    (TimeoutError(), "timed out after 3s"),
    (ConnectionResetError("reset"), "request failed"),
    (ValueError("Expecting value"), "request failed"),
])
def test_run_api_error_mapping(monkeypatch, error, match):
    fake_post(monkeypatch, raise_=error)
    with pytest.raises(assist.AssistError, match=match):
        assist.run_api(SYSTEM, "x", "m", "k", timeout=3)


# -- assist() / test() ---------------------------------------------------------------------------


def test_assist_end_to_end_cli(monkeypatch, cli, no_env_key):
    calls = []
    raw = {"title": " T ", "style": "s", "lyrics": "", "cot": "full", "cfg_scale": None,
           "notes": "if X, do Y"}
    fake_run(monkeypatch, stdout=json.dumps(cli_payload(structured_output=raw)), calls=calls)
    result = assist.assist({"assist_provider": "auto"}, prompt="a song", page="cover",
                           context={"style": "old"})
    assert result.to_api() == {"fields": {"title": "T", "style": "s"}, "notes": "if X, do Y",
                               "provider": "cli", "model": "claude-haiku-4-5-20251001",
                               "seconds": result.seconds}
    assert isinstance(result.seconds, float) and result.seconds >= 0
    assert calls[0][1]["input"] == assist.build_user_message("a song", "cover", {"style": "old"})
    assert calls[0][0][calls[0][0].index("--system-prompt") + 1] == assist.SYSTEM_PROMPT


def test_assist_end_to_end_api_and_notes_absent(monkeypatch, no_cli, no_env_key):
    fake_post(monkeypatch, payload=api_payload())
    result = assist.assist({"assist_provider": "api", "anthropic_api_key": "sk"}, prompt="x")
    assert result.provider == "api" and result.notes is None
    assert result.fields == {"title": "Ok", "lyrics": "[Verse]\nla"}


def test_assist_unavailable(no_cli, no_env_key):
    with pytest.raises(assist.AssistUnavailable) as info:
        assist.assist({"assist_provider": "auto"}, prompt="x")
    assert info.value.reasons == [assist.NO_CLI, assist.NO_KEY]
    assert str(info.value) == f"{assist.NO_CLI}; {assist.NO_KEY}"
    with pytest.raises(assist.AssistUnavailable) as info:
        assist.test({"assist_provider": "off"})
    assert info.value.reasons == [assist.OFF]


def test_test_sends_a_tiny_create_request(monkeypatch, cli):
    calls = []
    fake_run(monkeypatch, stdout=json.dumps(cli_payload(structured_output={"title": "ok"})), calls=calls)
    result = assist.test({"assist_provider": "cli"})
    assert result.fields == {"title": "ok"} and result.provider == "cli"
    assert calls[0][1]["input"].startswith("Page: create\n\nRequest: ")


# -- logging -------------------------------------------------------------------------------------


def test_logs_asking_and_answered_via_cli(monkeypatch, cli, no_env_key, caplog):
    fake_run(monkeypatch, stdout=json.dumps(cli_payload(total_cost_usd=0.0123,
                                                        usage={"input_tokens": 1500, "output_tokens": 200})))
    with caplog.at_level(logging.DEBUG, logger="yue2_studio.assist"):
        assist.assist({"assist_provider": "cli", "assist_model": "haiku"}, prompt="a very secret prompt text",
                      page="create", context={"style": "old", "title": "T"})
    lines = [r.getMessage() for r in caplog.records if r.name == "yue2_studio.assist"]
    info = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert info[0] == ("assist: request page=create prompt=25 chars context=style,title provider=cli "
                       "model=haiku")
    assert info[1].startswith("assist: asking claude via cli: claude -p --tools --strict-mcp-config")
    assert "(timeout 180s" in info[1] and SYSTEM not in info[1] and "properties" not in info[1]
    assert "assist: claude cli usage: cost=$0.0123 input_tokens=1500 output_tokens=200" in info
    assert info[-1].startswith("assist: claude answered via cli in ")
    assert "fields=[title, style, cot] notes=yes model=claude-haiku-4-5-20251001" in info[-1]
    assert not any("secret prompt" in line for line in info)  # prompt text only at DEBUG
    assert any(line == "assist: prompt starts: 'a very secret prompt text'" for line in lines)
    assert any(line.startswith("assist: claude cli exit 0 after") for line in lines)


def test_logs_failures_at_warning(monkeypatch, cli, no_env_key, caplog):
    fake_run(monkeypatch, raise_=subprocess.TimeoutExpired(["claude"], 180))
    with caplog.at_level(logging.INFO, logger="yue2_studio.assist"), pytest.raises(assist.AssistError):
        assist.assist({"assist_provider": "cli"}, prompt="x")
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings[0] == "assist: claude cli timed out after 180s"
    assert warnings[1].startswith("assist: failed via cli after ")
    caplog.clear()
    fake_run(monkeypatch, stdout="", stderr="boom", returncode=1)
    with caplog.at_level(logging.INFO, logger="yue2_studio.assist"), pytest.raises(assist.AssistError):
        assist.assist({"assist_provider": "cli"}, prompt="x")
    assert any(m.startswith("assist: claude cli failed (exit 1 after") and m.endswith(": boom")
               for m in [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING])


def test_logs_never_contain_the_api_key(monkeypatch, no_cli, caplog):
    key = "sk-ant-api03-verysecretkeyvalue"
    monkeypatch.setenv("ANTHROPIC_API_KEY", key)
    fake_post(monkeypatch, payload=api_payload(usage={"input_tokens": 10, "output_tokens": 5}))
    with caplog.at_level(logging.DEBUG, logger="yue2_studio.assist"):
        assist.assist({"assist_provider": "api", "anthropic_api_key": key}, prompt="x")
    fake_post(monkeypatch, raise_=http_error(401))
    with caplog.at_level(logging.DEBUG, logger="yue2_studio.assist"), pytest.raises(assist.AssistError):
        assist.assist({"assist_provider": "api", "anthropic_api_key": key}, prompt="x")
    text = "\n".join(r.getMessage() for r in caplog.records) + caplog.text
    assert key not in text and "x-api-key" not in text and "verysecret" not in text
    info = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    asking = f"assist: asking claude via api: POST {assist.API_URL} model=claude-sonnet-5 (timeout 120s"
    assert any(m.startswith(asking) for m in info)
    assert any(m.startswith("assist: anthropic api usage after") and m.endswith("output_tokens=5")
               for m in info)
    assert any(m.startswith("assist: claude answered via api in") for m in info)
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any(m.startswith("assist: anthropic api returned HTTP 401 after") and m.endswith("key rejected")
               for m in warnings)
