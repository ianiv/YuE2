"""Smoke-test the engine wrapper: generate one example request and print stage events.

    uv run python scripts/smoke.py --preset fast --example examples/quickstart.json

Writes to ``data/songs/smoke-<timestamp>/`` (``plan/`` right after planning, ``song/`` at the end).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yue2_studio import config  # noqa: E402  sets MLX_ENABLE_TF32=0 before mlx is imported


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", choices=sorted(config.PRESETS), default="fast")
    parser.add_argument("--precision", choices=config.PRECISIONS)
    parser.add_argument("--ode-steps", type=int)
    parser.add_argument("--example", type=Path, default=Path("examples/quickstart.json"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--memory-budget-gib", type=float, default=config.DEFAULT_MEMORY_BUDGET_GIB)
    parser.add_argument("--require-ac", action="store_true")
    parser.add_argument("--out", type=Path, help="output directory (default data/songs/smoke-<timestamp>)")
    args = parser.parse_args(argv)

    from yue2_studio.engine import Engine

    options = config.resolve_preset(args.preset, args.precision, args.ode_steps,
                                    memory_budget_gib=args.memory_budget_gib, require_ac=args.require_ac)
    request = json.loads(args.example.read_text())
    if args.seed is not None:
        request["seed"] = args.seed
    out_dir = args.out or config.SONGS_DIR / f"smoke-{time.strftime('%Y%m%d-%H%M%S')}"
    counts: Counter[str] = Counter()
    last_abc = {"text": ""}

    def on_event(event: dict) -> None:
        counts[event["type"]] += 1
        kind = event["type"]
        if kind == "stage":
            total = f"/{event['total']}" if event.get("total") is not None else ""
            tps = f" {event['tps']} tok/s" if event.get("tps") else ""
            boundary = event["status"] != "running" or event.get("completed") in (0, None)
            if boundary or counts["stage"] % 8 == 0:
                print(f"[stage] {event['stage']}: {event['status']} {event.get('completed', '')}{total} "
                      f"{event.get('unit') or ''}{tps} ({event['seconds']}s)", flush=True)
        elif kind == "abc":
            last_abc["text"] = event["text"]
            if event.get("status") == "final":
                print(f"[abc] final score: {len(event['text'])} chars, {event.get('tokens')} tokens",
                      flush=True)
        elif kind == "token":
            if counts["token"] % 8 == 0:
                print(f"[token] {event['phase']}: {event['tokens']} tokens {event.get('tps')} tok/s",
                      flush=True)
        elif kind == "log":
            print(f"[log] {event['text']}", flush=True)

    print(f"[smoke] preset={options.preset} precision={options.precision} ode_steps={options.ode_steps} "
          f"budget={options.memory_budget_gib} GiB -> {out_dir}", flush=True)
    engine = Engine()
    started = time.perf_counter()
    try:
        summary = engine.create_song(request, out_dir, options=options, on_event=on_event)
    finally:
        engine.unload()
    wall = time.perf_counter() - started
    timing = summary["timing"]
    print(json.dumps({
        "status": summary["status"], "audio_path": summary["audio_path"],
        "seconds": round(summary["seconds"], 2),
        "truncated": summary["truncated"],
        "abc_tok_s": _rate(timing["abc"]), "semantic_tok_s": _rate(timing["semantic"]),
        "nar_seconds": round(timing["nar_seconds"], 2), "vae_seconds": round(timing["vae_seconds"], 2),
        "e2e_seconds": round(timing["e2e_seconds"], 2), "wall_seconds": round(wall, 2),
        "load": {k: round(v, 2) for k, v in timing["load"].items() if not k.endswith("events_seconds")},
        "stages": timing["stages"], "events": dict(counts),
    }, indent=2))
    result = json.loads(Path(summary["song_dir"], "result.json").read_text())
    ok = result["status"] == "complete" and Path(summary["audio_path"]).is_file()
    print(f"[smoke] result.json status={result['status']} audio={'ok' if ok else 'MISSING'}", flush=True)
    return 0 if ok else 1


def _rate(timing: dict) -> float | None:
    seconds, tokens = timing.get("seconds"), timing.get("output_tokens")
    return round(tokens / seconds, 2) if seconds and tokens else None


if __name__ == "__main__":
    sys.exit(main())
