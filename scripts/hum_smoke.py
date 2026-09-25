"""Smoke-test hum-to-song end to end (or just the hum analysis) and print stage events.

    uv run python scripts/hum_smoke.py --audio my-hum.m4a --style "lo-fi, warm" --lyrics-file lyrics.txt
    uv run python scripts/hum_smoke.py --audio my-hum.m4a --analyse-only        # carrier only, no models
    uv run python scripts/hum_smoke.py --audio my-hum.m4a --adapter hum_adapter_v1_combined --influence 1.5

Writes to ``data/songs/hum-<timestamp>/`` (``transcription/``, ``hum/``, ``plan/``, ``song/``).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yue2_studio import config  # noqa: E402  sets MLX_ENABLE_TF32=0 before mlx is imported


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio", type=Path, required=True, help="the hum recording (anything ffmpeg reads)")
    parser.add_argument("--style", default="warm acoustic pop, female vocal, piano, 90 BPM")
    parser.add_argument("--lyrics", default="[verse]\nla la la\n\n[chorus]\nla la la la")
    parser.add_argument("--lyrics-file", type=Path)
    parser.add_argument("--melody", choices=("continue", "hum_only", "ignore"), default="continue")
    parser.add_argument("--adapter", help="hum adapter name in models/loras (kind=hum)")
    parser.add_argument("--influence", type=float, default=1.0)
    parser.add_argument("--offset", type=float, default=0.0,
                        help="seconds into the song where the hum starts")
    parser.add_argument("--preset", choices=sorted(config.PRESETS), default="fast")
    parser.add_argument("--precision", choices=config.PRECISIONS)
    parser.add_argument("--ode-steps", type=int)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--lora", action="append", default=[], metavar="NAME[:SCALE]")
    parser.add_argument("--max-abc-tokens", type=int, help="cap the continued score (smoke runs)")
    parser.add_argument("--max-semantic-tokens", type=int, help="cap the song length in tokens (25/s)")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--analyse-only", action="store_true", help="only pitch-track/encode the hum")
    parser.add_argument("--low-memory", choices=config.LOW_MEMORY_MODES, default=config.DEFAULT_LOW_MEMORY,
                        help="mlx-Yue low-memory mode (auto = on for Macs with 24 GB or less)")
    args = parser.parse_args(argv)

    from yue2_studio import audio, hum

    out_dir = args.out or config.SONGS_DIR / f"hum-{time.strftime('%Y%m%d-%H%M%S')}"
    if args.analyse_only:
        import numpy as np
        import soundfile as sf

        from yue2_studio import vae_encoder

        out_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        samples = audio.decode_pcm(args.audio, sample_rate=hum.SAMPLE_RATE, channels=1)
        analysis = hum.analyse_hum(samples)
        print(f"[hum] analysed {analysis.duration_s:.1f}s in {time.perf_counter() - started:.1f}s: "
              f"{json.dumps(analysis.prosody())}")
        stereo = hum.carrier_stereo(analysis.carrier)
        sf.write(out_dir / "carrier.flac", stereo, hum.SAMPLE_RATE, subtype="PCM_16")
        encoder = vae_encoder.load_encoder(config.VAE_DIR)
        latents = vae_encoder.encode(encoder, stereo)
        np.save(out_dir / "carrier_latents.npy", latents)
        hum.write_prosody(out_dir / "prosody.json", analysis, hum.HumOptions(),
                          {"latent_frames": int(len(latents))})
        print(f"[hum] carrier latents {latents.shape} -> {out_dir}")
        return 0

    from yue2_studio.engine import Engine

    lyrics = args.lyrics_file.read_text() if args.lyrics_file else args.lyrics
    loras = [(item.partition(":")[0], float(item.partition(":")[2] or 1.0)) for item in args.lora]
    options = config.resolve_preset(args.preset, args.precision, args.ode_steps, loras=loras,
                                    low_memory=config.resolve_low_memory(args.low_memory,
                                                                         config.DEFAULT_MEMORY_BUDGET_GIB))
    hum_options = hum.HumOptions(melody=args.melody, adapter=args.adapter, influence=args.influence,
                                 offset_s=args.offset)
    request = {"style": args.style, "lyrics": lyrics, "seed": args.seed, "id": "hum-smoke"}
    if args.max_abc_tokens:
        request["abc_sampling"] = {"max_tokens": args.max_abc_tokens}
    if args.max_semantic_tokens:
        request["semantic_sampling"] = {"max_tokens": args.max_semantic_tokens}

    def on_event(event: dict) -> None:
        kind = event["type"]
        if kind == "stage" and (event["status"] != "running" or event.get("completed") in (0, None)):
            total = f"/{event['total']}" if event.get("total") is not None else ""
            print(f"[stage] {event['stage']}: {event['status']} {event.get('completed', '')}{total} "
                  f"{event.get('unit') or ''} ({event['seconds']}s)", flush=True)
        elif kind == "abc" and event.get("status") == "final":
            print(f"[abc] final score: {len(event['text'])} chars, {event.get('tokens')} tokens", flush=True)
        elif kind == "log":
            print(f"[log] {event['text']}", flush=True)

    print(f"[hum-smoke] {hum_options.to_dict()} preset={options.preset} loras={options.loras} "
          f"low_memory={options.low_memory} -> {out_dir}")
    engine = Engine()
    started = time.perf_counter()
    try:
        summary = engine.hum_song(args.audio, out_dir, request=request, hum=hum_options, options=options,
                                  on_event=on_event)
    finally:
        engine.unload()
    timing = summary["timing"]
    print(json.dumps({
        "status": summary["status"], "seconds": round(summary["seconds"], 2),
        "truncated": summary["truncated"], "hum": summary["hum"], "abc_timing": timing["abc"],
        "nar_seconds": round(timing["nar_seconds"], 2), "e2e_seconds": round(timing["e2e_seconds"], 2),
        "wall_seconds": round(time.perf_counter() - started, 2), "stages": timing["stages"],
    }, indent=2, default=str))
    result = json.loads(Path(summary["song_dir"], "result.json").read_text())
    saved_config = json.loads(Path(summary["song_dir"], "config.json").read_text())
    print(f"[hum-smoke] result.json status={result['status']} "
          f"hum_adapter={'yes' if result['weights'].get('hum_adapter') else 'no'} "
          f"config.hum.melody={saved_config.get('hum', {}).get('melody')}")
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
