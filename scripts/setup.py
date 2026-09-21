"""Download the pinned YuE2 weights into ``models/`` and print a doctor report.

Usage::

    uv run python scripts/setup.py               # generator + VAE
    uv run python scripts/setup.py --with-cover  # also SheetSage2 + MERT for covers
    uv run python scripts/setup.py --with-hum    # covers' models + the hum-to-song prosody adapter
    uv run python scripts/setup.py --skip-download   # doctor only

Exit status is non-zero when any doctor check fails.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yue2_studio import config  # noqa: E402  sets MLX_ENABLE_TF32=0 before mlx is imported


def _complete(directory: Path, expected: dict[str, dict]) -> bool:
    """True when every pinned file exists with its pinned byte size (fast, no hashing)."""
    return all((directory / name).is_file() and (directory / name).stat().st_size == meta["bytes"]
               for name, meta in expected.items())


def _tidy(directory: Path) -> None:
    """Remove snapshot_download(local_dir=...) bookkeeping; lyra's verify_conversion rejects extra files."""
    cache = directory / ".cache"
    if cache.is_dir():
        shutil.rmtree(cache)
    attributes = directory / ".gitattributes"
    if attributes.is_file():
        attributes.unlink()


HUM_ADAPTER_REPO = "Mothersuperior/YuE2-hum-to-song"
HUM_ADAPTER_FILE = "hum_adapter_v1_combined.safetensors"  # self-contained against stock YuE2-3B


def download(with_cover: bool, force: bool = False, with_hum: bool = False) -> None:
    from huggingface_hub import snapshot_download
    from lyra.conversion import _VAE_SOURCE_FILES

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = config.CONVERTED_DIR / "conversion.json"
    if force or not manifest.is_file() or not _complete(config.CONVERTED_DIR,
                                                        json.loads(manifest.read_text())["files"]):
        print(f"[setup] downloading {config.MODEL_REPO} -> {config.CONVERTED_DIR}", flush=True)
        snapshot_download(config.MODEL_REPO, local_dir=str(config.CONVERTED_DIR))
    else:
        print(f"[setup] generator already complete in {config.CONVERTED_DIR}", flush=True)
    _tidy(config.CONVERTED_DIR)
    if force or not _complete(config.VAE_DIR, _VAE_SOURCE_FILES):
        print(f"[setup] downloading {config.VAE_REPO} -> {config.VAE_DIR}", flush=True)
        snapshot_download(config.VAE_REPO, local_dir=str(config.VAE_DIR))
    else:
        print(f"[setup] VAE already complete in {config.VAE_DIR}", flush=True)
    _tidy(config.VAE_DIR)
    if with_cover:
        # lyra.transcription.model.resolve_models() calls snapshot_download(repo, revision=<pinned>,
        # cache_dir=..., local_files_only=offline, allow_patterns=[config.json, model.safetensors]),
        # so the files must live in the HF cache layout under the same cache_dir and revision.
        from lyra.transcription import model as tmodel

        pinned = {tmodel.MODEL_REPO: tmodel.MODEL_REVISION, tmodel.MERT_REPO: tmodel.MERT_REVISION}
        wanted = {config.SHEETSAGE_REPO: config.SHEETSAGE_REVISION, config.MERT_REPO: config.MERT_REVISION}
        if pinned != wanted:
            raise RuntimeError(f"lyra pins {pinned}; update yue2_studio.config to match")
        for repo, revision in wanted.items():
            print(f"[setup] fetching {repo}@{revision[:8]} -> {config.HF_CACHE_DIR}", flush=True)
            snapshot_download(repo, revision=revision, cache_dir=str(config.HF_CACHE_DIR),
                              allow_patterns=["config.json", "model.safetensors"])
    if with_hum:
        from huggingface_hub import hf_hub_download

        target = config.LORAS_DIR / HUM_ADAPTER_FILE
        if force or not target.is_file():
            print(f"[setup] fetching {HUM_ADAPTER_REPO}/{HUM_ADAPTER_FILE} -> {config.LORAS_DIR}", flush=True)
            config.LORAS_DIR.mkdir(parents=True, exist_ok=True)
            hf_hub_download(HUM_ADAPTER_REPO, HUM_ADAPTER_FILE, local_dir=str(config.LORAS_DIR))
            _tidy(config.LORAS_DIR)
        else:
            print(f"[setup] hum adapter already present at {target}", flush=True)


def doctor(verify_hashes: bool, require_ffmpeg: bool = False) -> dict:
    """Same checks as ``lyra.commands.doctor`` (verify_conversion / model_identity) plus studio extras."""
    import importlib.metadata

    import psutil
    from lyra.conversion import verify_conversion
    from lyra.runtime import runtime_status
    from yue2.storage import model_identity

    # mlx-Yue's own gate: Apple Silicon + macOS >= 14.2, or >= 26.2 on M5 chips.
    runtime = runtime_status()
    mac = runtime["macos"]
    checks = {
        "macos": runtime["system"] == "Darwin",
        "metal": runtime["metal"],
        "supported_runtime": runtime["supported"],
        "tf32_disabled": config.os.environ.get("MLX_ENABLE_TF32") == "0",
    }
    warnings = []
    if runtime["error"]:
        warnings.append(runtime["error"])
    if require_ffmpeg:
        checks["ffmpeg"] = config.FFMPEG is not None
    elif config.FFMPEG is None:
        warnings.append("ffmpeg not found on PATH; needed only for covers (transcription) and mp3 export")
    report: dict = {
        "platform": platform.platform(),
        "macos": mac,
        "runtime": runtime,
        "memory_gib": round(psutil.virtual_memory().total / 2**30, 1),
        "versions": {n: importlib.metadata.version(n) for n in ("mlx-yue", "mlx", "mlx-lm")},
        "home": str(config.HOME),
        "ffmpeg": config.FFMPEG,
        "checks": checks,
        "warnings": warnings,
    }
    try:
        if verify_hashes:
            data = verify_conversion(config.CONVERTED_DIR)
        else:
            data = json.loads((config.CONVERTED_DIR / "conversion.json").read_text())
        checks["model"] = True
        report["model"] = {"path": str(config.CONVERTED_DIR), "identity_checked": verify_hashes,
                           "precisions": config.models_available()["precisions"],
                           "format": data.get("format"), "schema": data.get("schema")}
    except (OSError, ValueError, KeyError) as error:
        checks["model"] = False
        report["model"] = {"path": str(config.CONVERTED_DIR), "error": str(error)}
    try:
        if verify_hashes:
            data = model_identity(config.VAE_DIR)
        else:
            data = json.loads((config.VAE_DIR / "config.json").read_text())
        checks["vae"] = True
        report["vae"] = {"path": str(config.VAE_DIR), "identity_checked": verify_hashes,
                         "model_type": data.get("model_type", data.get("config", {}).get("model_type"))}
    except (OSError, ValueError, KeyError) as error:
        checks["vae"] = False
        report["vae"] = {"path": str(config.VAE_DIR), "error": str(error)}
    report["cover"] = config.cover_available()
    report["status"] = "pass" if all(checks.values()) else "fail"
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--with-cover", action="store_true", help="also fetch SheetSage2 + MERT-v2-FullSong")
    parser.add_argument("--with-hum", action="store_true",
                        help="also fetch the cover models and the hum-to-song adapter into models/loras")
    parser.add_argument("--skip-download", action="store_true", help="only run the doctor report")
    parser.add_argument("--force", action="store_true", help="re-download even when the trees look complete")
    parser.add_argument("--no-verify-hashes", action="store_true",
                        help="skip hashing the ~10 GB of weights in the doctor report")
    args = parser.parse_args(argv)
    if not args.skip_download:
        download(args.with_cover or args.with_hum, force=args.force, with_hum=args.with_hum)
    report = doctor(verify_hashes=not args.no_verify_hashes, require_ffmpeg=args.with_cover or args.with_hum)
    print(json.dumps(report, indent=2))
    for warning in report["warnings"]:
        print(f"[setup] warning: {warning}", file=sys.stderr)
    if report["status"] != "pass":
        failed = sorted(k for k, ok in report["checks"].items() if not ok)
        print(f"[setup] doctor FAILED: {', '.join(failed)}", file=sys.stderr)
        return 1
    if (args.with_cover or args.with_hum) and not report["cover"]["available"]:
        print("[setup] cover models incomplete", file=sys.stderr)
        return 1
    print("[setup] doctor passed", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
