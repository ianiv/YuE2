# Troubleshooting

Start with the doctor report, which checks macOS, Metal, ffmpeg and the model files:

```bash
uv run python scripts/setup.py --skip-download
```

**Settings** shows the same state live, and `GET /api/status` returns it as JSON.

## Models and setup

**`503 engine_unavailable`, or the UI says the models are missing.**
Run `uv run python scripts/setup.py` (add `--with-cover` for covers and hums), and check that `models/converted`
contains `conversion.json` and the `ar-*.safetensors` files.

**Cover or Hum is disabled.**
The page lists what is missing: ffmpeg, SheetSage2 or MERT (install them with `scripts/setup.py --with-cover`),
or `librosa` for Hum (`uv sync`).

**MP3 downloads answer 503.**
ffmpeg is not on `PATH` (`brew install ffmpeg`). FLAC downloads and playback still work.

**"mlx-Yue requires macOS >=14.2", "The MLX M5 runtime requires macOS >=26.2" or "Native Apple Silicon Python
on macOS is required".**
Update macOS, or make sure `uv` created an arm64 Python rather than a Rosetta one. The doctor report prints the
full runtime check.

**`MLX_ENABLE_TF32` error.**
The engine requires `MLX_ENABLE_TF32=0` before MLX starts. The studio sets it itself; if you import the package
from your own code, import `yue2_studio.config` before `mlx`.

## Memory

**`MemoryError: Process footprint exceeds budget`, and the engine pill goes back to `cold`.**
The memory watchdog stopped the job at the budget set in Settings. Generation peaks at about 10 GiB (about
5.3 GiB in low-memory mode). On a 16 GB Mac the budget tops out at 12 GiB, so keep low-memory mode on (Auto does
that). After a failure the pipeline is discarded and the next job rebuilds it automatically.

**`Stopping GPU workload after new swapping`, `Less than 2 GiB of available system memory remains` or `System
memory pressure is not normal`.**
The guard stops a job as soon as macOS starts swapping or runs short of memory. On 16–24 GB Macs turn low-memory
mode on (Auto does it), which also relaxes these thresholds, and close memory-hungry apps (browsers, Xcode, other
ML tools) before long batches. If it still happens, lower the memory budget or use the Fast preset.

**"Memory budget must exceed 5 GiB and leave 4 GiB OS headroom", or the budget changed by itself.**
The budget must be more than 5 GiB and at most total RAM − 4 GiB. A larger value saved earlier is lowered to the
cap. Macs with less than 10 GB of RAM cannot run the studio.

**"Low-memory mode needs a newer mlx-Yue than the one installed".**
Run `uv sync` after updating the studio, or set Low-memory mode to Off.

## Jobs

**`AC power disconnected` failures.**
**Require AC power** is on in Settings and the Mac went on battery. Turn the setting off or plug in.

**A job failed with "server restarted…".**
It was running when the server stopped. Jobs that were only queued are picked up again automatically; submit
this one again.

**A hum job fails early with "no notes".**
The transcriber found no melody. It needs something voice-like: a real hum works, a synthesised tone does not.

**The instrumental adapter runs to the length limit.**
Use bare section tags without timings, and re-roll the seed. See
[LoRA adapters](lora-adapters.md#tips-for-the-instrumental-adapter).

**An adapter is listed as unusable.**
The reason is shown in Settings → Models. Hum adapters (`hum_proj.*` tensors) belong on the Hum page, not in the
LoRA stack.

## Browser

**Scores don't render.**
The page loads abcjs from cdnjs, so it needs an internet connection.

**▶ Play score (MIDI) is silent.**
It downloads General MIDI soundfonts from `paulrosen.github.io` on first use, so it needs an internet connection.

**Recording a hum doesn't start.**
The browser only allows the microphone on `localhost`, `127.0.0.1` or HTTPS, and only after you allow access.

## Running several servers

Two servers pointed at the same home share `data/app.db` and steal each other's jobs. Give each its own
`YUE2_STUDIO_HOME` (with `models/` symlinked to the shared weights).
