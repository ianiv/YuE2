"""The HTTP side of the app must never load MLX: only the worker thread's real engine may.

Each case runs in a fresh interpreter (other test modules import ``mlx`` in this process).
"""

import json
import subprocess
import sys

import pytest

GPU_MODULES = ("mlx", "lyra", "yue2", "yue2_studio.engine")
PROBE = """
import json, os, sys, tempfile
os.environ.pop("MLX_ENABLE_TF32", None)
{body}
loaded = sorted(n for n in sys.modules if n in {gpu!r} or n.startswith(tuple(m + "." for m in {gpu!r})))
print(json.dumps({{"loaded": loaded, "tf32": os.environ.get("MLX_ENABLE_TF32")}}))
"""


def probe(body: str) -> dict:
    code = PROBE.format(body=body, gpu=GPU_MODULES)
    out = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True).stdout
    return json.loads(out)


HTTP_SIDE_MODULES = ("config", "jobs", "worker", "audio", "fake", "api", "main", "lora", "hum", "projects")


@pytest.mark.parametrize("module", HTTP_SIDE_MODULES)
def test_importing_http_side_modules_never_loads_mlx(module):
    result = probe(f"import yue2_studio.{module}")
    assert result == {"loaded": [], "tf32": "0"}


def test_fake_app_never_imports_mlx(tmp_path):
    result = probe(f"from yue2_studio.main import create_app; create_app(fake=True, home={str(tmp_path)!r})")
    assert result == {"loaded": [], "tf32": "0"}


def test_fake_job_end_to_end_never_imports_mlx(tmp_path):
    body = """
from yue2_studio import config
from yue2_studio.fake import FakeEngine
home = HOME
options = config.resolve_preset("fast")
summary = FakeEngine(delay=0).create_song({"style": "s", "lyrics": "l", "cot": "off", "seed": 1}, home,
                                          options=options)
assert summary["status"] == "complete"
"""
    assert probe(body.replace("HOME", repr(str(tmp_path)))) == {"loaded": [], "tf32": "0"}


def test_real_engine_is_only_imported_lazily():
    body = """
import yue2_studio.main as m
assert "yue2_studio.engine" not in sys.modules
src = open(m.__file__).read()
assert "from yue2_studio.engine import Engine" in src  # lazy import inside _real_engine
"""
    assert probe(body)["loaded"] == []


def test_fake_hum_job_never_imports_mlx(tmp_path):
    body = """
from pathlib import Path
from yue2_studio import config
from yue2_studio.fake import FakeEngine
from yue2_studio.hum import HumOptions
home = Path(HOME)
hum = home / "hum.webm"; hum.write_bytes(b"x")
summary = FakeEngine(delay=0).hum_song(hum, home / "song", request={"style": "s", "lyrics": "l", "seed": 1},
                                      hum=HumOptions(adapter="hum_v1"), options=config.resolve_preset("fast"))
assert summary["status"] == "complete" and summary["hum"]["adapter"] == "hum_v1"
"""
    assert probe(body.replace("HOME", repr(str(tmp_path)))) == {"loaded": [], "tf32": "0"}
