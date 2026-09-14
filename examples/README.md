# Example requests

`quickstart.json` and `full-song.json` are copied verbatim from
[vanch007/mlx-Yue](https://github.com/vanch007/mlx-Yue) at commit `ab0f058`
(`examples/`, Apache-2.0). They are plain YuE2 request JSON: `style`, `lyrics`, `cot`
(`full` | `melody` | `off`), `seed`, optional `abc` / `cfg_scale` / `id`, and optional
`abc_sampling` / `semantic_sampling` / `generation_config` overrides.

Run one through the engine wrapper:

```bash
uv run python scripts/smoke.py --preset fast --example examples/quickstart.json
```
