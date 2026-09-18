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

`instrumental-lora.json` is a short prompt in the caption style expected by the
[YuE2-instrumental-cot-full](https://huggingface.co/Mothersuperior/YuE2-instrumental-cot-full-loras)
AR LoRA (`cot=full`, untimed bare section tags in the lyrics field — timed tags tend to make it
overrun to the length cap). With the adapter files in `models/loras/`:

```bash
uv run python scripts/smoke.py --preset fast --example examples/instrumental-lora.json \
    --lora ar_lora_inst_v3abc.bf16 --lora nar_lora_joint_v4.bf16
```
