# LoRA adapters

LoRA adapters are small weight files that change how the model plans scores (the **AR** planner) or how it
renders sound (the **NAR** acoustic model). The studio merges them into the weights when a job starts, so they
cost nothing during generation.

## Adding adapters

Drop adapter files into `models/loras/`. The folder is rescanned every 5 seconds; there is no need to restart.
**Settings → Models → LoRA** lists what was found, and why any file is unusable.

Two layouts are recognised:

| Layout | Files | Notes |
|---|---|---|
| Single `.safetensors` (named after the file) | `layers.N.<block>.<proj>.lora_A` `[r,in]` and `.lora_B` `[out,r]`, optionally `vae2llm.*` / `llm2vae.*` full replacements | The layout of the YuE2 LoRAs published on Hugging Face. The file's `lora_scale` metadata (default 1.0) is honoured. |
| PEFT directory (named after the folder) | `adapter_config.json` and `adapter_model.safetensors` | `base_model.model.<module>.lora_{A,B}.weight`; scale `lora_alpha / r`. |

Examples from Hugging Face:

- [`YuE2-instrumental-cot-full-loras`](https://huggingface.co/Mothersuperior/YuE2-instrumental-cot-full-loras):
  `ar_lora_inst_v3abc.bf16.safetensors`, an **AR** adapter for instrumental music.
- [`yue2-mothersuperior-realaudio-tokenizer-v4`](https://huggingface.co/Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4):
  `nar_lora_joint_v4.bf16.safetensors`, a **NAR** adapter.
- [`YuE2-hum-to-song`](https://huggingface.co/Mothersuperior/YuE2-hum-to-song): hum adapters. Files with
  `hum_proj.*` tensors are listed with kind *hum* and chosen on the [Hum](hum-to-song.md) page, not in the
  LoRA stack.

## Using adapters

Create, Cover, Hum and the song page's regenerate form have a **LoRA adapters** field. Click **+ Add LoRA**,
pick an adapter and set its **scale** (0–4; 1 = as trained). You can stack up to 8, typically one AR adapter
and one NAR adapter. A regenerate inherits its parent's stack.

The adapters that shaped a song (name, sha256 and scale) are recorded on the job, in `summary.json` and in the
song's `result.json`, and shown under **Details** on the song page.

## How merging works

Supported targets are the AR linears (`self_attn.{q,k,v,o}_proj`, `mlp.{gate,up,down}_proj`, `lm_head`) and
the NAR linears (`nar_self_attn.*`, `nar_mlp.*`, `llm2vae`, `vae2llm`). At load time the studio computes
`W += scale · B @ A` in FP32; 8-bit and 4-bit AR weights are dequantised, merged and requantised. That takes
well under a second per adapter. A job with a different stack drops the loaded models and merges again.

## Tips for the instrumental adapter

- Use mode **full**.
- Put only *untimed*, bare section tags in the lyrics: `[intro]`, `[verse]`, `[chorus]`, … or just
  `[instrumental]`. Timed tags (`[verse 0:15-0:45]`) tend to make the adapter run to the length limit.
- Pair it with the NAR adapter. `examples/instrumental-lora.json` is a ready-made request.
- A CFG scale of 1.5–3 can help it follow the style.
- Songs land around 3–5 minutes whatever the plan says. An occasional overrun is a known trait: re-roll the
  seed rather than lowering the scale.

These weights are CC BY-NC 4.0, like YuE2 itself.
