# Design notes — `local-diffusion`

## Context
Host: 4-core Ivy Bridge (AVX only, no AVX2/FMA, no native bf16/fp16), 32 GB RAM,
GeForce GT 640 (Kepler; unsupported by CUDA 12 → treated as absent). Model:
LTX-Video, 2B distilled transformer + T5-XXL text encoder (~4.7B) + VAE.

## Decisions
1. **Sequential loading over `enable_model_cpu_offload`.** Offload is for GPUs;
   on a CPU-only host everything is already in RAM. fp32 T5-XXL (~19 GB) plus
   fp32 transformer (~8 GB) plus activations does not fit in 32 GB. Encode the
   prompt first, free the encoder, then load the transformer: peak ≈ max of the
   two phases, not their sum. `LTXPipeline` accepts `prompt_embeds`, so the
   pipeline is built with `text_encoder=None, tokenizer=None`.
2. **Text encoder in bf16, transformer in fp32.** bf16 halves the encoder's
   footprint (~10 GB) and runs once; the CPU emulates bf16 slowly but the cost is
   seconds. The transformer runs every step, and fp32 is the fast path on a CPU
   without native half-precision.
3. **Distilled checkpoint by default.** 4–10 steps at guidance 1.0 versus ~40
   with CFG; on this CPU that is the difference between usable and not.
4. **Engine seam.** `DiffusionEngine` (encode_prompt / render / frames_to_mp4)
   mirrors the Veo provider's injected `client`: tests never import torch.
5. **Timeout via the per-step callback.** Diffusion is synchronous; the budget is
   enforced from inside `callback_on_step_end` and surfaces as the contract's
   `VideoTimeoutError`. The same callback yields seconds-per-step, the POC's
   primary measurement.
6. **Resolution labels stay "NNNp".** So `assert_brief_within_capabilities`
   works unchanged. The label is nominal; the frame is snapped to LTX's 32-pixel
   grid (`720p` → 704 tall, as LTX itself documents) and `8k+1` frames.
7. **Errors are chained (`from exc`).** No credential exists on this path; the
   developer needs the torch/diffusers traceback. Messages still name only the
   exception class plus one actionable hint (memory, missing extra, weights).
8. **`model_verified=False`.** Written from the diffusers docs on a machine that
   cannot run it. Flip after `first_local_run.py` produces a clip on the host.

## Risks / open questions
- `LTXVideoTransformer3DModel.from_single_file` with an `https://…/blob/main/…`
  URL is the documented idiom; if the diffusers version on the host disagrees,
  fall back to `hf_hub_download` + a local path (one-line change in `render`).
- Padded-token masking of the T5 hidden states matches the pipeline's own
  `encode_prompt`; verify visually that the smoke clip is not noise.
- Ivy Bridge + recent torch wheels: torch ≥ 2.4 CPU wheels still run on AVX-only
  CPUs but are slower; if `import torch` reports an illegal instruction, pin an
  older wheel. Record what worked in the doc's measurement log.
