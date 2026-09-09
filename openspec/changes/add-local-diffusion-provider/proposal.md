# Change: add a self-hosted `local-diffusion` provider

## Why
Every hosted video engine in `docs/provider-survey-2026-09.md` bills per second,
and the Veo path is blocked on quota. The Pramana SOX-lesson proof-of-concept
needs a zero-cost way to produce clips from the *same* `VideoBrief`, and the
available host is a CPU-only desktop (Intel i5-3570K, 32 GB RAM, no usable GPU).
Open-weight diffusion models (LTX-Video 2B distilled) can run there — slowly —
and the survey's "self-hosting buys data residency, not savings" conclusion
deserves a measured number rather than an estimate.

## What changes
- **New provider** `local-diffusion` (`wegofwd_video/providers/local_diffusion.py`):
  LTX-Video via Hugging Face `diffusers`, CPU by default, with sequential
  text-encoder → transformer loading so both fit in 32 GB, an injectable
  `DiffusionEngine` seam for tests, a wall-clock budget raised as
  `VideoTimeoutError`, and typed, chained error mapping.
- **Registry**: a `local-diffusion` spec (no `base_url`, no key, `256p`–`720p`,
  ≤10 s, no audio, no ingredients, `deterministic=True`, `model_verified=False`)
  and a `local-preview` role.
- **`build_provider`**: constructs the provider without a key; rejects one.
- **Packaging**: a `local` optional extra (torch, diffusers, transformers,
  accelerate, sentencepiece, safetensors, imageio[ffmpeg]).
- **Harness** `scripts/first_local_run.py`: dry-run / smoke / measured runs
  that report seconds-per-step, wall clock, peak RSS, and write JSON.
- **Tests** with a fake engine and sample briefs under `tests/data/`.
- **Docs**: `docs/local-diffusion-cpu-poc.md` (host setup + measurement log),
  README provider row.

## Non-goals
- Audio, reference images (ingredients), upscaling, multi-clip stitching.
- GPU tuning (the `device`/`dtype` options exist; nothing is optimised for CUDA).
- Any change to the frozen contract (`VIDEO_CONTRACT_VERSION` stays 1).

## Impact
- Additive: existing providers, roles, and tests are unchanged except the two
  "known providers" assertions, which gain the new id.
- No new runtime dependency for consumers that do not install `[local]`.
- Spec: `local-diffusion-provider` (new).
