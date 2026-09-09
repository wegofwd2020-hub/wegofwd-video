# Tasks

## 1. Provider
- [x] 1.1 `providers/local_diffusion.py`: pure shaping (`snap_size`, `snap_frames`,
      `render_prompt`, `render_negative`, `total_duration`, `RenderPlan`)
- [x] 1.2 `DiffusionEngine` protocol + `DiffusersLTXEngine` (lazy imports,
      sequential T5 → transformer, pre-flight memory check, mp4 export)
- [x] 1.3 `LocalDiffusionProvider` (`build_request`, `generate`, timeout, progress,
      `_map_error` with chained cause)

## 2. Registry / packaging
- [x] 2.1 `local-diffusion` spec + `local-preview` role
- [x] 2.2 `build_provider` branch (no key; rejects one)
- [x] 2.3 `[local]` extra in `pyproject.toml`; ruff per-file ignore for S310

## 3. Verification
- [x] 3.1 `tests/test_local_diffusion.py` with `FakeEngine` + `tests/data/*.json`
- [x] 3.2 `tests/test_first_local_run.py` (dry-run, refusal, missing extra, full
      run with fake engine + JSON report)
- [x] 3.3 registry/allow-list "known providers" tests updated
- [x] 3.4 `tests/test_local_diffusion_integration.py`: the REAL diffusers path on a
      tiny random-weight LTX pipeline (auto-skips without the extra) — passed
      2026-09-09 on diffusers 0.40 / transformers 5.16: `text_encoder=None` load,
      prompt-embeds input, per-step callback, H.264 export, seed determinism,
      transformer-swap fallback
- [ ] 3.5 **On the host:** `python scripts/first_local_run.py --smoke` produces a
      clip with the real weights; record numbers in `docs/local-diffusion-cpu-poc.md`
- [ ] 3.6 Flip `model_verified=True`; bump `integration_version` if the pipeline
      call had to change

## 4. Docs
- [x] 4.1 `docs/local-diffusion-cpu-poc.md` (host setup, run matrix, log)
- [x] 4.2 README provider row + usage snippet
