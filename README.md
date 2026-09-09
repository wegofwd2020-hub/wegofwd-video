# wegofwd-video

Shared **video-generation seam** for the wegofwd product family — a
provider-agnostic brief/request/result, a provider registry with role-pinning and
provenance, and a capability pre-check. One interface fronts N video providers.

This is the third member of the `wegofwd-*` library family, alongside
[`wegofwd-llm`](../wegofwd-llm) (text) and `wegofwd-secure` (key handling).

> **Decision record:** [ADR-026 — Video generation as a shared library](../StudyBuddy_SelfLearner/docs/adr/ADR-026-shared-video-generation-library.md).
> **Content contract + Veo prompt template:** [`project-critique/story-video-template/`](../project-critique/story-video-template).

## Library, not a service (ADR-026 D1)

Each consumer `pip install`s this and runs it **in its own process**, making its
own outbound vendor call with its own key. There is no video service to deploy.
This is what lets **kathai-chithiram** keep child content inside its own trust
boundary (no shared multi-tenant component) and what keeps every key inside the
process that owns it (ADR-001).

The package:
- **never sources keys** — the caller passes the `api_key` string (BYOK).
- **never persists assets** — it returns a `VideoResult`; storage (S3 / filesystem)
  is the caller's job.
- **never orchestrates** — `generate()` blocks until the asset is ready; the caller
  wraps it in its own queue (Pramana=Celery, kathai=subprocess).
- **never lets a key reach** an exception, log line, `raw` field, or `repr`.

## Providers

| id | model | status | notes |
|----|-------|--------|-------|
| `veo` | `veo-3.1-lite-generate-preview` | **first run done; blocked on quota** — see [`docs/provider-survey-2026-09.md`](docs/provider-survey-2026-09.md) §5 | Submit→poll→download via google-genai. Reach via Gemini API; not the consumer app. **The declared 1080p/4k + native-audio + ingredients capabilities are true on Vertex and false on the Developer API the provider actually calls** — that split is the open decision. |
| `deterministic-renderer` | `blender-grease-pencil-v2` | functional | wraps a **caller-supplied** render fn — kathai's matplotlib/blender stays in kathai; no key, no vendor. |
| `local-diffusion` | `Lightricks/LTX-Video-0.9.5` + 2B distilled checkpoint | **wired; awaiting first real run** — see [`docs/local-diffusion-cpu-poc.md`](docs/local-diffusion-cpu-poc.md) | self-hosted open weights via `diffusers`, CPU-first; no key, nothing leaves the machine. 256p–720p, ≤10 s, no audio, no ingredients. `pip install wegofwd-video[local]`. |
| `runway` | `gen-4.5` | **UNVERIFIED** | placeholder |
| `kling` | `kling-3.0` | **UNVERIFIED** | placeholder |

Logical roles decouple call sites from model ids: `narrative-video` → veo,
`safety-render` → deterministic-renderer, `fast-preview` → veo,
`local-preview` → local-diffusion.

> **Choosing a provider:** [`docs/provider-survey-2026-09.md`](docs/provider-survey-2026-09.md)
> — twelve engines compared on per-second rate, blind-vote quality, which config
> fields each API surface actually honours, and indemnity/provenance terms.
> Dated, and marks each figure's source confidence; video pricing moves monthly.

## Usage

```python
import wegofwd_video as wv

# 1) Resolve a role (no hardcoded model ids in app code)
provider_id, model = wv.resolve_role("narrative-video")   # ("veo", "veo-3.1")

# 2) Pre-check the brief against the provider's limits (fail fast, pre-dispatch)
wv.assert_brief_within_capabilities(
    provider_id, resolution="1080p", aspect="16:9", duration_s=12, ingredients=2)

# 3) Build a BYOK provider and generate (runs inside YOUR worker)
provider = wv.build_provider(provider_id, api_key=my_key)     # BYOK
result = provider.generate(wv.VideoRequest(brief=my_brief, seed=9071))

# 4) Persist where YOU choose, and stamp provenance onto the unit
store(result.asset_bytes or result.asset_uri)                 # caller owns storage
unit["provenance"].append(wv.provenance(provider_id, model, seed=9071))
```

kathai's safety path injects its own renderer:

```python
provider = wv.build_provider("deterministic-renderer", render_fn=my_blender_render)
result = provider.generate(req)   # child content never leaves this process
```

The zero-cost local path takes no key and reports its own timing:

```python
provider = wv.build_provider("local-diffusion", steps=8, on_progress=print)
result = provider.generate(wv.VideoRequest(brief=my_brief, resolution="320p", seed=9071))
result.raw["seconds_per_step"]   # the number a CPU proof-of-concept exists to measure
```

## Verifying a provider

`scripts/first_veo_run.py` exercises the Veo path end to end and reports which
stage failed rather than a bare traceback — role resolution, the capability
pre-check, request construction, submit/poll/download, and persistence.

```bash
python scripts/first_veo_run.py --dry-run           # everything but the API call
python scripts/first_veo_run.py --out /tmp/veo.mp4  # the real, billable call
```

`scripts/first_local_run.py` does the same for the local path — `--dry-run`
plans without loading torch, `--smoke` renders the smallest possible clip, and
every real run prints seconds-per-step, wall clock and peak RSS (`--json` keeps
them). The runbook and measurement log are in
[`docs/local-diffusion-cpu-poc.md`](docs/local-diffusion-cpu-poc.md).

`--dry-run` prints the exact `generate_videos` kwargs, so a brief can be
inspected before anything is spent. The key comes from `GEMINI_API_KEY` or
`~/.config/wegofwd/gemini.key`, and is never printed or logged.

Veo is reached through the **Gemini API**, not the consumer app; a key without
that access fails as `VideoAuthError`, which the script calls out separately.

## Layout

```
wegofwd_video/
  contract.py     # VIDEO_CONTRACT_VERSION, VideoBrief/Shot/Ingredient, VideoRequest/Result, VideoProvider ABC
  errors.py       # typed VideoError hierarchy
  registry.py     # specs, registry, roles, build_provider, validate_selection, provenance, capability check
  providers/
    veo.py            # Veo 3.1 — request shaping done; live call stubbed (ADR-026 D7)
    local_render.py   # CallableRenderProvider (caller-supplied renderer)
    local_diffusion.py # LocalDiffusionProvider — LTX-Video via diffusers, CPU-first
openspec/             # project context + change proposals/specs (OpenSpec)
schema/video_brief.v1.json
tests/                # the conformance gate — travels with the code
```

## Status

`v1.0.0` — **interface frozen** (additive-by-default; breaking changes bump major
+ `VIDEO_CONTRACT_VERSION`). The ADR-026 D7 gate is met: both real consumers are
merged on two provider paths — pramana (`veo`) and kathai-chithiram
(`deterministic-renderer`). The Veo live call is wired (submit→poll→download) and
**the first real run happened on 2026-09-02**: it corrected the model id
(`veo-3.1` was a marketing name) and then stopped at `429 RESOURCE_EXHAUSTED` —
Veo is paid and the project has no quota. `model_verified` is now `True` for the
id, but the *path* is unresolved: the registry's `base_url` is Vertex while the
provider builds a Developer-API client, and `seed`, `generate_audio`,
`reference_images` and `duration_seconds` are all rejected on that path. A
provider-integration detail that does **not** affect the frozen contract
(ADR-026); see [`docs/provider-survey-2026-09.md`](docs/provider-survey-2026-09.md) §8.

## Dev

```bash
pip install -e ".[dev,veo]"
pytest        # the test suite is the conformance gate
ruff check .
```

For the self-hosted path on a CPU box, install torch from the CPU index first
(`pip install torch --index-url https://download.pytorch.org/whl/cpu`) and then
`pip install -e ".[dev,local]"` — the runbook explains why.
