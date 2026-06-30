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
| `veo` | `veo-3.1` | **live call wired** (docs-verified; awaiting first real run) | 1080p/4k, native audio. Submit→poll→download via google-genai. Reach via Gemini API; not the consumer app. Reference-image *ingredients* are a follow-up. |
| `deterministic-renderer` | `blender-grease-pencil-v2` | functional | wraps a **caller-supplied** render fn — kathai's matplotlib/blender stays in kathai; no key, no vendor. |
| `runway` | `gen-4.5` | **UNVERIFIED** | placeholder |
| `kling` | `kling-3.0` | **UNVERIFIED** | placeholder |

Logical roles decouple call sites from model ids: `narrative-video` → veo,
`safety-render` → deterministic-renderer, `fast-preview` → veo.

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

## Layout

```
wegofwd_video/
  contract.py     # VIDEO_CONTRACT_VERSION, VideoBrief/Shot/Ingredient, VideoRequest/Result, VideoProvider ABC
  errors.py       # typed VideoError hierarchy
  registry.py     # specs, registry, roles, build_provider, validate_selection, provenance, capability check
  providers/
    veo.py            # Veo 3.1 — request shaping done; live call stubbed (ADR-026 D7)
    local_render.py   # CallableRenderProvider (caller-supplied renderer)
schema/video_brief.v1.json
tests/                # the conformance gate — travels with the code
```

## Status

`v1.0.0` — **interface frozen** (additive-by-default; breaking changes bump major
+ `VIDEO_CONTRACT_VERSION`). The ADR-026 D7 gate is met: both real consumers are
merged on two provider paths — pramana (`veo`) and kathai-chithiram
(`deterministic-renderer`). The Veo live call is wired (submit→poll→download); a
first real generation is still pending, so `veo` `model_verified` stays
docs-verified until then — a provider-integration detail that does **not** affect
the frozen contract (ADR-026).

## Dev

```bash
pip install -e ".[dev,veo]"
pytest        # the test suite is the conformance gate
ruff check .
```
