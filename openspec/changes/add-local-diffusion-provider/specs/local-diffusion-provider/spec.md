# local-diffusion-provider

## ADDED Requirements

### Requirement: Registered, key-less provider
The registry SHALL expose a `local-diffusion` provider with no `base_url`, no
managed key, capabilities `256p|320p|480p|720p`, aspects `16:9|9:16|1:1`,
`max_duration_s=10`, `native_audio=False`, `reference_images=0`,
`deterministic=True`, and a `local-preview` role that resolves to it.

#### Scenario: build without a key
- **WHEN** `build_provider("local-diffusion")` is called with no `api_key`
- **THEN** a `LocalDiffusionProvider` is returned whose `model` is the spec default

#### Scenario: a key is refused
- **WHEN** `build_provider("local-diffusion", api_key="x")` is called
- **THEN** `VideoConfigurationError` is raised before any engine is built

### Requirement: Capability pre-check applies unchanged
`assert_brief_within_capabilities("local-diffusion", …)` SHALL refuse
resolutions outside the list, clips over 10 s, and any ingredients.

#### Scenario: 1080p refused
- **WHEN** the pre-check is called with `resolution="1080p"`
- **THEN** `VideoCapabilityError` is raised naming the resolution

### Requirement: Geometry snapped to the model grid
`build_request` SHALL produce width and height that are multiples of 32 and a
frame count of the form 8k+1 (k ≥ 1), derived from the request's resolution,
aspect, fps, and duration (`target_duration_s`, else the shots' sum, else 2 s).

#### Scenario: 720p is nominal
- **WHEN** `resolution="720p"`, `aspect_ratio="16:9"`
- **THEN** the plan is 1248×704

#### Scenario: four seconds at 24 fps
- **WHEN** duration is 4 s and fps is 24
- **THEN** `num_frames` is 97

### Requirement: Prose prompt, no dialogue
`render_prompt` SHALL flatten the brief into descriptive prose (style, then each
shot's prompt with framing/camera/lighting) and SHALL omit dialogue and the Veo
tag block; `render_negative` SHALL merge global and per-shot negatives without
duplicates.

#### Scenario: two-shot brief
- **WHEN** a brief has two shots
- **THEN** the second shot's sentence is introduced with "Then, " and no
  `DIALOGUE`/`STYLE:` tokens appear

### Requirement: Seeded, reproducible results
`generate` SHALL use `req.seed` when given and otherwise choose one, and SHALL
report the seed used in `VideoResult.seed`.

#### Scenario: no seed supplied
- **WHEN** `VideoRequest.seed` is `None`
- **THEN** `VideoResult.seed` is an integer in `[0, 2^31)`

### Requirement: Engine seam and result shape
`generate` SHALL call `encode_prompt`, then `render` (passing a per-step
callback), then `frames_to_mp4` on the engine, and SHALL return a `VideoResult`
with `asset_bytes`, `has_audio=False`, `c2pa_signed=False`, `duration_s =
num_frames / fps`, and timing in `raw` (`render_seconds`, `seconds_per_step`).

#### Scenario: fake engine round-trip
- **WHEN** an injected engine returns bytes
- **THEN** the result carries those bytes and the progress callback saw every step

#### Scenario: empty output
- **WHEN** the engine returns empty bytes
- **THEN** `VideoResponseError` is raised

### Requirement: Wall-clock budget
`generate` SHALL raise `VideoTimeoutError` from the step callback once elapsed
time exceeds `timeout` seconds.

#### Scenario: budget exceeded mid-render
- **WHEN** elapsed time passes the budget at step N
- **THEN** `VideoTimeoutError` is raised naming step N

### Requirement: Typed, chained error mapping
Backend exceptions SHALL be mapped to the `VideoError` hierarchy with the
original chained as `__cause__`: memory exhaustion → `VideoResponseError`
(with a reduction hint); `ImportError` → `VideoConfigurationError` (naming the
`[local]` extra); `OSError` (hub/missing weights) → `VideoConfigurationError`;
geometry complaints → `VideoConfigurationError`; anything else →
`VideoResponseError`.

#### Scenario: torch absent
- **WHEN** no engine is injected and `torch` cannot be imported
- **THEN** `VideoConfigurationError` mentions `wegofwd-video[local]`

### Requirement: Honest provenance
`provenance("local-diffusion")` SHALL report `model_verified=False` until a real
run on the host has produced a clip.

#### Scenario: before verification
- **WHEN** provenance is stamped today
- **THEN** `model_verified` is `False`
