# wegofwd-video — OpenSpec project context

## Purpose
Shared, provider-agnostic video-generation seam for the wegofwd product family
(ADR-026). One contract (`VideoBrief` → `VideoRequest` → `VideoResult`), a
registry with role-pinning and provenance, N providers.

## Tech stack
Python ≥ 3.10, zero runtime dependencies in the core; provider SDKs are optional
extras (`veo`, `local`). pytest is the conformance gate; ruff for lint/format.

## Conventions
- A provider never sources a key, never persists an asset, never orchestrates,
  and never lets a key reach an exception, log line, `raw`, or `repr`.
- Every provider call in tests is driven through an injected fake (SDK client,
  engine); no network, no weights, no GPU in CI.
- New vendors/models stay `model_verified=False` until a real run has produced
  an asset, and provenance reports that honestly.
- Contract changes bump `VIDEO_CONTRACT_VERSION`; provider-integration changes
  bump the spec's `integration_version` only.

## Specs
Capability specs live under `openspec/specs/<capability>/spec.md` once a change
is archived; proposals under `openspec/changes/<change-id>/`.
