"""
wegofwd_video/registry.py

Provider metadata + a BYOK factory. Logical *roles* and provider ids map to a
(provider, model) pair so model ids live in ONE place with one update policy —
app code never hardcodes a model string. Mirrors wegofwd_llm/registry.py.

Model ids marked UNVERIFIED are placeholders and MUST be validated against the
vendor before use. Veo 3.1 base_url/model/capabilities verified against Google
docs 2026-06-30 (720p/1080p/4k, native audio, Ingredients-to-Video) but NOT yet
live-tested from our stack — see ADR-026 open questions (flip model_verified after
the first real generation).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from wegofwd_video.contract import (
    VIDEO_CONTRACT_VERSION,
    VideoCapabilities,
    VideoProvider,
    VideoRequest,
    VideoResult,
)
from wegofwd_video.errors import (
    VideoCapabilityError,
    VideoConfigurationError,
    VideoNotAllowedError,
)


@dataclass(frozen=True)
class VideoProviderSpec:
    provider_id: str
    default_model: str
    capabilities: VideoCapabilities
    base_url: str | None = None  # None for SDK-native / local providers
    managed_env_key: str = ""  # env var for the MANAGED key (unused on the BYOK path)
    model_verified: bool = False
    key_prefix: str = ""  # BYOK shape check; "" = length-only (no stable prefix)
    integration_version: int = 1


VIDEO_PROVIDER_REGISTRY: dict[str, VideoProviderSpec] = {
    "veo": VideoProviderSpec(
        provider_id="veo",
        # Reach via Vertex AI Veo API or Google Flow. NOTE: the consumer Gemini
        # app is the fast/720p tier — point production at Vertex/Flow for
        # 1080p/4k + Ingredients + seeds.
        base_url="https://aiplatform.googleapis.com",
        default_model="veo-3.1",
        capabilities=VideoCapabilities(
            max_duration_s=60,
            resolutions=("720p", "1080p", "4k"),
            aspect_ratios=("16:9", "9:16", "1:1"),
            native_audio=True,
            reference_images=4,  # Ingredients-to-Video
            upscaling=True,
        ),
        managed_env_key="VEO_API_KEY",
        model_verified=True,
    ),
    # kathai-chithiram's safety path: deterministic LOCAL render of the same brief.
    # No vendor, no key; reproducible frames for the human-review gate. The package
    # holds only the interface — the blender/matplotlib code stays in kathai and is
    # injected as a render callable (see providers.local_render / build_provider).
    "deterministic-renderer": VideoProviderSpec(
        provider_id="deterministic-renderer",
        base_url=None,
        default_model="blender-grease-pencil-v2",
        capabilities=VideoCapabilities(
            max_duration_s=120,
            resolutions=("720p", "1080p"),
            native_audio=False,
            reference_images=0,
            deterministic=True,
        ),
        model_verified=True,
    ),
    "runway": VideoProviderSpec(
        provider_id="runway",
        base_url="https://api.dev.runwayml.com",  # UNVERIFIED
        default_model="gen-4.5",  # UNVERIFIED — best for granular camera control
        capabilities=VideoCapabilities(
            max_duration_s=20, resolutions=("720p", "1080p"), native_audio=True, reference_images=1
        ),
        managed_env_key="RUNWAY_API_KEY",
        key_prefix="key_",
    ),
    "kling": VideoProviderSpec(
        provider_id="kling",
        base_url="",  # UNVERIFIED
        default_model="kling-3.0",  # UNVERIFIED — 4k/60fps, multi-shot storyboard
        capabilities=VideoCapabilities(
            max_duration_s=15,
            resolutions=("720p", "1080p", "4k"),
            aspect_ratios=("16:9", "9:16"),
            native_audio=True,
            reference_images=1,
        ),
        managed_env_key="KLING_API_KEY",
    ),
}

# Logical role -> (provider_id, model). The seam both apps call. One place to
# route by cost/safety without touching call sites.
ROLE_DEFAULTS: dict[str, tuple[str, str]] = {
    "narrative-video": ("veo", "veo-3.1"),  # pramana lessons; kathai once safety-cleared
    "safety-render": ("deterministic-renderer", "blender-grease-pencil-v2"),  # kathai default
    "fast-preview": ("veo", "veo-3.1"),  # generate cheap, upscale the keeper
}


def available_providers(allowed: Iterable[str] | None = None) -> list[str]:
    """Known providers, in registry order. `allowed` restricts to an include set
    (unknown names ignored; empty set -> []). `None` means no restriction. This is
    what a GET-available-video-providers endpoint hands the picker."""
    ids = list(VIDEO_PROVIDER_REGISTRY)
    if allowed is None:
        return ids
    allowset = set(allowed)
    return [p for p in ids if p in allowset]


def validate_selection(
    provider_id: str, model: str | None = None, *, allowed: Iterable[str] | None = None
) -> tuple[str, str]:
    """Resolve + validate a caller's choice. Unknown -> VideoConfigurationError (422);
    excluded by allow-list -> VideoNotAllowedError (403). Model accepted as-is (no
    vendor catalogue) and defaults to the spec default. Unknown check precedes allow."""
    spec = VIDEO_PROVIDER_REGISTRY.get(provider_id)
    if spec is None:
        raise VideoConfigurationError(f"unknown video provider {provider_id!r}")
    if allowed is not None and provider_id not in set(allowed):
        raise VideoNotAllowedError(f"provider {provider_id!r} excluded by the allow-list")
    return provider_id, (model or spec.default_model)


def resolve_role(role: str) -> tuple[str, str]:
    """(provider_id, model) for a logical role."""
    try:
        return ROLE_DEFAULTS[role]
    except KeyError:
        raise VideoConfigurationError(f"unknown role {role!r}") from None


def provenance(provider_id: str, model: str | None = None, *, seed: int | None = None) -> dict:
    """A stampable record of WHICH video model + versions produced an asset —
    written into StoryUnit.provenance[stage=video] so stale/outdated renders are
    detectable and regenerable. Mirrors wegofwd_llm.provenance()."""
    pid, chosen = validate_selection(provider_id, model)
    spec = VIDEO_PROVIDER_REGISTRY[pid]
    return {
        "stage": "video",
        "engine": "wegofwd-video",
        "provider": pid,
        "model": chosen,
        "model_verified": spec.model_verified,
        "integration_version": spec.integration_version,
        "contract_version": VIDEO_CONTRACT_VERSION,
        "seed": seed,
    }


def assert_brief_within_capabilities(
    provider_id: str, *, resolution: str, aspect: str, duration_s: float, ingredients: int
) -> None:
    """Fail fast BEFORE dispatch if a brief asks for more than the provider
    supports (e.g. 4 ingredients to a model that takes 1, or 4k from a 1080p-max
    provider). Raises VideoCapabilityError listing every violation."""
    spec = VIDEO_PROVIDER_REGISTRY[validate_selection(provider_id)[0]]
    caps = spec.capabilities
    problems = []
    if resolution not in caps.resolutions:
        problems.append(f"resolution {resolution} not in {caps.resolutions}")
    if aspect not in caps.aspect_ratios:
        problems.append(f"aspect {aspect} not in {caps.aspect_ratios}")
    if duration_s > caps.max_duration_s:
        problems.append(f"duration {duration_s}s > max {caps.max_duration_s}s")
    if ingredients > caps.reference_images:
        problems.append(f"{ingredients} ingredients > max {caps.reference_images}")
    if problems:
        raise VideoCapabilityError(f"{provider_id} cannot satisfy brief: " + "; ".join(problems))


def build_provider(
    provider_id: str,
    *,
    api_key: str | None = None,
    model: str | None = None,
    render_fn: Callable[[VideoRequest], VideoResult] | None = None,
    allowed: Iterable[str] | None = None,
    **vendor_opts: object,
) -> VideoProvider:
    """Construct a BYOK provider from the registry. `model` overrides the spec
    default. `allowed` is enforced here too (raises before any provider is built).

    - veo / runway / kling: require `api_key` (BYOK).
    - deterministic-renderer: requires `render_fn` (the caller's local renderer);
      no key — child content never leaves the process (ADR-026 D1/D4).
    """
    provider_id, chosen_model = validate_selection(provider_id, model, allowed=allowed)
    spec = VIDEO_PROVIDER_REGISTRY[provider_id]

    if provider_id == "deterministic-renderer":
        if render_fn is None:
            raise VideoConfigurationError(
                "deterministic-renderer requires a render_fn (the caller's local renderer)"
            )
        from wegofwd_video.providers.local_render import CallableRenderProvider

        return CallableRenderProvider(
            render_fn=render_fn, model=chosen_model, capabilities=spec.capabilities
        )

    if provider_id == "veo":
        from wegofwd_video.providers.veo import VeoProvider

        if not api_key:
            raise VideoConfigurationError("veo provider requires a non-empty api_key (BYOK)")
        return VeoProvider(
            api_key=api_key,
            model=chosen_model,
            base_url=spec.base_url or "",
            capabilities=spec.capabilities,
            **vendor_opts,
        )

    raise VideoConfigurationError(f"no constructor wired for provider {provider_id!r}")
