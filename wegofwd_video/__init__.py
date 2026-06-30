"""wegofwd_video — the shared video-generation seam.

A provider-agnostic seam: a typed brief/request/result, a capability descriptor,
and a provider registry with role-pinning + provenance. One interface fronts N
video providers — Google Veo (AI) and a caller-supplied deterministic renderer
(kathai's matplotlib/blender safety path), with Kling/Runway as placeholders.

Shared as a LIBRARY, not a service (ADR-026 D1): each consumer imports this and
runs it in-process, making its own vendor call with its own key. The package
NEVER sources keys (the caller passes the api_key — D2), NEVER persists assets
(the caller owns storage — D2), and NEVER lets a key reach an exception, log line,
`raw` field, or `repr`.

Consumers: pramana (BYOK Veo -> S3), kathai-chithiram (deterministic renderer).
The StoryUnit content contract + Veo brief template live in
project-critique/story-video-template/.
"""

from __future__ import annotations

from wegofwd_video.contract import (
    VIDEO_CONTRACT_VERSION,
    Ingredient,
    Shot,
    VideoBrief,
    VideoCapabilities,
    VideoProvider,
    VideoRequest,
    VideoResult,
)
from wegofwd_video.errors import (
    VideoAuthError,
    VideoCapabilityError,
    VideoConfigurationError,
    VideoError,
    VideoNotAllowedError,
    VideoRateLimitError,
    VideoResponseError,
    VideoTimeoutError,
)
from wegofwd_video.registry import (
    ROLE_DEFAULTS,
    VIDEO_PROVIDER_REGISTRY,
    VideoProviderSpec,
    assert_brief_within_capabilities,
    available_providers,
    build_provider,
    provenance,
    resolve_role,
    validate_selection,
)

__version__ = "0.1.0"

__all__ = [
    "ROLE_DEFAULTS",
    "VIDEO_CONTRACT_VERSION",
    "VIDEO_PROVIDER_REGISTRY",
    "Ingredient",
    "Shot",
    "VideoAuthError",
    "VideoBrief",
    "VideoCapabilities",
    "VideoCapabilityError",
    "VideoConfigurationError",
    "VideoError",
    "VideoNotAllowedError",
    "VideoProvider",
    "VideoProviderSpec",
    "VideoRateLimitError",
    "VideoRequest",
    "VideoResponseError",
    "VideoResult",
    "VideoTimeoutError",
    "__version__",
    "assert_brief_within_capabilities",
    "available_providers",
    "build_provider",
    "provenance",
    "resolve_role",
    "validate_selection",
]
