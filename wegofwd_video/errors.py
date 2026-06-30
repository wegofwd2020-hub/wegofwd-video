"""
wegofwd_video/errors.py

Typed error hierarchy for the video seam. Routing/UX branch on error *type*, and
mapping vendor SDK/HTTP errors into these keeps raw exceptions — which may
stringify an API key — from leaking upward. Raise these with KEY-FREE messages
only. Mirrors wegofwd_llm/errors.py.
"""

from __future__ import annotations


class VideoError(Exception):
    """Base for all video provider errors. Message must never contain key material."""


class VideoConfigurationError(VideoError):
    """Misconfiguration — missing/empty key, unknown provider/role, bad model,
    missing SDK extra, or a missing required constructor argument. Maps to 4xx."""


class VideoNotAllowedError(VideoError):
    """The selected provider is real/known but excluded by the author's allow-list.
    Distinct from VideoConfigurationError so the API maps it to 403 (policy) vs 422."""


class VideoAuthError(VideoError):
    """Provider rejected the credentials (401/403)."""


class VideoRateLimitError(VideoError):
    """Provider rate-limited the request (429). Retryable / failover candidate."""


class VideoTimeoutError(VideoError):
    """The generation request timed out."""


class VideoResponseError(VideoError):
    """The provider returned an unusable result (empty, malformed, 5xx)."""


class VideoCapabilityError(VideoConfigurationError):
    """The brief asks for more than the provider/model supports (resolution,
    aspect, duration, or ingredient count). Caught before dispatch."""
