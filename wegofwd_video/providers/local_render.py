"""
wegofwd_video/providers/local_render.py

The non-AI provider. Wraps a caller-supplied render callable so a consumer
(kathai-chithiram) keeps its OWN deterministic renderer (matplotlib/blender) while
driving it through the same registry + brief + provenance vocabulary as the AI
path. The render code stays in the app; only the interface lives here (ADR-026 D4).

Child content never leaves the caller's process — there is no key and no vendor.
"""

from __future__ import annotations

from collections.abc import Callable

from wegofwd_video.contract import (
    VideoCapabilities,
    VideoProvider,
    VideoRequest,
    VideoResult,
)
from wegofwd_video.errors import VideoResponseError


class CallableRenderProvider(VideoProvider):
    """Adapts `render_fn(VideoRequest) -> VideoResult` to the VideoProvider ABC."""

    def __init__(
        self,
        *,
        render_fn: Callable[[VideoRequest], VideoResult],
        model: str,
        capabilities: VideoCapabilities,
    ) -> None:
        self.provider_id = "deterministic-renderer"
        self._render_fn = render_fn
        self._model = model
        self.capabilities = capabilities

    @property
    def model(self) -> str:
        return self._model

    def generate(self, req: VideoRequest) -> VideoResult:
        result = self._render_fn(req)
        if not isinstance(result, VideoResult):
            raise VideoResponseError(
                "render_fn must return a wegofwd_video.VideoResult"
            )
        return result
