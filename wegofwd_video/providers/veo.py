"""
wegofwd_video/providers/veo.py

Google Veo provider (Veo 3.1). BYOK: the caller passes the key; this module never
sources or logs it. Reach via Vertex AI / Google Flow, NOT the consumer Gemini app
(that is the fast/720p tier — see registry).

STATUS: request-shaping (brief -> vendor payload) is implemented and tested;
the live network call is a documented stub (`generate` raises NotImplementedError)
pending the first real integration. Per ADR-026 D7 the first two real integrations
gate v1.0, and per the open questions `model_verified` flips to a live-tested basis
after the first successful generation. The google SDK is an OPTIONAL extra
(`wegofwd-video[veo]`) imported lazily, mirroring wegofwd_llm's anthropic path.
"""

from __future__ import annotations

from wegofwd_video.contract import (
    VideoBrief,
    VideoCapabilities,
    VideoProvider,
    VideoRequest,
    VideoResult,
)
from wegofwd_video.errors import VideoConfigurationError


def render_brief_text(brief: VideoBrief) -> str:
    """Flatten a VideoBrief into the structured prompt text Veo consumes, in the
    fixed order from veo_video_brief.template.md (global block, then per-shot).
    Pure + deterministic so it can be unit-tested without the SDK."""
    lines: list[str] = [f"STYLE: {brief.global_style}"]
    if brief.audio_direction:
        lines.append(f"AUDIO: {brief.audio_direction}")
    if brief.global_negative:
        lines.append(f"NEGATIVE: {brief.global_negative}")
    for ing in brief.ingredients:
        lines.append(f"INGREDIENT[{ing.role}] ref={ing.ref} \"{ing.description}\"")
    for shot in brief.shots:
        parts = [f"[{shot.scene_index}] {shot.prompt}"]
        for val in (shot.shot_type, shot.camera_move, shot.lighting):
            if val:
                parts.append(val)
        if shot.dialogue:
            parts.append(f'DIALOGUE: "{shot.dialogue}"')
        if shot.sfx:
            parts.append(f"SFX: {list(shot.sfx)}")
        if shot.duration_s:
            parts.append(f"dur={shot.duration_s}")
        if shot.negative:
            parts.append(f"NEG: {shot.negative}")
        lines.append(" | ".join(parts))
    return "\n".join(lines)


class VeoProvider(VideoProvider):
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = "",
        capabilities: VideoCapabilities | None = None,
        timeout: float = 600.0,
        client: object | None = None,
    ) -> None:
        if not api_key:
            raise VideoConfigurationError("veo provider requires a non-empty api_key (BYOK)")
        self.provider_id = "veo"
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self.capabilities = capabilities or VideoCapabilities(
            max_duration_s=60, resolutions=("1080p",), native_audio=True, reference_images=4
        )
        self._client = client  # injectable for tests; real client built lazily in generate()

    @property
    def model(self) -> str:
        return self._model

    def build_request(self, req: VideoRequest) -> dict:
        """Brief + per-call params -> the vendor request payload. Pure/testable."""
        return {
            "model": self._model,
            "prompt": render_brief_text(req.brief),
            "config": {
                "resolution": req.resolution,
                "aspectRatio": req.aspect_ratio,
                "fps": req.fps,
                "durationSeconds": req.target_duration_s or None,
                "seed": req.seed,
                "generateAudio": req.audio,
                "referenceImages": [
                    {"role": ing.role, "ref": ing.ref} for ing in req.brief.ingredients
                ],
            },
        }

    def _load_sdk(self):  # pragma: no cover - exercised only with the extra installed
        try:
            from google import genai  # type: ignore
        except ImportError:
            raise VideoConfigurationError(
                "veo provider requires the 'veo' extra: pip install wegofwd-video[veo]"
            ) from None
        return genai

    def generate(self, req: VideoRequest) -> VideoResult:  # pragma: no cover - stub
        payload = self.build_request(req)
        # TODO(ADR-026 D7): wire the long-running Vertex AI Veo op (submit + poll)
        # against the first real consumer, then flip registry model_verified to a
        # live-tested basis. Keep the call KEY-FREE in any raised error.
        raise NotImplementedError(
            "VeoProvider.generate is a scaffold stub; live Veo wiring is pending the "
            f"first real integration (built request for model {self._model!r})"
        )
