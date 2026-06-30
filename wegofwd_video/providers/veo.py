"""
wegofwd_video/providers/veo.py

Google Veo provider (Veo 3.1). BYOK: the caller passes the key; this module never
sources or logs it. Reach via the Gemini Developer API (api_key) — Vertex AI (ADC)
is a future option. NOT the consumer Gemini app (that is the fast/720p tier).

The generation is a long-running operation: we submit, poll until done, and
download the asset bytes. The whole flow runs in the caller's process (ADR-026 D1).
The `google-genai` SDK is an OPTIONAL extra (`wegofwd-video[veo]`), imported lazily
in `_make_client`; an injected `client` (a test double or a pre-built SDK client)
bypasses that import entirely.

STATUS: the call is wired. A live run still requires a real key + the [veo] extra;
once a first generation succeeds end-to-end, flip the registry's `model_verified`
to a live-tested basis (ADR-026 open item).
"""

from __future__ import annotations

import time

from wegofwd_video.contract import (
    VideoBrief,
    VideoCapabilities,
    VideoProvider,
    VideoRequest,
    VideoResult,
)
from wegofwd_video.errors import (
    VideoAuthError,
    VideoConfigurationError,
    VideoError,
    VideoRateLimitError,
    VideoResponseError,
    VideoTimeoutError,
)


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
        poll_interval_s: float = 10.0,
        client: object | None = None,
    ) -> None:
        if not api_key:
            raise VideoConfigurationError("veo provider requires a non-empty api_key (BYOK)")
        self.provider_id = "veo"
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._poll_interval_s = poll_interval_s
        self.capabilities = capabilities or VideoCapabilities(
            max_duration_s=60, resolutions=("1080p",), native_audio=True, reference_images=4
        )
        self._client = client  # injectable for tests; real client built lazily

    @property
    def model(self) -> str:
        return self._model

    def build_request(self, req: VideoRequest) -> dict:
        """Brief + per-call params -> the SDK `generate_videos` kwargs. Config keys
        are the google-genai `GenerateVideosConfig` snake_case fields (the SDK
        coerces a dict). Pure/testable."""
        config: dict = {
            "aspect_ratio": req.aspect_ratio,
            "resolution": req.resolution,
            "number_of_videos": 1,
            "generate_audio": req.audio,
        }
        if req.target_duration_s:
            config["duration_seconds"] = int(req.target_duration_s)
        if req.brief.global_negative:
            config["negative_prompt"] = req.brief.global_negative
        if req.seed is not None:
            config["seed"] = req.seed
        return {
            "model": self._model,
            "prompt": render_brief_text(req.brief),
            "config": config,
        }

    def generate(self, req: VideoRequest) -> VideoResult:
        # Reference-image (Ingredients-to-Video) wiring is a follow-up: it needs
        # resolved image bytes/handles, not the brief's pointer strings. Fail loudly
        # rather than silently dropping them.
        if req.brief.ingredients:
            raise VideoConfigurationError(
                "Veo reference-image (ingredients) wiring is not yet implemented; "
                "send a brief with no ingredients"
            )
        request = self.build_request(req)
        client = self._client or self._make_client()
        try:
            operation = client.models.generate_videos(
                model=request["model"], prompt=request["prompt"], config=request["config"]
            )
            operation = self._await_operation(client, operation)
            if getattr(operation, "error", None):
                raise VideoResponseError("Veo generation operation failed")
            videos = getattr(operation.response, "generated_videos", None) or []
            if not videos:
                raise VideoResponseError("Veo returned no video")
            video = videos[0].video
            data = self._download_bytes(client, video)
        except VideoError:
            raise
        except Exception as exc:  # noqa: BLE001 - classified + re-raised key-free
            raise self._map_error(exc) from None

        return VideoResult(
            provider_id="veo",
            model=self._model,
            asset_bytes=data,
            asset_uri=getattr(video, "uri", None),
            duration_s=float(request["config"].get("duration_seconds") or req.target_duration_s),
            resolution=req.resolution,
            has_audio=req.audio,
            c2pa_signed=True,  # Veo emits Google C2PA + SynthID
            watermark="SynthID",
            seed=req.seed,
        )

    def _await_operation(self, client, operation):
        """Poll the long-running op until done or the timeout elapses."""
        deadline = time.monotonic() + self._timeout
        while not getattr(operation, "done", False):
            if time.monotonic() > deadline:
                raise VideoTimeoutError("Veo generation timed out")
            if self._poll_interval_s:
                time.sleep(self._poll_interval_s)
            operation = client.operations.get(operation)
        return operation

    @staticmethod
    def _download_bytes(client, video) -> bytes | None:
        """Return the asset bytes, fetching them if the SDK left them lazy."""
        data = getattr(video, "video_bytes", None)
        if data:
            return data
        client.files.download(file=video)
        return getattr(video, "video_bytes", None)

    def _make_client(self):
        try:
            from google import genai  # type: ignore
        except ImportError:
            raise VideoConfigurationError(
                "veo provider requires the 'veo' extra: pip install wegofwd-video[veo]"
            ) from None
        return genai.Client(api_key=self._api_key)

    @staticmethod
    def _map_error(exc: Exception) -> VideoError:
        """Classify an SDK/transport error into the typed hierarchy, KEY-FREE.

        Branches on an HTTP status code if the SDK exposes one (`code` /
        `status_code` / `response_status`); the message is generic + the code only,
        never the exception string (which can echo request details)."""
        code = (
            getattr(exc, "code", None)
            or getattr(exc, "status_code", None)
            or getattr(exc, "response_status", None)
        )
        if code in (401, 403):
            return VideoAuthError("Veo authentication failed")
        if code == 429:
            return VideoRateLimitError("Veo rate limited")
        if code is not None:
            return VideoResponseError(f"Veo returned HTTP {code}")
        return VideoResponseError("Veo request failed")
