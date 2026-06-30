import pytest

from wegofwd_video import (
    Ingredient,
    Shot,
    VideoBrief,
    VideoRequest,
    VideoResult,
    build_provider,
)
from wegofwd_video.errors import VideoAuthError, VideoConfigurationError, VideoResponseError
from wegofwd_video.providers.veo import VeoProvider, render_brief_text


def _brief():
    return VideoBrief(
        global_style="warm storybook",
        global_negative="no flashing",
        audio_direction="gentle narrator",
        ingredients=(
            Ingredient(role="character:child", ref="child.png", description="child ~6yo"),
        ),
        shots=(
            Shot(
                scene_index=1,
                prompt="child walks to the sink",
                shot_type="medium",
                camera_move="static",
                dialogue="CHILD walks to the sink.",
                duration_s=4,
            ),
        ),
    )


# ── deterministic-renderer: the caller-supplied render fn ─────────────────────
def test_deterministic_renderer_uses_caller_render_fn():
    calls = {}

    def fake_render(req: VideoRequest) -> VideoResult:
        calls["seen"] = req
        return VideoResult(
            provider_id="deterministic-renderer",
            model="blender-grease-pencil-v2",
            asset_uri="media/out.mp4",
            duration_s=4,
            resolution="1080p",
        )

    p = build_provider("deterministic-renderer", render_fn=fake_render)
    assert p.provider_id == "deterministic-renderer"
    req = VideoRequest(brief=_brief())
    out = p.generate(req)
    assert out.asset_uri == "media/out.mp4"
    assert calls["seen"] is req  # the caller's fn actually ran


def test_deterministic_renderer_requires_render_fn():
    with pytest.raises(VideoConfigurationError):
        build_provider("deterministic-renderer")


def test_deterministic_renderer_rejects_bad_return():
    p = build_provider("deterministic-renderer", render_fn=lambda req: "not-a-result")
    with pytest.raises(VideoResponseError):
        p.generate(VideoRequest(brief=_brief()))


# ── veo: BYOK construction + pure request shaping (no network) ────────────────
def test_veo_requires_api_key():
    with pytest.raises(VideoConfigurationError):
        build_provider("veo", api_key="")


def _brief_no_ingredients():
    return VideoBrief(
        global_style="warm storybook",
        global_negative="no flashing",
        audio_direction="gentle narrator",
        shots=(Shot(scene_index=1, prompt="child waves", dialogue="CHILD waves.", duration_s=8),),
    )


def test_veo_build_request_shapes_sdk_config():
    p = build_provider("veo", api_key="byok-key")
    assert isinstance(p, VeoProvider)
    payload = p.build_request(
        VideoRequest(brief=_brief(), seed=42, resolution="1080p", target_duration_s=8)
    )
    assert payload["model"] == "veo-3.1"
    cfg = payload["config"]
    assert cfg["seed"] == 42
    assert cfg["generate_audio"] is True
    assert cfg["aspect_ratio"] == "16:9"
    assert cfg["resolution"] == "1080p"
    assert cfg["duration_seconds"] == 8
    assert cfg["negative_prompt"] == "no flashing"
    assert "STYLE: warm storybook" in payload["prompt"]


# ── live call wiring, exercised with an injected fake SDK client (no network) ──
class _FakeVideo:
    def __init__(self):
        self.video_bytes = None
        self.uri = "https://veo/out.mp4"


class _FakeOp:
    def __init__(self, response, *, done_after=0, error=None):
        self._left = done_after
        self.response = response
        self.error = error

    @property
    def done(self):
        return self._left <= 0


class _FakeModels:
    def __init__(self, op):
        self._op = op
        self.calls = []

    def generate_videos(self, *, model, prompt, config):
        self.calls.append({"model": model, "prompt": prompt, "config": config})
        return self._op


class _FakeOps:
    def get(self, op):
        op._left -= 1
        return op


class _FakeFiles:
    def download(self, *, file):
        file.video_bytes = b"MP4DATA"


class _FakeClient:
    def __init__(self, op):
        self.models = _FakeModels(op)
        self.operations = _FakeOps()
        self.files = _FakeFiles()


def test_veo_generate_submits_polls_downloads():
    video = _FakeVideo()
    response = type("R", (), {"generated_videos": [type("G", (), {"video": video})()]})()
    op = _FakeOp(response, done_after=2)  # forces two poll iterations
    client = _FakeClient(op)
    p = build_provider("veo", api_key="k", client=client, poll_interval_s=0)

    result = p.generate(VideoRequest(brief=_brief_no_ingredients(), seed=7, target_duration_s=8))

    assert result.provider_id == "veo" and result.model == "veo-3.1"
    assert result.asset_bytes == b"MP4DATA"
    assert result.asset_uri == "https://veo/out.mp4"
    assert result.has_audio is True and result.c2pa_signed is True and result.watermark == "SynthID"
    assert result.seed == 7 and result.duration_s == 8
    assert client.models.calls[0]["model"] == "veo-3.1"


def test_veo_generate_rejects_ingredients_until_wired():
    client = _FakeClient(_FakeOp(None))
    p = build_provider("veo", api_key="k", client=client, poll_interval_s=0)
    with pytest.raises(VideoConfigurationError):
        p.generate(VideoRequest(brief=_brief()))  # _brief() carries one ingredient


def test_veo_generate_maps_auth_error_keyfree():
    class _BoomModels:
        def generate_videos(self, **_):
            raise RuntimeError("boom with sk-secret in it")

    class _BoomClient:
        def __init__(self):
            self.models = _BoomModels()

    # give the raised error a 403 code so it maps to VideoAuthError
    class _AuthErr(RuntimeError):
        code = 403

    class _AuthModels:
        def generate_videos(self, **_):
            raise _AuthErr("nope")

    client = _BoomClient()
    client.models = _AuthModels()
    p = build_provider("veo", api_key="super-secret", client=client, poll_interval_s=0)
    with pytest.raises(VideoAuthError) as ei:
        p.generate(VideoRequest(brief=_brief_no_ingredients()))
    assert "super-secret" not in str(ei.value)


def test_render_brief_text_includes_global_block_and_shots():
    text = render_brief_text(_brief())
    assert text.splitlines()[0] == "STYLE: warm storybook"
    assert "AUDIO: gentle narrator" in text
    assert 'INGREDIENT[character:child] ref=child.png "child ~6yo"' in text
    assert (
        '[1] child walks to the sink | medium | static | DIALOGUE: "CHILD walks to the sink." | dur=4'
        in text
    )


def test_no_key_in_repr():
    p = build_provider("veo", api_key="super-secret-key")
    assert "super-secret-key" not in repr(p)
