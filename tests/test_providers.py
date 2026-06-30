import pytest

from wegofwd_video import (
    Ingredient,
    Shot,
    VideoBrief,
    VideoRequest,
    VideoResult,
    build_provider,
)
from wegofwd_video.errors import VideoConfigurationError
from wegofwd_video.providers.veo import VeoProvider, render_brief_text


def _brief():
    return VideoBrief(
        global_style="warm storybook",
        global_negative="no flashing",
        audio_direction="gentle narrator",
        ingredients=(Ingredient(role="character:child", ref="child.png", description="child ~6yo"),),
        shots=(
            Shot(scene_index=1, prompt="child walks to the sink", shot_type="medium",
                 camera_move="static", dialogue="CHILD walks to the sink.", duration_s=4),
        ),
    )


# ── deterministic-renderer: the caller-supplied render fn ─────────────────────
def test_deterministic_renderer_uses_caller_render_fn():
    calls = {}

    def fake_render(req: VideoRequest) -> VideoResult:
        calls["seen"] = req
        return VideoResult(provider_id="deterministic-renderer", model="blender-grease-pencil-v2",
                           asset_uri="media/out.mp4", duration_s=4, resolution="1080p")

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
    with pytest.raises(Exception):
        p.generate(VideoRequest(brief=_brief()))


# ── veo: BYOK construction + pure request shaping (no network) ────────────────
def test_veo_requires_api_key():
    with pytest.raises(VideoConfigurationError):
        build_provider("veo", api_key="")


def test_veo_build_request_shapes_brief_and_params():
    p = build_provider("veo", api_key="byok-key")
    assert isinstance(p, VeoProvider)
    payload = p.build_request(VideoRequest(brief=_brief(), seed=42, resolution="1080p"))
    assert payload["model"] == "veo-3.1"
    assert payload["config"]["seed"] == 42
    assert payload["config"]["generateAudio"] is True
    assert payload["config"]["referenceImages"] == [{"role": "character:child", "ref": "child.png"}]
    assert "STYLE: warm storybook" in payload["prompt"]


def test_veo_generate_is_a_documented_stub():
    p = build_provider("veo", api_key="byok-key")
    with pytest.raises(NotImplementedError):
        p.generate(VideoRequest(brief=_brief()))


def test_render_brief_text_includes_global_block_and_shots():
    text = render_brief_text(_brief())
    assert text.splitlines()[0] == "STYLE: warm storybook"
    assert "AUDIO: gentle narrator" in text
    assert 'INGREDIENT[character:child] ref=child.png "child ~6yo"' in text
    assert '[1] child walks to the sink | medium | static | DIALOGUE: "CHILD walks to the sink." | dur=4' in text


def test_no_key_in_repr():
    p = build_provider("veo", api_key="super-secret-key")
    assert "super-secret-key" not in repr(p)
