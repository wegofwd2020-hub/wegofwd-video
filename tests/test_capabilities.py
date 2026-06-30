import pytest

from wegofwd_video.errors import VideoCapabilityError
from wegofwd_video.registry import assert_brief_within_capabilities


def test_veo_brief_within_capabilities_passes():
    assert_brief_within_capabilities(
        "veo", resolution="1080p", aspect="16:9", duration_s=12, ingredients=2
    )


def test_kling_rejects_overlong_clip_and_too_many_ingredients():
    with pytest.raises(VideoCapabilityError) as ei:
        assert_brief_within_capabilities(
            "kling", resolution="4k", aspect="16:9", duration_s=20, ingredients=4
        )
    msg = str(ei.value)
    assert "duration 20" in msg
    assert "ingredients" in msg


def test_veo_rejects_unsupported_resolution():
    with pytest.raises(VideoCapabilityError):
        assert_brief_within_capabilities(
            "veo", resolution="8k", aspect="16:9", duration_s=10, ingredients=0
        )


def test_deterministic_renderer_takes_no_ingredients():
    with pytest.raises(VideoCapabilityError):
        assert_brief_within_capabilities(
            "deterministic-renderer",
            resolution="1080p",
            aspect="16:9",
            duration_s=10,
            ingredients=1,
        )
