import pytest

from wegofwd_video.errors import VideoConfigurationError
from wegofwd_video.registry import (
    VIDEO_PROVIDER_REGISTRY,
    available_providers,
    resolve_role,
    validate_selection,
)


def test_registry_has_the_known_providers():
    assert set(available_providers()) == {
        "veo",
        "deterministic-renderer",
        "runway",
        "kling",
    }


def test_veo_is_the_verified_default():
    spec = VIDEO_PROVIDER_REGISTRY["veo"]
    assert spec.default_model == "veo-3.1"
    assert spec.model_verified is True
    assert spec.capabilities.native_audio is True
    assert spec.capabilities.reference_images == 4


def test_validate_selection_defaults_model():
    assert validate_selection("veo") == ("veo", "veo-3.1")
    assert validate_selection("veo", "veo-3.1-fast") == ("veo", "veo-3.1-fast")


def test_validate_selection_unknown_provider():
    with pytest.raises(VideoConfigurationError):
        validate_selection("sora")


def test_resolve_role():
    assert resolve_role("narrative-video") == ("veo", "veo-3.1")
    assert resolve_role("safety-render")[0] == "deterministic-renderer"


def test_resolve_role_unknown():
    with pytest.raises(VideoConfigurationError):
        resolve_role("hologram")


def test_placeholders_are_marked_unverified():
    # Honesty convention (matches wegofwd-llm): unwired vendors stay unverified.
    assert VIDEO_PROVIDER_REGISTRY["runway"].model_verified is False
    assert VIDEO_PROVIDER_REGISTRY["kling"].model_verified is False
