import pytest

from wegofwd_video.errors import VideoNotAllowedError
from wegofwd_video.registry import available_providers, build_provider, validate_selection

ALL = {"veo", "deterministic-renderer", "runway", "kling"}


def test_available_providers_no_restriction():
    assert set(available_providers()) == ALL
    assert set(available_providers(None)) == ALL


def test_available_providers_restricted_preserves_registry_order():
    assert available_providers({"kling", "veo"}) == ["veo", "kling"]


def test_available_providers_empty_set():
    assert available_providers(set()) == []


def test_available_providers_ignores_unknown_names():
    assert available_providers({"veo", "totally-made-up"}) == ["veo"]


def test_validate_selection_excluded_is_not_allowed():
    with pytest.raises(VideoNotAllowedError):
        validate_selection("kling", allowed={"veo", "deterministic-renderer"})


def test_build_provider_enforces_allowlist_before_construction():
    with pytest.raises(VideoNotAllowedError):
        build_provider("veo", api_key="k", allowed={"deterministic-renderer"})
