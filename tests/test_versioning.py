from wegofwd_video.contract import VIDEO_CONTRACT_VERSION
from wegofwd_video.registry import VIDEO_PROVIDER_REGISTRY, provenance


def test_provenance_shape_for_veo():
    prov = provenance("veo", seed=9071)
    assert prov == {
        "stage": "video",
        "engine": "wegofwd-video",
        "provider": "veo",
        "model": "veo-3.1",
        "model_verified": True,
        "integration_version": VIDEO_PROVIDER_REGISTRY["veo"].integration_version,
        "contract_version": VIDEO_CONTRACT_VERSION,
        "seed": 9071,
    }


def test_provenance_surfaces_unverified_honestly():
    prov = provenance("kling", "kling-3.0")
    assert prov["provider"] == "kling"
    assert prov["model"] == "kling-3.0"
    # registry kling model is UNVERIFIED — provenance must not claim otherwise
    assert prov["model_verified"] is False


def test_provenance_defaults_model_and_seed():
    prov = provenance("deterministic-renderer")
    assert prov["model"] == "blender-grease-pencil-v2"
    assert prov["seed"] is None
