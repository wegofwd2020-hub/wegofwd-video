"""local-diffusion provider: request shaping, the engine seam, error mapping.

No torch, no diffusers, no weights: every test drives the provider through a
FakeEngine (the same injection pattern as the Veo tests' fake SDK client), and
briefs come from tests/data/*.json — the on-disk shape of schema/video_brief.v1.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import wegofwd_video as wv
from wegofwd_video.errors import (
    VideoCapabilityError,
    VideoConfigurationError,
    VideoResponseError,
    VideoTimeoutError,
)
from wegofwd_video.providers.local_diffusion import (
    DEFAULT_STEPS,
    DEFAULT_TRANSFORMER_FILE,
    LocalDiffusionProvider,
    RenderPlan,
    render_negative,
    render_prompt,
    snap_frames,
    snap_size,
    total_duration,
)

DATA = Path(__file__).parent / "data"


# ── fixtures: briefs from the sample JSON ─────────────────────────────────────
def load_brief(name: str) -> wv.VideoBrief:
    raw = json.loads((DATA / name).read_text(encoding="utf-8"))
    return wv.VideoBrief(
        global_style=raw["global_style"],
        global_negative=raw.get("global_negative", ""),
        audio_direction=raw.get("audio_direction", ""),
        ingredients=tuple(wv.Ingredient(**i) for i in raw.get("ingredients", [])),
        shots=tuple(wv.Shot(**{**s, "sfx": tuple(s.get("sfx", ()))}) for s in raw["shots"]),
    )


@pytest.fixture
def sox_brief() -> wv.VideoBrief:
    return load_brief("sox_lesson_brief.json")


@pytest.fixture
def single_brief() -> wv.VideoBrief:
    return load_brief("single_shot_brief.json")


# ── a fake engine: records calls, drives the step callback, returns bytes ─────
class FakeEngine:
    def __init__(self, *, mp4: bytes = b"MP4DATA", step_elapsed: float = 0.0):
        self.mp4 = mp4
        self.step_elapsed = step_elapsed
        self.encoded: list[tuple[str, str]] = []
        self.rendered: list[RenderPlan] = []
        self.fps_seen: int | None = None

    def encode_prompt(self, prompt, negative):
        self.encoded.append((prompt, negative))
        return {"prompt": prompt, "negative": negative}

    def render(self, plan, embeds, on_step):
        assert embeds["prompt"] == plan.prompt
        self.rendered.append(plan)
        for step in range(1, plan.steps + 1):
            on_step(step, plan.steps, self.step_elapsed * step)
        return ["frame"] * plan.num_frames

    def frames_to_mp4(self, frames, fps):
        self.fps_seen = fps
        assert len(frames) == self.rendered[-1].num_frames
        return self.mp4


# ── pure shaping ──────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    ("resolution", "aspect", "expected"),
    [
        ("256p", "16:9", (448, 256)),
        ("320p", "16:9", (576, 320)),
        ("480p", "16:9", (864, 480)),
        ("720p", "16:9", (1248, 704)),  # 720 is not on the 32-grid; LTX documents 704
        ("480p", "9:16", (256, 480)),
        ("320p", "1:1", (320, 320)),
        ("720p", "9:16", (384, 704)),
    ],
)
def test_snap_size_multiples_of_32(resolution, aspect, expected):
    w, h = snap_size(resolution, aspect)
    assert (w, h) == expected
    assert w % 32 == 0 and h % 32 == 0


def test_snap_size_rejects_unknown_labels():
    with pytest.raises(VideoConfigurationError):
        snap_size("1080p", "16:9")
    with pytest.raises(VideoConfigurationError):
        snap_size("480p", "4:3")


@pytest.mark.parametrize(
    ("duration", "fps", "frames"),
    [(1.0, 24, 25), (2.0, 24, 49), (0.1, 24, 9), (5.0, 24, 121), (10.0, 24, 241)],
)
def test_snap_frames_is_8k_plus_1(duration, fps, frames):
    n = snap_frames(duration, fps)
    assert n == frames
    assert (n - 1) % 8 == 0


def test_snap_frames_rejects_nonpositive():
    with pytest.raises(VideoConfigurationError):
        snap_frames(0, 24)


def test_render_prompt_is_prose_without_dialogue(sox_brief):
    text = render_prompt(sox_brief)
    assert text.startswith("Clean corporate explainer")
    assert "reviews a printed control checklist, then looks up thoughtfully." in text
    assert "Then, the same person stamps" in text
    assert "Medium, slow push-in, soft daylight from a window." in text
    assert "Every completion is recorded" not in text  # no audio track -> no dialogue
    assert "DIALOGUE" not in text and "STYLE:" not in text  # not the Veo block format


def test_render_negative_merges_and_dedupes(sox_brief):
    assert render_negative(sox_brief) == (
        "text overlays, logos, watermarks, distorted faces, blurry hands"
    )


def test_total_duration_prefers_request_then_shots(sox_brief, single_brief):
    assert total_duration(wv.VideoRequest(brief=sox_brief)) == 4.0
    assert total_duration(wv.VideoRequest(brief=sox_brief, target_duration_s=3)) == 3.0
    bare = wv.VideoBrief(global_style="x", shots=(wv.Shot(scene_index=0, prompt="p"),))
    assert total_duration(wv.VideoRequest(brief=bare)) == 2.0


# ── registry + construction ───────────────────────────────────────────────────
def test_registered_with_honest_capabilities():
    spec = wv.VIDEO_PROVIDER_REGISTRY["local-diffusion"]
    assert spec.base_url is None and spec.managed_env_key == ""
    assert spec.model_verified is False  # flips after the first real clip
    assert spec.capabilities.native_audio is False
    assert spec.capabilities.reference_images == 0
    assert spec.capabilities.deterministic is True
    assert "256p" in spec.capabilities.resolutions
    assert wv.resolve_role("local-preview") == ("local-diffusion", "Lightricks/LTX-Video-0.9.5")


def test_build_provider_needs_no_key_and_rejects_one():
    p = wv.build_provider("local-diffusion", engine=FakeEngine())
    assert isinstance(p, LocalDiffusionProvider)
    assert p.provider_id == "local-diffusion"
    assert p.model == "Lightricks/LTX-Video-0.9.5"
    assert p.transformer_file == DEFAULT_TRANSFORMER_FILE
    with pytest.raises(VideoConfigurationError):
        wv.build_provider("local-diffusion", api_key="not-needed")


def test_build_provider_validates_options():
    with pytest.raises(VideoConfigurationError):
        wv.build_provider("local-diffusion", steps=0, engine=FakeEngine())
    with pytest.raises(VideoConfigurationError):
        wv.build_provider("local-diffusion", timeout=0, engine=FakeEngine())


def test_capability_precheck_refuses_1080p_and_ingredients():
    with pytest.raises(VideoCapabilityError):
        wv.assert_brief_within_capabilities(
            "local-diffusion", resolution="1080p", aspect="16:9", duration_s=4, ingredients=0
        )
    with pytest.raises(VideoCapabilityError):
        wv.assert_brief_within_capabilities(
            "local-diffusion", resolution="480p", aspect="16:9", duration_s=4, ingredients=1
        )
    with pytest.raises(VideoCapabilityError):
        wv.assert_brief_within_capabilities(
            "local-diffusion", resolution="480p", aspect="16:9", duration_s=11, ingredients=0
        )


# ── build_request ─────────────────────────────────────────────────────────────
def test_build_request_snaps_geometry_and_pins_seed(sox_brief):
    p = wv.build_provider("local-diffusion", engine=FakeEngine(), steps=6, guidance=1.0)
    plan = p.build_request(wv.VideoRequest(brief=sox_brief, resolution="320p", seed=42))
    assert (plan.width, plan.height) == (576, 320)
    assert plan.num_frames == 97  # 4 s * 24 fps = 96 -> 97
    assert plan.fps == 24 and plan.steps == 6 and plan.guidance == 1.0
    assert plan.seed == 42
    assert plan.negative.startswith("text overlays")


def test_build_request_picks_a_seed_when_none_given(single_brief):
    p = wv.build_provider("local-diffusion", engine=FakeEngine())
    plan = p.build_request(wv.VideoRequest(brief=single_brief, resolution="256p"))
    assert isinstance(plan.seed, int) and 0 <= plan.seed < 2**31


def test_build_request_rejects_ingredients_and_overlong(single_brief):
    p = wv.build_provider("local-diffusion", engine=FakeEngine())
    with pytest.raises(VideoConfigurationError):
        p.build_request(wv.VideoRequest(brief=load_brief("brief_with_ingredient.json")))
    with pytest.raises(VideoConfigurationError):
        p.build_request(wv.VideoRequest(brief=single_brief, target_duration_s=11))


# ── generate through the engine seam ──────────────────────────────────────────
def test_generate_encodes_renders_and_returns_bytes(sox_brief):
    engine = FakeEngine(step_elapsed=2.5)
    progress: list[tuple[int, int, float]] = []
    p = wv.build_provider(
        "local-diffusion", engine=engine, steps=4, on_progress=lambda *a: progress.append(a)
    )
    req = wv.VideoRequest(brief=sox_brief, resolution="256p", seed=7, target_duration_s=1)
    result = p.generate(req)

    assert result.provider_id == "local-diffusion"
    assert result.model == "Lightricks/LTX-Video-0.9.5"
    assert result.asset_bytes == b"MP4DATA" and result.asset_uri is None
    assert result.has_audio is False and result.c2pa_signed is False and result.watermark == ""
    assert result.seed == 7
    assert result.duration_s == pytest.approx(25 / 24, abs=1e-3)
    assert result.resolution == "256p"
    assert engine.encoded == [(render_prompt(sox_brief), render_negative(sox_brief))]
    assert engine.rendered[0].steps == 4 and engine.fps_seen == 24
    assert [s for s, _, _ in progress] == [1, 2, 3, 4]
    assert result.raw["seconds_per_step"] == 2.5 and result.raw["num_frames"] == 25
    assert result.raw["transformer_file"] == DEFAULT_TRANSFORMER_FILE


def test_generate_defaults_to_distilled_step_count(single_brief):
    engine = FakeEngine()
    wv.build_provider("local-diffusion", engine=engine).generate(
        wv.VideoRequest(brief=single_brief, resolution="256p")
    )
    assert engine.rendered[0].steps == DEFAULT_STEPS


def test_generate_enforces_wall_clock_budget(single_brief, monkeypatch):
    # advance a fake clock by 100 s per step so a 150 s budget trips on step 2
    import wegofwd_video.providers.local_diffusion as mod

    clock = {"t": 0.0}

    def fake_monotonic():
        clock["t"] += 100.0
        return clock["t"]

    monkeypatch.setattr(mod.time, "monotonic", fake_monotonic)
    p = wv.build_provider("local-diffusion", engine=FakeEngine(), steps=8, timeout=150)
    with pytest.raises(VideoTimeoutError) as ei:
        p.generate(wv.VideoRequest(brief=single_brief, resolution="256p"))
    assert "budget" in str(ei.value)


def test_generate_rejects_empty_output(single_brief):
    p = wv.build_provider("local-diffusion", engine=FakeEngine(mp4=b""))
    with pytest.raises(VideoResponseError):
        p.generate(wv.VideoRequest(brief=single_brief, resolution="256p"))


def _hub_transport_error() -> Exception:
    """An exception whose class lives in httpx (the Hub's transport), not OSError."""
    cls = type("ProxyError", (Exception,), {"__module__": "httpx"})
    return cls("403 Forbidden")


class _BoomEngine(FakeEngine):
    def __init__(self, exc: Exception):
        super().__init__()
        self.exc = exc

    def render(self, plan, embeds, on_step):
        raise self.exc


@pytest.mark.parametrize(
    ("exc", "expected", "hint"),
    [
        (MemoryError(), VideoResponseError, "out of memory"),
        (RuntimeError("DefaultCPUAllocator: not enough memory"), VideoResponseError, "memory"),
        (ImportError("No module named diffusers"), VideoConfigurationError, "[local]"),
        (OSError("Lightricks/LTX-Video is not a local folder"), VideoConfigurationError, "weights"),
        (_hub_transport_error(), VideoConfigurationError, "weights"),
        (ValueError("num_frames must be 8k+1"), VideoConfigurationError, "geometry"),
        (RuntimeError("shape mismatch"), VideoResponseError, "render failed"),
    ],
)
def test_generate_maps_backend_errors(single_brief, exc, expected, hint):
    p = wv.build_provider("local-diffusion", engine=_BoomEngine(exc))
    with pytest.raises(expected) as ei:
        p.generate(wv.VideoRequest(brief=single_brief, resolution="256p"))
    assert hint in str(ei.value)
    assert ei.value.__cause__ is exc  # chained: no key on this path, keep the traceback


def test_provenance_is_honest_about_verification():
    prov = wv.provenance("local-diffusion", seed=3)
    assert prov["provider"] == "local-diffusion"
    assert prov["model"] == "Lightricks/LTX-Video-0.9.5"
    assert prov["model_verified"] is False
    assert prov["seed"] == 3


def test_missing_extra_maps_to_configuration_error(single_brief, monkeypatch):
    """Without an injected engine the real one is built; with torch absent the
    first call must surface the install hint, not a bare ImportError."""
    import builtins

    real_import = builtins.__import__

    def no_torch(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("No module named 'torch'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_torch)
    p = wv.build_provider("local-diffusion")
    with pytest.raises(VideoConfigurationError) as ei:
        p.generate(wv.VideoRequest(brief=single_brief, resolution="256p"))
    assert "wegofwd-video[local]" in str(ei.value)
