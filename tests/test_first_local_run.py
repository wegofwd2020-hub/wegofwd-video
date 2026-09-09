"""The local first-run harness stays runnable — without torch, weights, or hours.

The dry-run path is exercised as-is; the full path is exercised by swapping the
provider factory for one that injects a fake engine, so persistence and the
measurement report are covered too.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

import wegofwd_video as wv

HARNESS = Path(__file__).resolve().parents[1] / "scripts" / "first_local_run.py"
DATA = Path(__file__).parent / "data"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("first_local_run", HARNESS)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["first_local_run"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def harness() -> ModuleType:
    return _load()


class _FakeEngine:
    def encode_prompt(self, prompt, negative):
        return {"prompt": prompt, "negative": negative}

    def render(self, plan, embeds, on_step):
        for step in range(1, plan.steps + 1):
            on_step(step, plan.steps, 1.5 * step)
        return ["frame"] * plan.num_frames

    def frames_to_mp4(self, frames, fps):
        return b"\x00\x00\x00\x1cftypisom"  # an MP4-looking header is enough


def test_demo_brief_is_within_capabilities_at_smoke_settings(harness: ModuleType) -> None:
    provider_id, _ = wv.resolve_role(harness.ROLE)
    brief = harness.build_demo_brief()
    wv.assert_brief_within_capabilities(
        provider_id,
        resolution=harness.SMOKE["resolution"],
        aspect="16:9",
        duration_s=harness.SMOKE["duration"],
        ingredients=len(brief.ingredients),
    )
    assert brief.ingredients == ()


def test_dry_run_prints_plan_and_touches_no_backend(harness: ModuleType, capsys) -> None:
    assert harness.main(["--dry-run", "--smoke"]) == 0
    out = capsys.readouterr().out
    assert "448x256, 25 frames" in out and "4 steps" in out
    assert "prompt:" in out and "negative:" in out


def test_brief_file_is_loaded(harness: ModuleType, capsys) -> None:
    rc = harness.main(["--dry-run", "--brief", str(DATA / "sox_lesson_brief.json")])
    assert rc == 0
    assert "stamps the checklist" in capsys.readouterr().out


def test_capability_refusal_exits_1(harness: ModuleType, capsys) -> None:
    assert harness.main(["--dry-run", "--resolution", "1080p"]) == 1
    assert "REFUSED" in capsys.readouterr().out


def test_missing_extra_exits_2_with_install_hint(harness: ModuleType, capsys, monkeypatch) -> None:
    import builtins

    real_import = builtins.__import__

    def no_torch(name, *a, **k):
        if name == "torch":
            raise ImportError("no torch here")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_torch)
    assert harness.main(["--smoke", "--out", "/dev/null"]) == 2
    assert "wegofwd-video[local]" in capsys.readouterr().out


def test_full_run_with_fake_engine_persists_and_reports(
    harness: ModuleType, tmp_path: Path, capsys, monkeypatch
) -> None:
    real_build = wv.build_provider

    def build_with_fake(provider_id, **opts):
        return real_build(provider_id, engine=_FakeEngine(), **opts)

    monkeypatch.setattr(harness.wv, "build_provider", build_with_fake)
    out = tmp_path / "clip.mp4"
    report = tmp_path / "clip.json"
    rc = harness.main(["--smoke", "--out", str(out), "--json", str(report), "--seed", "5"])
    assert rc == 0
    assert out.read_bytes().startswith(b"\x00\x00\x00\x1cftyp")
    data = json.loads(report.read_text())
    assert data["plan"]["width"] == 448 and data["plan"]["num_frames"] == 25
    assert data["seconds_per_step"] == 1.5
    assert data["result"]["seed"] == 5
    assert data["provenance"]["provider"] == "local-diffusion"
    assert data["provenance"]["model_verified"] is False
    printed = capsys.readouterr().out
    assert "s/step" in printed and "stored" in printed
