"""The first-run harness stays runnable.

A verification script that has quietly stopped working is worse than none — it
is reached for exactly when something else is already suspected. These exercise
everything up to the network call, so the harness cannot rot unnoticed.

No API key and no network: the dry-run path is the whole surface under test.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

import wegofwd_video as wv

HARNESS = Path(__file__).resolve().parents[1] / "scripts" / "first_veo_run.py"


def _load() -> ModuleType:
    """Import the script by path — `scripts/` is not a package."""
    spec = importlib.util.spec_from_file_location("first_veo_run", HARNESS)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["first_veo_run"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def harness() -> ModuleType:
    return _load()


class TestDemoBrief:
    def test_the_brief_is_within_the_provider_capabilities(self, harness: ModuleType) -> None:
        """The default must not be refused before it is ever dispatched."""
        provider_id, _ = wv.resolve_role(harness.ROLE)
        brief = harness.build_demo_brief()
        wv.assert_brief_within_capabilities(
            provider_id,
            resolution="1080p",
            aspect="16:9",
            duration_s=6.0,
            ingredients=len(brief.ingredients),
        )

    def test_the_brief_carries_no_ingredients(self, harness: ModuleType) -> None:
        """Veo's reference-image path is explicitly unimplemented and raises."""
        assert harness.build_demo_brief().ingredients == ()

    def test_the_brief_has_a_shot_with_a_prompt(self, harness: ModuleType) -> None:
        brief = harness.build_demo_brief()
        assert brief.shots and brief.shots[0].prompt.strip()


class TestDryRun:
    def test_dry_run_builds_a_request_without_a_key(
        self, harness: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        """The point of --dry-run: see what would be sent, spend nothing."""
        code = harness.main(["--dry-run", "--key-file", str(tmp_path / "absent.key")])
        assert code == 0
        out = capsys.readouterr().out
        assert "built without contacting the API" in out
        assert "veo-3.1" in out

    def test_a_real_run_without_a_key_fails_before_dispatch(
        self, harness: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Exit 2 for a missing key, distinct from exit 1 for a failed call."""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        assert harness.main(["--key-file", str(tmp_path / "absent.key")]) == 2


class TestKeyHandling:
    def test_the_environment_wins_over_the_file(
        self, harness: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        key_file = tmp_path / "k"
        key_file.write_text("from-file", encoding="utf-8")
        monkeypatch.setenv("GEMINI_API_KEY", "from-env")
        assert harness.read_api_key(key_file) == "from-env"

    def test_the_file_is_read_when_the_environment_is_unset(
        self, harness: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        key_file = tmp_path / "k"
        key_file.write_text("  from-file\n", encoding="utf-8")
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        assert harness.read_api_key(key_file) == "from-file"

    def test_an_empty_file_is_not_a_key(
        self, harness: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty file should read as absent, not as a key of length zero."""
        key_file = tmp_path / "k"
        key_file.write_text("\n", encoding="utf-8")
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        assert harness.read_api_key(key_file) is None
