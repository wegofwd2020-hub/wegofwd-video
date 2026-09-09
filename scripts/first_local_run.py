#!/usr/bin/env python3
"""Exercise the local-diffusion provider on THIS machine and measure it.

The provider was written from the diffusers documentation, on a box that
cannot run it (no weights, no hours to spare). Like `first_veo_run.py`, this
script's job is to find the small specific ways a docs-written integration is
wrong — a renamed pipeline kwarg, a checkpoint file that moved — and to report
the stage that failed rather than a bare traceback.

It is also the proof-of-concept's instrument. On a CPU the only question that
matters is *how long*, so every run prints seconds-per-step, total wall clock,
and peak resident memory, and `--json` writes them somewhere a table can be
built from later.

Usage::

    python scripts/first_local_run.py --dry-run           # plan only, no torch
    python scripts/first_local_run.py --smoke             # 256p, 1 s, 4 steps
    python scripts/first_local_run.py --resolution 480p --duration 3 --steps 8 \\
        --out /tmp/ltx-480p.mp4 --json /tmp/ltx-480p.json

The first real run downloads ~10 GB from Hugging Face into the cache
(`HF_HOME`, default ~/.cache/huggingface). Nothing is billable.

Expectation-setting for a 2012 quad-core (Ivy Bridge, AVX only, fp32):
the smoke setting is minutes, 320p/3 s is tens of minutes, 480p/5 s is hours.
Measure before extrapolating — that is what this script is for.
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
import time
from dataclasses import asdict
from pathlib import Path

import wegofwd_video as wv
from wegofwd_video.providers.local_diffusion import (
    DEFAULT_GUIDANCE,
    DEFAULT_STEPS,
    DEFAULT_TRANSFORMER_FILE,
    LocalDiffusionProvider,
)

ROLE = "local-preview"
SMOKE = {"resolution": "256p", "duration": 1.0, "steps": 4}


def _stage(label: str, detail: str = "") -> None:
    print(f"[ {label:<10} ] {detail}", flush=True)


def build_demo_brief() -> wv.VideoBrief:
    """One compliance-flavoured shot — the same brief first_veo_run uses, so the
    two providers can be compared on identical input."""
    return wv.VideoBrief(
        global_style=(
            "clean corporate explainer, soft neutral lighting, muted palette, no on-screen text"
        ),
        global_negative="text overlays, logos, watermarks, distorted faces",
        shots=(
            wv.Shot(
                scene_index=0,
                prompt=(
                    "A person at a desk in a modern office reviews a document, "
                    "then looks up thoughtfully."
                ),
                shot_type="medium",
                camera_move="slow push-in",
                lighting="soft daylight from a window",
                duration_s=6.0,
            ),
        ),
    )


def load_brief(path: Path) -> wv.VideoBrief:
    """A brief from a JSON file in the schema/video_brief.v1.json shape."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return wv.VideoBrief(
        global_style=raw["global_style"],
        global_negative=raw.get("global_negative", ""),
        audio_direction=raw.get("audio_direction", ""),
        ingredients=tuple(wv.Ingredient(**i) for i in raw.get("ingredients", [])),
        shots=tuple(wv.Shot(**{**s, "sfx": tuple(s.get("sfx", ()))}) for s in raw["shots"]),
    )


def peak_rss_gb() -> float:
    """Peak resident set size of this process, in GB (Linux reports KB)."""
    kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return kb / 1024**2 if platform.system() == "Linux" else kb / 1024**3


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=Path("local-first-run.mp4"))
    parser.add_argument("--json", type=Path, default=None, help="write the measurements here")
    parser.add_argument("--brief", type=Path, default=None, help="a video_brief.v1 JSON file")
    parser.add_argument("--resolution", default="320p")
    parser.add_argument("--aspect", default="16:9")
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--guidance", type=float, default=DEFAULT_GUIDANCE)
    parser.add_argument("--seed", type=int, default=9071)
    parser.add_argument("--threads", type=int, default=None, help="torch CPU threads")
    parser.add_argument("--timeout", type=float, default=4 * 3600, help="seconds")
    parser.add_argument(
        "--model",
        default=None,
        help="override the role's HF repo id / local diffusers folder (e.g. a tiny test model)",
    )
    parser.add_argument(
        "--transformer-file",
        default=DEFAULT_TRANSFORMER_FILE,
        help="single-file checkpoint to swap in; 'none' keeps the repo's own transformer",
    )
    parser.add_argument(
        "--no-preflight", action="store_true", help="skip the RAM check (tiny test models)"
    )
    parser.add_argument("--text-encoder-dtype", default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    parser.add_argument("--smoke", action="store_true", help=f"use the smallest settings {SMOKE}")
    parser.add_argument("--dry-run", action="store_true", help="build the plan, load nothing")
    args = parser.parse_args(argv)
    if args.smoke:
        args.resolution, args.duration, args.steps = (
            SMOKE["resolution"],
            SMOKE["duration"],
            SMOKE["steps"],
        )

    # 1. Role -> provider; no model id hardcoded here either.
    provider_id, model = wv.resolve_role(ROLE)
    if args.model:
        model = args.model
    transformer_file = None if args.transformer_file.lower() == "none" else args.transformer_file
    _stage("role", f"{ROLE} -> {provider_id} / {model} ({transformer_file or 'repo transformer'})")

    brief = load_brief(args.brief) if args.brief else build_demo_brief()

    # 2. Pre-check before anything heavy loads.
    try:
        wv.assert_brief_within_capabilities(
            provider_id,
            resolution=args.resolution,
            aspect=args.aspect,
            duration_s=args.duration,
            ingredients=len(brief.ingredients),
        )
        _stage("capability", f"{args.resolution} {args.aspect} {args.duration}s accepted")
    except wv.VideoCapabilityError as exc:
        _stage("capability", f"REFUSED — {exc}")
        return 1

    request = wv.VideoRequest(
        brief=brief,
        resolution=args.resolution,
        aspect_ratio=args.aspect,
        fps=args.fps,
        target_duration_s=args.duration,
        seed=args.seed,
        audio=False,
    )

    step_log: list[tuple[int, int, float]] = []

    def on_progress(step: int, total: int, elapsed: float) -> None:
        step_log.append((step, total, elapsed))
        per_step = elapsed / step
        remaining = per_step * (total - step)
        _stage("step", f"{step}/{total}  {per_step:6.1f} s/step  ~{remaining / 60:5.1f} min left")

    provider = wv.build_provider(
        provider_id,
        model=model,
        steps=args.steps,
        guidance=args.guidance,
        timeout=args.timeout,
        on_progress=on_progress,
        transformer_file=transformer_file,
        dtype=args.dtype,
        text_encoder_dtype=args.text_encoder_dtype,
        threads=args.threads,
        preflight=not args.no_preflight,
    )
    assert isinstance(provider, LocalDiffusionProvider)
    plan = provider.build_request(request)
    _stage(
        "plan",
        f"{plan.width}x{plan.height}, {plan.num_frames} frames @ {plan.fps} fps, "
        f"{plan.steps} steps, guidance {plan.guidance}, seed {plan.seed}",
    )
    if args.dry_run:
        print("\n  prompt:\n    " + plan.prompt)
        print("  negative:\n    " + (plan.negative or "(none)"))
        return 0

    # 3. The real thing: T5 encode -> free -> transformer render -> mp4.
    _stage("generate", "encoding prompt, then rendering (first run downloads weights)…")
    started = time.monotonic()
    try:
        result = provider.generate(request)
    except wv.VideoConfigurationError as exc:
        _stage("generate", f"CONFIG — {exc}")
        if exc.__cause__ is not None:
            print(f"\n  cause: {type(exc.__cause__).__name__}: {exc.__cause__}", file=sys.stderr)
        return 2
    except wv.VideoTimeoutError as exc:
        _stage("generate", f"TIMEOUT — {exc}")
        return 3
    except wv.VideoError as exc:
        _stage("generate", f"FAILED — {type(exc).__name__}: {exc}")
        if exc.__cause__ is not None:
            print(f"\n  cause: {type(exc.__cause__).__name__}: {exc.__cause__}", file=sys.stderr)
        return 1
    elapsed = time.monotonic() - started
    _stage("generated", f"{elapsed / 60:.1f} min wall clock, peak RSS {peak_rss_gb():.1f} GB")
    if result.raw and result.raw.get("transformer_fallback_reason"):
        _stage(
            "transformer",
            "FELL BACK to the repo's own (non-distilled) transformer — "
            f"{result.raw['transformer_fallback_reason']}; distilled step counts no longer apply",
        )

    # 4. Persist — the library returns bytes and stores nothing.
    if not result.asset_bytes:
        _stage("stored", "NOTHING — asset_bytes was empty")
        return 1
    args.out.write_bytes(result.asset_bytes)
    _stage("stored", f"{len(result.asset_bytes)} bytes -> {args.out}")

    measurements = {
        "host": platform.node(),
        "cpu": platform.processor() or platform.machine(),
        "plan": asdict(plan),
        "wall_clock_s": round(elapsed, 1),
        "seconds_per_step": result.raw.get("seconds_per_step") if result.raw else None,
        "transformer_used": result.raw.get("transformer_used") if result.raw else None,
        "peak_rss_gb": round(peak_rss_gb(), 2),
        "result": {
            "duration_s": result.duration_s,
            "resolution": result.resolution,
            "bytes": len(result.asset_bytes),
            "seed": result.seed,
        },
        "provenance": wv.provenance(provider_id, model, seed=result.seed),
    }
    print("\n  measurements:")
    for key in ("wall_clock_s", "seconds_per_step", "peak_rss_gb"):
        print(f"    {key}: {measurements[key]}")
    print(f"  provenance: {measurements['provenance']}")
    if args.json:
        args.json.write_text(json.dumps(measurements, indent=2), encoding="utf-8")
        _stage("json", str(args.json))
    print(
        "\n  Next: if this clip looks right, flip VIDEO_PROVIDER_REGISTRY['local-diffusion']"
        ".model_verified to True and record the numbers in docs/local-diffusion-cpu-poc.md."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
