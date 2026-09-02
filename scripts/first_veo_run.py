#!/usr/bin/env python3
"""Exercise the Veo provider against the live API.

The `veo` provider was written from Google's documentation and, per the README,
is "awaiting first real run". Docs-written SDK integrations tend to be wrong in
small specific ways — a renamed config field, a different operation-polling
shape, a file handle that is not where the example says. This script finds out
which, and reports the stage that failed rather than a bare traceback.

It is also the answer to a question every consumer of this library has: *how do
I check this provider actually works before I wire it into my product?*

Usage::

    python scripts/first_veo_run.py --dry-run          # everything but the call
    python scripts/first_veo_run.py --out /tmp/veo.mp4 # the real thing

The key is read from ``GEMINI_API_KEY``, else ``~/.config/wegofwd/gemini.key``.
It is never printed, logged, or placed in an exception — the same rule the
package holds itself to.

**A real run is billable.** The default brief is one short shot at the smallest
sensible settings.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import wegofwd_video as wv

DEFAULT_KEY_FILE = Path.home() / ".config" / "wegofwd" / "gemini.key"
ROLE = "narrative-video"


def _stage(label: str, detail: str = "") -> None:
    print(f"[ {label:<9} ] {detail}", flush=True)


def read_api_key(key_file: Path) -> str | None:
    """Env first, then a mode-600 file. Never returned to the caller's logs."""
    env = os.environ.get("GEMINI_API_KEY", "").strip()
    if env:
        return env
    if key_file.is_file():
        return key_file.read_text(encoding="utf-8").strip() or None
    return None


def build_demo_brief() -> wv.VideoBrief:
    """One short compliance-flavoured shot.

    Deliberately minimal: the point is to learn whether the provider round-trips
    at all, and every extra field is another thing that could fail for reasons
    unrelated to the integration.
    """
    return wv.VideoBrief(
        global_style=(
            "clean corporate explainer, soft neutral lighting, muted palette, no on-screen text"
        ),
        global_negative="text overlays, logos, watermarks, distorted faces",
        audio_direction="calm professional narrator, measured pace",
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
                dialogue="Every completion is recorded as evidence.",
                duration_s=6.0,
            ),
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("veo-first-run.mp4"))
    parser.add_argument("--resolution", default="1080p")
    parser.add_argument("--aspect", default="16:9")
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=9071)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve, pre-check and build the request without calling the API",
    )
    parser.add_argument("--key-file", type=Path, default=DEFAULT_KEY_FILE)
    args = parser.parse_args(argv)

    # 1. Role -> provider, so no model id is hardcoded here either.
    provider_id, model = wv.resolve_role(ROLE)
    _stage("role", f"{ROLE} -> {provider_id} / {model}")

    brief = build_demo_brief()

    # 2. Pre-check BEFORE dispatch. A brief the provider cannot serve should cost
    #    nothing to discover.
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
        target_duration_s=args.duration,
        seed=args.seed,
        audio=True,
    )

    api_key = read_api_key(args.key_file)
    if not api_key and not args.dry_run:
        _stage("key", "MISSING")
        print(
            f"\nSet GEMINI_API_KEY, or write the key to {args.key_file}:\n"
            f"  mkdir -p {args.key_file.parent} && chmod 700 {args.key_file.parent}\n"
            f"  printf '%s' 'YOUR_KEY' > {args.key_file} && chmod 600 {args.key_file}",
            file=sys.stderr,
        )
        return 2
    _stage("key", "present" if api_key else "not needed (dry run)")

    if args.dry_run:
        provider = wv.build_provider(provider_id, api_key="dry-run-placeholder")
        payload = provider.build_request(request)
        _stage("request", "built without contacting the API")
        print("\n  SDK kwargs the provider would send:")
        for key, value in sorted(payload.items()):
            rendered = str(value)
            if len(rendered) > 300:
                rendered = rendered[:300] + " …"
            print(f"    {key}: {rendered}")
        return 0

    # 3. The real call. generate() blocks: submit, poll, download (ADR-026 D1 —
    #    the library never orchestrates, so the wait is ours).
    provider = wv.build_provider(provider_id, api_key=api_key)
    _stage("submit", "calling generate_videos and polling until done…")
    started = time.monotonic()
    try:
        result = provider.generate(request)
    except wv.VideoAuthError as exc:
        _stage("submit", f"AUTH REFUSED — {exc}")
        print(
            "\n  The key is reaching Google but is not accepted for Veo. Veo is "
            "reached through the Gemini API, not the consumer app — check the "
            "key has that access.",
            file=sys.stderr,
        )
        return 1
    except wv.VideoError as exc:
        _stage("submit", f"FAILED — {type(exc).__name__}: {exc}")
        return 1

    elapsed = time.monotonic() - started
    _stage("generated", f"{elapsed:.0f}s elapsed")

    # 4. Persist. The library returns bytes or a URI and stores nothing itself.
    if result.asset_bytes:
        args.out.write_bytes(result.asset_bytes)
        _stage("stored", f"{len(result.asset_bytes)} bytes -> {args.out}")
    elif result.asset_uri:
        _stage("stored", f"no bytes returned; vendor URI {result.asset_uri}")
    else:
        _stage("stored", "NOTHING — neither asset_bytes nor asset_uri was set")
        return 1

    print("\n  result:")
    for field_name in ("duration_s", "resolution", "has_audio", "c2pa_signed", "watermark", "seed"):
        print(f"    {field_name}: {getattr(result, field_name)}")
    print(f"\n  provenance: {wv.provenance(provider_id, model, seed=args.seed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
