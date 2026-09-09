"""
wegofwd_video/providers/local_diffusion.py

Self-hosted open-weight diffusion provider. Runs LTX-Video (Lightricks, 2B
distilled checkpoint by default) through Hugging Face `diffusers` inside the
caller's process — no vendor, no key, no bytes leave the machine. CPU-first: the
first consumer is a 4-core box with 32 GB of RAM and no usable GPU, so the
design goal is *fits and finishes*, not speed.

Why it is shaped the way it is
------------------------------
* **Sequential loading.** LTX's text encoder (T5-XXL, ~4.7B params) and its video
  transformer (~2B) do not both fit in 32 GB at fp32. The engine loads the
  encoder, embeds the prompt, frees it, and only then loads the transformer +
  VAE. `LTXPipeline` accepts pre-computed `prompt_embeds`, which is what makes
  this possible.
* **Injectable engine.** `DiffusionEngine` is the seam the provider talks to. The
  real one (`DiffusersLTXEngine`) imports torch/diffusers lazily so the core
  package keeps zero runtime deps; tests inject a fake and never touch weights.
  Mirrors the injected `client` on the Veo provider.
* **A wall-clock budget, not a poll.** Diffusion runs synchronously; the provider
  enforces `timeout` from inside the pipeline's per-step callback and raises the
  contract's VideoTimeoutError. The same callback reports seconds-per-step, which
  is the number the CPU proof-of-concept exists to measure.
* **Honest capabilities.** 256p-720p, no audio, no reference images, ≤10 s. The
  registry marks the model UNVERIFIED until `scripts/first_local_run.py` has
  produced a clip on real hardware (same convention as Veo).

Not a key path: nothing here handles credentials, so exceptions are chained
(`from exc`) to keep the developer's traceback — the opposite of veo.py, and
deliberate.
"""

from __future__ import annotations

import gc
import os
import random
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from wegofwd_video.contract import (
    VideoBrief,
    VideoCapabilities,
    VideoProvider,
    VideoRequest,
    VideoResult,
)
from wegofwd_video.errors import (
    VideoConfigurationError,
    VideoError,
    VideoResponseError,
    VideoTimeoutError,
)

PROVIDER_ID = "local-diffusion"

#: The 2B distilled checkpoint inside the `Lightricks/LTX-Video` repo. Distilled
#: means 4-10 steps at guidance 1.0 instead of ~40 with CFG — on a CPU that is
#: the difference between "overnight" and "a coffee".
DEFAULT_TRANSFORMER_FILE = "ltxv-2b-0.9.8-distilled.safetensors"
DEFAULT_TRANSFORMER_REPO = "Lightricks/LTX-Video"
#: The diffusers-layout repo the pipeline (VAE, scheduler, T5, transformer config)
#: is loaded from. 0.9.5 is the last 2B release published in that layout.
DEFAULT_MODEL = "Lightricks/LTX-Video-0.9.5"
DEFAULT_STEPS = 8
DEFAULT_GUIDANCE = 1.0
#: LTX's VAE compresses 32x spatially and 8x temporally: width/height must be
#: multiples of 32 and num_frames must be 8k+1.
SIZE_QUANTUM = 32
FRAME_QUANTUM = 8
#: LTX's T5 sequence length.
PROMPT_MAX_TOKENS = 128
#: Four hours: a 480p/5 s clip on a 2012 quad-core is plausibly that long.
DEFAULT_TIMEOUT_S = 4 * 60 * 60

#: Resolution label -> frame height. Labels match the contract's "NNNp" vocabulary
#: so `assert_brief_within_capabilities` works unchanged; the low end exists for
#: CPU smoke tests.
RESOLUTION_HEIGHT: dict[str, int] = {"256p": 256, "320p": 320, "480p": 480, "720p": 720}
ASPECT_RATIO: dict[str, float] = {"16:9": 16 / 9, "9:16": 9 / 16, "1:1": 1.0}

#: Approximate fp32 parameter counts, used only for the pre-flight memory check.
_T5_XXL_PARAMS = 4.76e9
_LTX_2B_PARAMS = 1.92e9
_BYTES_PER_DTYPE = {"float32": 4, "bfloat16": 2, "float16": 2}
_HEADROOM_BYTES = 3 * 1024**3  # activations + VAE + the interpreter itself
#: Exception origins that mean "the weights could not be fetched", not "render failed".
_WEIGHT_TRANSPORT_MODULES = frozenset(
    {"huggingface_hub", "httpx", "httpcore", "requests", "urllib3"}
)


# ── pure request shaping (unit-tested without torch) ─────────────────────────


@dataclass(frozen=True)
class RenderPlan:
    """Everything the engine needs for one clip, already snapped to LTX's grid."""

    prompt: str
    negative: str
    width: int
    height: int
    num_frames: int
    fps: int
    steps: int
    guidance: float
    seed: int


def snap_size(resolution: str, aspect_ratio: str) -> tuple[int, int]:
    """(width, height) for a resolution label + aspect, both multiples of 32.

    The label is nominal: height snaps to the grid ("720p" renders 704 tall, the
    size LTX itself documents) and width follows the aspect, so "480p" 16:9 is
    864x480 (853 rounds up) and 9:16 is 480 tall, 256 wide.
    """
    try:
        height = RESOLUTION_HEIGHT[resolution]
    except KeyError:
        raise VideoConfigurationError(
            f"{PROVIDER_ID} does not render {resolution!r}; one of {tuple(RESOLUTION_HEIGHT)}"
        ) from None
    try:
        ratio = ASPECT_RATIO[aspect_ratio]
    except KeyError:
        raise VideoConfigurationError(
            f"{PROVIDER_ID} does not render aspect {aspect_ratio!r}; one of {tuple(ASPECT_RATIO)}"
        ) from None
    height = max(SIZE_QUANTUM, round(height / SIZE_QUANTUM) * SIZE_QUANTUM)
    width = max(SIZE_QUANTUM, round(height * ratio / SIZE_QUANTUM) * SIZE_QUANTUM)
    return width, height


def snap_frames(duration_s: float, fps: int) -> int:
    """Frame count for a duration, snapped to LTX's 8k+1 rule (never below 9)."""
    if duration_s <= 0 or fps <= 0:
        raise VideoConfigurationError("duration_s and fps must be positive")
    wanted = duration_s * fps
    k = max(1, round((wanted - 1) / FRAME_QUANTUM))
    return FRAME_QUANTUM * k + 1


def _sentence(text: str) -> str:
    text = text.strip()
    if text and text[-1] not in ".!?":
        text += "."
    return text


def render_prompt(brief: VideoBrief) -> str:
    """Flatten a VideoBrief into the single prose prompt a diffusion model wants.

    Diffusion models respond to descriptive prose, not the tagged block Veo
    consumes: style first, then each shot as sentences (subject/action, framing,
    camera, lighting). Dialogue is dropped — there is no audio track — and
    ingredients are not supported (the capability check refuses them upstream).
    Pure and deterministic.
    """
    style = _sentence(brief.global_style)
    parts: list[str] = [style[:1].upper() + style[1:]]
    for i, shot in enumerate(brief.shots):
        text = _sentence(shot.prompt)
        parts.append(("Then, " + text[:1].lower() + text[1:]) if i else text)
        details = [d for d in (shot.shot_type, shot.camera_move, shot.lighting) if d]
        if details:
            parts.append(_sentence(", ".join(details)).capitalize())
    return " ".join(p for p in parts if p)


def render_negative(brief: VideoBrief) -> str:
    """Global negative plus every per-shot negative, comma-joined, de-duplicated."""
    seen: list[str] = []
    for item in (brief.global_negative, *(s.negative for s in brief.shots)):
        for term in (t.strip() for t in item.split(",")):
            if term and term not in seen:
                seen.append(term)
    return ", ".join(seen)


def total_duration(req: VideoRequest, default_s: float = 2.0) -> float:
    """Requested duration, else the shots' sum, else a small default."""
    if req.target_duration_s:
        return float(req.target_duration_s)
    from_shots = sum(s.duration_s for s in req.brief.shots)
    return float(from_shots) if from_shots else default_s


# ── the engine seam ───────────────────────────────────────────────────────────


class DiffusionEngine(Protocol):
    """What the provider needs from a backend. Implemented by DiffusersLTXEngine
    (real) and by test fakes. `on_step(step, total, elapsed_s)` is called after
    every denoising step and may raise to abort the render."""

    def encode_prompt(self, prompt: str, negative: str) -> object:
        """Return an opaque embeddings bundle for `render`."""

    def render(
        self,
        plan: RenderPlan,
        embeds: object,
        on_step: Callable[[int, int, float], None],
    ) -> object:
        """Return an opaque frames object for `frames_to_mp4`."""

    def frames_to_mp4(self, frames: object, fps: int) -> bytes:
        """Encode frames to an H.264 MP4 and return its bytes."""


class DiffusersLTXEngine:
    """The real backend: Hugging Face diffusers + torch, CPU by default.

    All heavy imports are inside methods so importing this module costs nothing
    and the core package keeps zero runtime deps (`pip install wegofwd-video[local]`
    brings them in).
    """

    def __init__(
        self,
        *,
        model: str,
        transformer_file: str | None = DEFAULT_TRANSFORMER_FILE,
        transformer_repo: str = DEFAULT_TRANSFORMER_REPO,
        fallback_to_repo_transformer: bool = True,
        device: str = "cpu",
        dtype: str = "float32",
        text_encoder_dtype: str = "bfloat16",
        threads: int | None = None,
        cache_dir: str | os.PathLike[str] | None = None,
        preflight: bool = True,
        available_bytes_override: int | None = None,
    ) -> None:
        self._model = model
        self._transformer_file = transformer_file
        self._transformer_repo = transformer_repo
        self._fallback_to_repo_transformer = fallback_to_repo_transformer
        self._device = device
        self._dtype = dtype
        self._te_dtype = text_encoder_dtype
        self._threads = threads
        self._cache_dir = str(cache_dir) if cache_dir else None
        self._preflight = preflight
        self._available_bytes_override = available_bytes_override
        self.transformer_used: str = ""
        self.transformer_fallback_reason: str = ""

    # -- helpers ---------------------------------------------------------------
    def _torch(self):
        try:
            import torch  # type: ignore
        except ImportError:
            raise VideoConfigurationError(
                f"{PROVIDER_ID} requires the 'local' extra: pip install wegofwd-video[local] "
                "(torch, diffusers, transformers, accelerate, sentencepiece, imageio-ffmpeg)"
            ) from None
        if self._threads:
            torch.set_num_threads(self._threads)
        return torch

    def _torch_dtype(self, torch, name: str):
        try:
            return getattr(torch, name)
        except AttributeError:
            raise VideoConfigurationError(f"unknown torch dtype {name!r}") from None

    def preflight_memory(self) -> None:
        """Refuse early, with numbers, when RAM cannot hold the larger of the two
        loading phases. Linux-only (reads /proc/meminfo); silently skipped elsewhere."""
        available = self._available_bytes_override or _available_memory_bytes()
        if available is None:
            return
        need_encoder = _T5_XXL_PARAMS * _BYTES_PER_DTYPE.get(self._te_dtype, 4) + _HEADROOM_BYTES
        need_transformer = _LTX_2B_PARAMS * _BYTES_PER_DTYPE.get(self._dtype, 4) + _HEADROOM_BYTES
        need = max(need_encoder, need_transformer)
        if available < need:
            raise VideoConfigurationError(
                f"{PROVIDER_ID}: {available / 1024**3:.1f} GB RAM available, "
                f"{need / 1024**3:.1f} GB needed (text encoder {self._te_dtype}, "
                f"transformer {self._dtype}); use bfloat16 for the text encoder or free memory"
            )

    # -- DiffusionEngine -------------------------------------------------------
    def encode_prompt(self, prompt: str, negative: str) -> object:
        torch = self._torch()
        from transformers import AutoTokenizer, T5EncoderModel  # type: ignore

        if self._preflight:
            self.preflight_memory()
        te_dtype = self._torch_dtype(torch, self._te_dtype)
        tokenizer = AutoTokenizer.from_pretrained(
            self._model, subfolder="tokenizer", cache_dir=self._cache_dir
        )
        encoder = T5EncoderModel.from_pretrained(
            self._model,
            subfolder="text_encoder",
            dtype=te_dtype,
            cache_dir=self._cache_dir,
        ).to(self._device)
        encoder.eval()

        def _embed(tok, enc, text: str) -> tuple[object, object]:
            batch = tok(
                text,
                padding="max_length",
                max_length=PROMPT_MAX_TOKENS,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            ids = batch.input_ids.to(self._device)
            # exactly what LTXPipeline._get_t5_prompt_embeds does: ids only, bool mask
            mask = batch.attention_mask.bool().to(self._device)
            with torch.no_grad():
                hidden = enc(ids)[0]
            return hidden, mask

        embeds = {
            "prompt": _embed(tokenizer, encoder, prompt),
            "negative": _embed(tokenizer, encoder, negative or ""),
        }

        # Free the 4.7B-parameter encoder before the transformer is loaded.
        del encoder, tokenizer
        gc.collect()
        return embeds

    def render(
        self,
        plan: RenderPlan,
        embeds: object,
        on_step: Callable[[int, int, float], None],
    ) -> object:
        torch = self._torch()
        from diffusers import LTXPipeline, LTXVideoTransformer3DModel  # type: ignore

        dtype = self._torch_dtype(torch, self._dtype)
        transformer = None
        self.transformer_used = "repo:transformer"
        if self._transformer_file:
            # The distilled 2B checkpoint lives as a single file in the
            # Lightricks/LTX-Video repo; the config is pinned to the diffusers-layout
            # repo we load the rest of the pipeline from, so diffusers' key-based
            # guess cannot pick a 13B config for a 2B file.
            try:
                transformer = LTXVideoTransformer3DModel.from_single_file(
                    f"https://huggingface.co/{self._transformer_repo}/blob/main/"
                    f"{self._transformer_file}",
                    config=self._model,
                    subfolder="transformer",
                    dtype=dtype,
                    cache_dir=self._cache_dir,
                )
                self.transformer_used = f"{self._transformer_repo}:{self._transformer_file}"
            except Exception as exc:  # fall back, but say so
                if not self._fallback_to_repo_transformer:
                    raise
                self.transformer_fallback_reason = f"{type(exc).__name__}: {exc}"[:300]
        pipe = LTXPipeline.from_pretrained(
            self._model,
            text_encoder=None,
            tokenizer=None,
            dtype=dtype,
            cache_dir=self._cache_dir,
            **({"transformer": transformer} if transformer is not None else {}),
        ).to(self._device)
        pipe.set_progress_bar_config(disable=True)

        bundle = embeds  # type: ignore[assignment]
        p_emb, p_mask = bundle["prompt"]
        n_emb, n_mask = bundle["negative"]
        generator = torch.Generator(device=self._device).manual_seed(plan.seed)
        started = time.monotonic()

        def _callback(_pipe, step: int, _timestep, kwargs: dict) -> dict:
            on_step(step + 1, plan.steps, time.monotonic() - started)
            return kwargs

        with torch.no_grad():
            out = pipe(
                prompt_embeds=p_emb.to(dtype),
                prompt_attention_mask=p_mask,
                negative_prompt_embeds=n_emb.to(dtype),
                negative_prompt_attention_mask=n_mask,
                width=plan.width,
                height=plan.height,
                num_frames=plan.num_frames,
                num_inference_steps=plan.steps,
                guidance_scale=plan.guidance,
                generator=generator,
                output_type="pil",
                callback_on_step_end=_callback,
                callback_on_step_end_tensor_inputs=[],
            )
        frames = out.frames[0]
        del pipe, transformer
        gc.collect()
        return frames

    def frames_to_mp4(self, frames: object, fps: int) -> bytes:
        from diffusers.utils import export_to_video  # type: ignore

        with tempfile.TemporaryDirectory(prefix="wegofwd-video-") as tmp:
            path = Path(tmp) / "clip.mp4"
            export_to_video(frames, str(path), fps=fps)
            return path.read_bytes()


def _available_memory_bytes() -> int | None:
    """MemAvailable from /proc/meminfo, or None when that file does not exist."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


# ── the provider ──────────────────────────────────────────────────────────────


class LocalDiffusionProvider(VideoProvider):
    """LTX-Video under the VideoProvider contract, driven through a DiffusionEngine."""

    def __init__(
        self,
        *,
        model: str,
        capabilities: VideoCapabilities,
        transformer_file: str | None = DEFAULT_TRANSFORMER_FILE,
        steps: int = DEFAULT_STEPS,
        guidance: float = DEFAULT_GUIDANCE,
        timeout: float = DEFAULT_TIMEOUT_S,
        on_progress: Callable[[int, int, float], None] | None = None,
        engine: DiffusionEngine | None = None,
        **engine_opts: object,
    ) -> None:
        if steps < 1:
            raise VideoConfigurationError("steps must be >= 1")
        if timeout <= 0:
            raise VideoConfigurationError("timeout must be positive seconds")
        self.provider_id = PROVIDER_ID
        self.capabilities = capabilities
        self._model = model
        self._transformer_file = transformer_file
        self._steps = steps
        self._guidance = guidance
        self._timeout = timeout
        self._on_progress = on_progress
        self._engine = engine
        self._engine_opts = engine_opts

    @property
    def model(self) -> str:
        return self._model

    @property
    def transformer_file(self) -> str | None:
        """Single-file checkpoint swapped in for the repo's transformer (None = repo's own)."""
        return self._transformer_file

    def build_request(self, req: VideoRequest) -> RenderPlan:
        """VideoRequest -> RenderPlan. Pure: no torch, no weights, unit-testable."""
        if req.brief.ingredients:
            raise VideoConfigurationError(
                f"{PROVIDER_ID} takes no reference images (ingredients); send a brief without them"
            )
        if not req.brief.shots:
            raise VideoConfigurationError("brief has no shots")
        width, height = snap_size(req.resolution, req.aspect_ratio)
        duration = total_duration(req)
        if duration > self.capabilities.max_duration_s:
            raise VideoConfigurationError(
                f"duration {duration}s exceeds {PROVIDER_ID} max {self.capabilities.max_duration_s}s"
            )
        seed = req.seed if req.seed is not None else random.randrange(2**31)  # noqa: S311
        return RenderPlan(
            prompt=render_prompt(req.brief),
            negative=render_negative(req.brief),
            width=width,
            height=height,
            num_frames=snap_frames(duration, req.fps),
            fps=req.fps,
            steps=self._steps,
            guidance=self._guidance,
            seed=seed,
        )

    def generate(self, req: VideoRequest) -> VideoResult:
        plan = self.build_request(req)
        engine = self._engine or self._make_engine()
        started = time.monotonic()
        step_times: list[float] = []

        def _on_step(step: int, total: int, elapsed_s: float) -> None:
            step_times.append(elapsed_s)
            if self._on_progress:
                self._on_progress(step, total, elapsed_s)
            if time.monotonic() - started > self._timeout:
                raise VideoTimeoutError(
                    f"{PROVIDER_ID} exceeded its {self._timeout:.0f}s budget at step {step}/{total}"
                )

        try:
            embeds = engine.encode_prompt(plan.prompt, plan.negative)
            frames = engine.render(plan, embeds, _on_step)
            data = engine.frames_to_mp4(frames, plan.fps)
        except VideoError:
            raise
        except Exception as exc:
            raise self._map_error(exc) from exc

        if not data:
            raise VideoResponseError(f"{PROVIDER_ID} produced an empty video")
        render_s = time.monotonic() - started
        seconds_per_step = (step_times[-1] / len(step_times)) if step_times else None
        return VideoResult(
            provider_id=PROVIDER_ID,
            model=self._model,
            asset_bytes=data,
            duration_s=round(plan.num_frames / plan.fps, 3),
            resolution=req.resolution,
            has_audio=False,
            c2pa_signed=False,
            watermark="",
            seed=plan.seed,
            raw={
                "transformer_file": self._transformer_file,
                "width": plan.width,
                "height": plan.height,
                "num_frames": plan.num_frames,
                "steps": plan.steps,
                "guidance": plan.guidance,
                "render_seconds": round(render_s, 1),
                "seconds_per_step": round(seconds_per_step, 1) if seconds_per_step else None,
                **self._engine_report(engine),
            },
        )

    def _make_engine(self) -> DiffusionEngine:
        return DiffusersLTXEngine(
            model=self._model,
            transformer_file=self._transformer_file,
            **self._engine_opts,  # type: ignore[arg-type]
        )

    @staticmethod
    def _engine_report(engine: object) -> dict:
        """What the real engine actually loaded (fakes report nothing)."""
        used = getattr(engine, "transformer_used", "")
        reason = getattr(engine, "transformer_fallback_reason", "")
        report: dict = {}
        if used:
            report["transformer_used"] = used
        if reason:
            report["transformer_fallback_reason"] = reason
        return report

    @staticmethod
    def _map_error(exc: Exception) -> VideoError:
        """Classify a torch/diffusers/hub failure into the typed hierarchy.

        There is no credential on this path, so the original is chained by the
        caller (`from exc`); the message here still names only the class, plus
        the one hint a CPU user needs most (memory)."""
        name = type(exc).__name__
        text = str(exc).lower()
        if isinstance(exc, MemoryError) or "out of memory" in text or "cpuallocator" in text:
            return VideoResponseError(
                f"{PROVIDER_ID} ran out of memory ({name}); lower the resolution, "
                "shorten the clip, or set text_encoder_dtype='bfloat16'"
            )
        if isinstance(exc, ImportError):
            return VideoConfigurationError(
                f"{PROVIDER_ID} is missing a dependency ({name}); pip install wegofwd-video[local]"
            )
        origin = type(exc).__module__.split(".")[0]
        # Missing files raise OSError; the Hub's transport stack (httpx/requests)
        # raises its own hierarchy — either way the weights did not load.
        if isinstance(exc, OSError) or origin in _WEIGHT_TRANSPORT_MODULES:
            return VideoConfigurationError(
                f"{PROVIDER_ID} could not load model weights ({name}); check the model id, "
                "the Hugging Face cache, and network access for the first download"
            )
        if "must be divisible" in text or "num_frames" in text:
            return VideoConfigurationError(f"{PROVIDER_ID} rejected the render geometry ({name})")
        return VideoResponseError(f"{PROVIDER_ID} render failed ({name})")


__all__ = [
    "DEFAULT_GUIDANCE",
    "DEFAULT_MODEL",
    "DEFAULT_STEPS",
    "DEFAULT_TRANSFORMER_FILE",
    "DEFAULT_TRANSFORMER_REPO",
    "PROVIDER_ID",
    "DiffusersLTXEngine",
    "DiffusionEngine",
    "LocalDiffusionProvider",
    "RenderPlan",
    "render_negative",
    "render_prompt",
    "snap_frames",
    "snap_size",
    "total_duration",
]
