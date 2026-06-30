"""
wegofwd_video/contract.py

The provider-agnostic VIDEO contract (ADR-026). A typed brief/request/result
plus a capability descriptor, so N video providers (Veo, a local deterministic
renderer, later Kling/Runway) are driven through one interface.

Mirrors wegofwd_llm/contract.py. Imports nothing app-specific; holds no app's
StoryUnit governance, prompts, or data model (ADR-026 D3). The StoryUnit schema
and the Veo brief *template* live in project-critique/story-video-template/; this
module only models the brief a provider needs to make a call.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

# Version of the brief/request/result/Provider contract defined in THIS module.
# Bump on any change stored data or callers must notice. Stamped into provenance
# (registry.provenance). Distinct from a provider's integration_version and the
# vendor's model id. See ADR-026 D5.
VIDEO_CONTRACT_VERSION = 1


@dataclass(frozen=True)
class VideoCapabilities:
    """What a provider/model can do. Drives assert_brief_within_capabilities."""

    max_duration_s: int
    resolutions: tuple[str, ...] = ("720p",)
    aspect_ratios: tuple[str, ...] = ("16:9",)
    native_audio: bool = False
    reference_images: int = 0  # how many "ingredient" reference images it accepts
    upscaling: bool = False
    deterministic: bool = False  # True => same seed/brief reproduces frames exactly


@dataclass(frozen=True)
class Ingredient:
    """A reference image pinning identity/setting across shots (Veo 3.1
    Ingredients-to-Video). `role` ties to a story character id, 'setting', or 'style'."""

    role: str
    ref: str = ""  # asset pointer to the reference image (caller-resolved)
    description: str = ""


@dataclass(frozen=True)
class Shot:
    """One shot, derived from one story scene (scene_index links them)."""

    scene_index: int
    prompt: str  # subject + action + setting, assembled from the scene
    shot_type: str = ""  # 'medium' | 'wide establishing' | 'close-up'
    camera_move: str = ""  # 'static' | 'slow push-in' | 'gentle pan'
    lighting: str = ""
    dialogue: str = ""  # spoken line; drives native audio (usually = scene.narration)
    sfx: tuple[str, ...] = ()
    duration_s: float = 0.0
    negative: str = ""  # per-shot overrides on top of brief.global_negative


@dataclass(frozen=True)
class VideoBrief:
    """The structured prompt built FROM story.scenes. global_* fields hold
    continuity applied to every shot; ingredients lock identity across shots."""

    global_style: str
    shots: tuple[Shot, ...]
    global_negative: str = ""
    audio_direction: str = ""
    ingredients: tuple[Ingredient, ...] = ()


@dataclass(frozen=True)
class VideoRequest:
    """One generation request, provider-independent. The provider INSTANCE holds
    the model id + key; this holds the per-call parameters."""

    brief: VideoBrief
    resolution: str = "1080p"
    aspect_ratio: str = "16:9"
    fps: int = 24
    target_duration_s: float = 0.0
    seed: int | None = None  # pin for reproducibility (required once a unit is APPROVED)
    audio: bool = True


@dataclass(frozen=True)
class VideoResult:
    """A generation result. The package returns bytes OR a vendor URI + metadata;
    PERSISTING it (S3 key / filesystem path) is the caller's job (ADR-026 D2)."""

    provider_id: str
    model: str
    asset_bytes: bytes | None = field(default=None, repr=False)
    asset_uri: str | None = None  # vendor URI when the asset is fetched separately
    duration_s: float = 0.0
    resolution: str = ""
    has_audio: bool = False
    c2pa_signed: bool = False  # Veo emits Google C2PA + SynthID
    watermark: str = ""  # e.g. 'SynthID'
    seed: int | None = None
    raw: object | None = field(default=None, repr=False)  # provider payload, debug only


class VideoProvider(ABC):
    """A video provider under the contract. BYOK: the key is passed by the caller,
    never sourced here. generate() BLOCKS until the asset is ready (it may submit +
    poll a long-running vendor op internally); the CALLER runs it inside its own
    worker/queue — orchestration is not the package's job (ADR-026 D2). An
    implementation must NEVER let a key reach an exception, log line, or `raw`."""

    provider_id: str = ""
    capabilities: VideoCapabilities = VideoCapabilities(max_duration_s=0)

    @property
    @abstractmethod
    def model(self) -> str:
        """The concrete model id this instance produces with."""

    @abstractmethod
    def generate(self, req: VideoRequest) -> VideoResult:
        """Produce one video. Raises a wegofwd_video.errors.VideoError subclass on
        failure — never a raw SDK/HTTP exception that might stringify a key."""
