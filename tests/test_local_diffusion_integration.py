"""Real diffusers path, tiny random weights — the integration gate for [local].

Skipped unless torch + diffusers + transformers are importable (CI has none of
them), so the unit suite stays weight-free while a host with the `local` extra
proves the actual pipeline mechanics: pipeline loaded with `text_encoder=None`,
pre-computed prompt embeds accepted, the per-step callback firing, frames
exported to H.264, seed determinism, and the transformer-swap fallback.

Builds a diffusers-layout LTX pipeline with a 1-layer transformer, a tiny VAE
and a 32-dim T5 in tmp_path (~2 MB, ~20 s on a laptop). Random weights produce
noise, which is fine: this checks plumbing, not pictures.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import wegofwd_video as wv

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")
transformers = pytest.importorskip("transformers")
pytest.importorskip("imageio_ffmpeg")

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory) -> Path:
    from diffusers import (
        AutoencoderKLLTXVideo,
        FlowMatchEulerDiscreteScheduler,
        LTXPipeline,
        LTXVideoTransformer3DModel,
    )
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, T5Config, T5EncoderModel

    root = tmp_path_factory.mktemp("tiny-ltx")
    torch.manual_seed(0)

    vocab = {"<pad>": 0, "</s>": 1, "<unk>": 2}
    for i, word in enumerate("a the person desk office reviews document clean corporate".split()):
        vocab[word] = 3 + i
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tk, pad_token="<pad>", eos_token="</s>", unk_token="<unk>"
    )
    text_encoder = T5EncoderModel(
        T5Config(
            vocab_size=len(vocab),
            d_model=32,
            d_ff=64,
            d_kv=8,
            num_heads=4,
            num_layers=1,
            num_decoder_layers=1,
            decoder_start_token_id=0,
        )
    )
    transformer = LTXVideoTransformer3DModel(
        in_channels=8,
        out_channels=8,
        patch_size=1,
        patch_size_t=1,
        num_attention_heads=4,
        attention_head_dim=8,
        cross_attention_dim=32,
        num_layers=1,
        caption_channels=32,
    )
    vae = AutoencoderKLLTXVideo(
        in_channels=3,
        out_channels=3,
        latent_channels=8,
        block_out_channels=(8, 8, 8, 8),
        decoder_block_out_channels=(8, 8, 8, 8),
        layers_per_block=(1, 1, 1, 1, 1),
        decoder_layers_per_block=(1, 1, 1, 1, 1),
        spatio_temporal_scaling=(True, True, False, False),
        decoder_spatio_temporal_scaling=(True, True, False, False),
        decoder_inject_noise=(False,) * 5,
        upsample_residual=(False,) * 4,
        upsample_factor=(1, 1, 1, 1),
        timestep_conditioning=False,
        patch_size=1,
        patch_size_t=1,
        encoder_causal=True,
        decoder_causal=False,
    )
    LTXPipeline(
        transformer=transformer,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=FlowMatchEulerDiscreteScheduler(),
    ).save_pretrained(str(root))
    return root


def _brief() -> wv.VideoBrief:
    return wv.VideoBrief(
        global_style="clean corporate",
        shots=(wv.Shot(scene_index=0, prompt="a person at the desk reviews a document"),),
    )


def _provider(tiny_model: Path, **opts):
    return wv.build_provider(
        "local-diffusion",
        model=str(tiny_model),
        transformer_file=None,  # the tiny repo's own transformer
        preflight=False,  # the RAM check is sized for the real T5-XXL
        threads=2,
        **opts,
    )


def test_real_engine_renders_an_h264_clip_and_reports_timing(tiny_model: Path) -> None:
    progress: list[tuple[int, int, float]] = []
    p = _provider(tiny_model, steps=2, on_progress=lambda *a: progress.append(a))
    req = wv.VideoRequest(brief=_brief(), resolution="256p", seed=1, target_duration_s=0.2)

    result = p.generate(req)

    assert result.asset_bytes and result.asset_bytes[4:12] == b"ftypisom"  # MP4 container
    assert result.duration_s == pytest.approx(9 / 24, abs=1e-3)  # 0.2 s -> 9 frames
    assert result.seed == 1 and result.has_audio is False
    assert [s for s, _, _ in progress] == [1, 2]
    assert result.raw["transformer_used"] == "repo:transformer"
    assert result.raw["seconds_per_step"] is not None and result.raw["render_seconds"] > 0
    assert (result.raw["width"], result.raw["height"], result.raw["num_frames"]) == (448, 256, 9)


def test_same_seed_reproduces_the_same_bytes(tiny_model: Path) -> None:
    req = wv.VideoRequest(brief=_brief(), resolution="256p", seed=7, target_duration_s=0.2)
    a = _provider(tiny_model, steps=1).generate(req).asset_bytes
    b = _provider(tiny_model, steps=1).generate(req).asset_bytes
    assert a == b


def test_transformer_swap_failure_falls_back_and_says_so(tiny_model: Path) -> None:
    p = wv.build_provider(
        "local-diffusion",
        model=str(tiny_model),
        transformer_file="does-not-exist.safetensors",  # cannot be fetched -> fallback
        preflight=False,
        threads=2,
        steps=1,
    )
    result = p.generate(
        wv.VideoRequest(brief=_brief(), resolution="256p", seed=1, target_duration_s=0.2)
    )
    assert result.raw["transformer_used"] == "repo:transformer"
    assert result.raw["transformer_fallback_reason"]


def test_swap_failure_is_an_error_when_fallback_is_disabled(tiny_model: Path) -> None:
    p = wv.build_provider(
        "local-diffusion",
        model=str(tiny_model),
        transformer_file="does-not-exist.safetensors",
        fallback_to_repo_transformer=False,
        preflight=False,
        steps=1,
    )
    with pytest.raises(wv.VideoConfigurationError):
        p.generate(wv.VideoRequest(brief=_brief(), resolution="256p", target_duration_s=0.2))
