# Local diffusion on a CPU — proof-of-concept runbook

**Provider:** `local-diffusion` · **Role:** `local-preview` · **Model:**
pipeline from `Lightricks/LTX-Video-0.9.5` (diffusers layout: VAE, scheduler,
T5-XXL, transformer config) with the transformer swapped for
`ltxv-2b-0.9.8-distilled.safetensors` from `Lightricks/LTX-Video` (2B params,
distilled: 4–10 steps, guidance 1.0). If that swap fails on the host, the
provider falls back to 0.9.5's own (non-distilled) transformer and says so in
`result.raw["transformer_fallback_reason"]` — then use `--steps 30`. License: LTX-Video license
(free for commercial use under $10M ARR — check before shipping).

**Host this was written for:** `mambakkam` — Intel i5-3570K (4 cores, Ivy Bridge,
AVX only), 32 GB RAM, 3.6 TB disk, GeForce GT 640 (Kepler, not CUDA-12 capable →
treated as no GPU). Ubuntu 24.04, Python 3.12.

**What this proves:** the same `VideoBrief` that goes to Veo can be rendered on
a machine with no GPU and no vendor account, and *how long that takes*. It is
not a quality path; that is the measurement's job to show.

## 1. One-time setup on the host

```bash
# system tools
sudo apt install -y ffmpeg

# venv OUTSIDE the repo (a venv inside it gets staged into git)
python3 -m venv ~/venvs/wegofwd-video
source ~/venvs/wegofwd-video/bin/activate

# CPU torch FIRST, from the CPU index (≈200 MB, not the 3 GB CUDA wheel)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -c "import torch; print(torch.__version__, torch.get_num_threads())"

# the package with the local extra (diffusers, transformers, accelerate, …)
cd ~/Documents/code/projects/AIStuff/STEM_studybuddy/wegofwd-video
pip install -e ".[dev,local]"
pytest            # unit suite needs no weights; the [local] integration test builds a tiny model
```

If `import torch` dies with *Illegal instruction* on this CPU, the wheel was
built assuming AVX2; pin an older CPU wheel (`torch==2.3.1`) and note it below.
**It does not.** `torch 2.14.0+cpu` imports and runs fine on this Ivy Bridge
(AVX-only) host — 4 threads, `cuda.is_available() False`. No pin needed.

**`protobuf` is required and the extra used to omit it.** T5's tokenizer ships
only `spiece.model`; `transformers` converts it to the fast format with
`SentencePieceExtractor`, which imports protobuf. Without it the conversion does
not fail cleanly — it falls back to a TikToken reader and dies one layer down
with *"`tiktoken` is required to read a `tiktoken` file"*, which points at the
wrong package. `protobuf>=4` is now in the `local` extra. The tiny-model
integration test cannot catch this: it builds a `PreTrainedTokenizerFast`
directly, so it never exercises the slow→fast conversion.

`pytest tests/test_local_diffusion_integration.py -v` runs the real diffusers
path against a tiny random-weight LTX pipeline built in a temp dir (~30 s, no
download). If that passes on the host, everything except the real weights is
proven before the 10 GB download starts.

Weights download on the first real run into `~/.cache/huggingface`. Set `HF_HOME`
to put them on the big disk if home is small.

**Budget 26 GB, not the ~10 GB this section used to claim** — measured
2026-09-09: 21 GB for `Lightricks/LTX-Video-0.9.5` plus 6 GB for the distilled
transformer pulled from `Lightricks/LTX-Video`. The 0.9.5 fetch takes the whole
repo, including a T5 variant and a non-distilled transformer that the swap then
makes unused, so roughly 11 GB of it is dead weight on this path. Narrowing that
fetch with `allow_patterns` is the obvious saving if disk ever matters; it did
not here (2.5 TB free).

## 2. Run matrix

Run these in order; each one is the go/no-go for the next. Always pass `--json`
so the numbers survive.

| # | Command | Geometry | Purpose | Expect on this CPU |
|---|---------|----------|---------|--------------------|
| 0 | `python scripts/first_local_run.py --dry-run --smoke` | 448×256, 25 f, 4 steps | plan only, no torch | seconds |
| 1 | `python scripts/first_local_run.py --smoke --out /tmp/smoke.mp4 --json /tmp/smoke.json` | 448×256, 25 f, 4 steps | end-to-end: weights load, T5 encode, render, mp4 | **measured 15.1 min** incl. 4.6 min download; 19.7 s/step |
| 2 | `… --resolution 320p --duration 2 --steps 8 --out /tmp/320p.mp4 --json /tmp/320p.json` | 576×320, 49 f | first "watchable" clip | ~20 min projected (≈63 s/step × 8 + fixed cost, weights now cached) |
| 3 | `… --resolution 480p --duration 4 --steps 8 --brief tests/data/sox_lesson_brief.json --out /tmp/sox.mp4 --json /tmp/sox.json` | 864×480, 97 f | a real Pramana scene | ~50–70 min projected — **memory, not time, is the risk here** (see §3) |

The "expect" column started as a guess; row 1 is now measured and rows 2–3 are
extrapolated from it rather than invented. The transformer
cost is roughly linear in latent tokens (`width/32 × height/32 × (frames-1)/8`),
so one measured seconds-per-step at row 1 predicts rows 2–3 reasonably:

| row | latent tokens | × row 1 |
|-----|---------------|---------|
| 1 | 14 × 8 × 3 = 336 | 1 |
| 2 | 18 × 10 × 6 = 1 080 | ≈ 3.2 |
| 3 | 27 × 15 × 12 = 4 860 | ≈ 14.5 |

Useful flags: `--model <hf-repo-or-folder>` (a different diffusers-layout
repo), `--transformer-file none` (skip the distilled swap and use the repo's own
transformer — then raise `--steps` to ~30), `--threads 4` (pin torch to the physical cores; hyper-threads can
slow fp32 matmuls), `--timeout 7200` (abort rather than run all night),
`--text-encoder-dtype float32` only if bf16 emulation misbehaves (needs ~22 GB
free), `--seed` to reproduce a clip exactly.

## 3. Reading the output

```
[ step       ] 3/8    41.2 s/step  ~  3.4 min left
[ generated  ] 6.1 min wall clock, peak RSS 13.8 GB
[ stored     ] 412337 bytes -> /tmp/320p.mp4
```

`seconds_per_step` is the number that matters; `peak RSS` tells you whether the
sequential-loading assumption held. **It held, but with far less headroom than
this document originally predicted:** row 1 peaked at **23.9 GB of 32 GB**, not
the 10–12 GB estimated for the bf16 T5 phase. Sequential loading is doing real
work — without it this would not fit at all — but only ~8 GB is left over, and
row 3 asks for 14.5× the latent tokens of row 1. Treat OOM as the likely failure
mode of the 480p run, and drop to `--resolution 320p` or fewer frames before
assuming the machine is merely slow.

Failure stages: `CONFIG` (missing extra, weights not downloadable, geometry the
model rejects — the chained cause is printed), `TIMEOUT` (budget exceeded),
`FAILED` (backend error; cause printed). Memory exhaustion says so and suggests a
smaller geometry.

## 4. Measurement log

Fill in as runs complete. One row per run; keep the JSON files.

| date | row | torch | wall clock | s/step | peak RSS | looks like | notes |
|------|-----|-------|-----------:|-------:|---------:|------------|-------|
| 2026-09-09 | 1 | 2.14.0+cpu | 15.1 min | 19.7 s | 23.9 GB | a woman at a desk in an office holding a sheet of paper, coherent across all 25 frames — not noise | `from_single_file` idiom verified: `transformer_used` = `Lightricks/LTX-Video:ltxv-2b-0.9.8-distilled.safetensors`, **no fallback**. 24 174 bytes, h264, 448×256, 25 f @ 24 fps. Needed `protobuf` (§1). ≈4.6 min of the wall clock was the one-time 10 GB download |
| | 2 | | | | | | |
| | 3 | | | | | | |

When row 1 has produced a clip that is not noise:
1. set `model_verified=True` on the `local-diffusion` spec in `registry.py`;
2. if the pipeline call had to change to make it work, bump that spec's
   `integration_version`;
3. tick tasks 3.5/3.6 in `openspec/changes/add-local-diffusion-provider/tasks.md`.

## 5. Where this sits in the bigger picture

The provider abstraction is the point: the Pramana pipeline (script → scenes →
brief) does not change between `local-preview` and `narrative-video`. Local runs
prove briefs and geometry for free; a rented GPU (RTX 4090-class, ~$0.40–0.80/h
spot) renders the same brief in about a minute; Veo/Kling render it with audio.
The survey's conclusion stands — self-hosting on a CPU buys independence, not
speed — and after §4 it will stand on numbers.

Not covered here: audio (pair with a TTS step in the consumer), reference images,
multi-shot continuity (each brief renders as one clip; cut longer lessons into
one brief per scene and concatenate with ffmpeg).
