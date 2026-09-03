# Video engine survey — September 2026

**Compiled 2026-09-03** for the outstanding `veo` provider decision (Vertex AI vs
Gemini Developer API). Twelve engines: who serves them, what the output is worth,
what a second costs, and which of the seven config fields the provider sends are
actually honoured.

> **Prices move monthly.** Google's rates below were read from Google's own
> pricing page and carry the highest confidence. Everything else is vendor or
> aggregator reporting, marked per row. **Reconfirm before committing spend.**
>
> Web version of this survey (same content, charted):
> <https://claude.ai/code/artifact/fe545967-bc9b-4f50-afad-e321c464febd>

---

## 0. Time-critical: Sora 2 is being switched off

**The OpenAI Videos API and every Sora 2 model are removed on 2026-09-24** — 21
days from compilation.

| Event | Date |
|---|---|
| Deprecation notified to developers | 2026-03-24 |
| Sora web + app experiences discontinued | 2026-04-26 |
| **Videos API stops accepting requests** | **2026-09-24** |

Removed model ids: `sora-2`, `sora-2-pro`, `sora-2-2025-10-06`,
`sora-2-2025-12-08`, `sora-2-pro-2025-10-06`. OpenAI has announced **no successor
video product**. Anything built on it stops returning results with no fallback.

Sora is excluded from the shortlist below **on availability, not on quality.**

---

## 1. The read

Compliance training inverts the usual ranking. Reproducibility, provenance and
indemnity outrank cinematic quality, because the artefact has to survive an
auditor rather than an audience. Three engines survive that filter, and they are
not the three with the best pictures.

1. **Veo 3.1 on Vertex AI — for anything that must be reproducible.**
   The only shortlisted path accepting `seed`, and the only one carrying Google's
   generated-output IP indemnity plus SynthID provenance. Most expensive per
   second; needs service-account auth rather than an API key. This is the path
   the registry's `base_url` already points at.

2. **Kling 3.0 — for volume drafting.**
   #1 on the text-to-video arena at 1934 Elo, native audio, ~⅕ the price of
   standard Veo. No indemnity, no seed, PRC-hosted control plane. Fine for drafts
   an author reviews; wrong for a frozen approved artefact.

3. **An avatar engine — for most of the actual content.**
   A large share of compliance training is a presenter talking to camera over
   slides. Synthesia and HeyGen do that for **$3/min**, deterministically, from a
   script — and Synthesia carries ISO 42001, ISO 27001 and SOC 2 Type II with EU
   data residency, which no cinematic model offers.

---

## 2. Rates, by provider

Per second of output, USD. `8s shot` is one 720p shot with audio where supported
— the unit `scripts/first_veo_run.py` submits. `180s module` is three minutes of
**accepted** output and **excludes retakes** (see §3).

| Engine | Model id | 720p | 1080p | 4K | 8s shot | 180s module | Basis |
|---|---|---|---|---|---|---|---|
| **Veo 3.1** | `veo-3.1-generate-preview` | $0.40 | $0.40 | $0.60 | $3.20 | $72.00 | Google, official |
| **Veo 3.1 Fast** | `veo-3.1-fast-generate-preview` | $0.10 | $0.12 | $0.30 | $0.80 | $18.00 | Google, official |
| **Veo 3.1 Lite** | `veo-3.1-lite-generate-preview` | $0.05 | $0.08 | — | $0.40 | $9.00 | Google, official |
| Kling 3.0 | `kwaivgi/kling-v3.0-std` | $0.084<br>$0.126 w/ audio | — | — | $0.67 | $15.12 | Kuaishou / OpenRouter |
| Runway Gen-4.5 | `gen-4.5` | $0.12 | $0.12 | — | $0.96 | $21.60 | Runway API, 12 cr/s @ $0.01 |
| Seedance 2.0 | `bytedance/seedance-2.0` | $0.067 (480p floor) | ~$0.15 | $0.778 | $0.54 | $12.11 | BytePlus / resellers |
| HappyHorse 1.0 | Alibaba ATH | ¥0.9 ≈ $0.125 | ¥1.6 ≈ $0.22 | — | ≈$1.00 | ≈$22.50 | Vendor, CNY |
| Wan 2.5 | open weights, Apache 2.0 | $0.05 (480p) | $0.15 | preview | $1.20 | $27.00 | fal.ai hosted |
| Hailuo 2.3 | MiniMax | ~$0.04 | ~$0.08 | — | $0.36 | $8.10 | Aggregator |
| Luma Ray3 | — | ~$0.21 | ~$0.21 | — | $1.68 | $37.80 | Aggregator |
| Pika 2.2 | — | ~$0.05 | — | — | $0.40 | $9.00 | Aggregator |
| ~~Sora 2 / Pro~~ | removed 2026-09-24 | $0.10 | $0.30–0.50 | — | $0.80 | $18.00 | **API ends** |

### Notes on the rates

- **Google is authoritative here and contradicts the blogs.** Multiple
  third-party sources quote Veo 3.1 at `$0.75/s`; Google's own pricing page says
  `$0.40/s` for 720p/1080p and `$0.60/s` for 4K. Treat unsourced round numbers
  with suspicion.
- **No free tier for video** on any Veo variant. Confirmed on Google's page.
  This is why a valid key gets `429 RESOURCE_EXHAUSTED` — see §7.
- **Google charges only on successful generation** ("an audio processing issue
  may prevent a video from being generated. You will only be charged if your
  video is successfully generated"). This is the exception; most platforms bill
  rejected generations.
- Kling and Seedance publish **separately higher rates** for audio and for
  video-input / motion-control modes. Kling Pro with video input reaches
  $0.168/s.
- Sora 2 offered a **50% batch discount** at 24-hour latency ($0.05/s standard).
  Now moot.
- Reseller spreads are wide and inconsistent: EvoLink quotes $0.075/s for Kling
  3.0 against Kuaishou's own $0.084/s, Atlas Cloud claims "~30% reduced" while
  listing $0.126/s. Aggregator pricing is not reliably cheaper than official.

---

## 3. The retake multiplier is the real cost driver

Prompt-to-usable-shot ratios of **3–5×** are normal for cinematic generation, and
on most platforms rejected generations still bill.

| Tier | 3-min module, accepted | Realistic at 3–5× |
|---|---|---|
| Veo 3.1 Lite 720p | $9.00 | **$27–45** |
| Veo 3.1 Fast 720p | $18.00 | $54–90 |
| Veo 3.1 standard 720p | $72.00 | **$216–360** |
| Kling 3.0 std | $15.12 | $45–76 |

A 3-minute module is ~23 shots at the 8-second clip ceiling most of these models
impose. **That spread, not the picture quality, is what should pick the drafting
tier.** Generating drafts on the standard tier is the single most expensive
mistake available here.

---

## 4. Quality — blind pairwise voting

Artificial Analysis video arena, September 2026. Least gameable public signal.

| Rank | Model | Arena Elo | Native audio | Notes |
|---|---|---|---|---|
| 1 | **Kling v3** | 1934 | Yes | Text-to-video leader; four entries in the top 10 |
| 2 | HappyHorse 1.0 | 1816 | Yes | Alibaba; also #1 image-to-video at 1392 Elo |
| 3 | Seedance 2.0 Fast | 1747 | Yes | Strong reference-driven character consistency |
| ≈3 | **Veo 3.1** | — | Yes | Held #3 on the with-audio board mid-2026 |
| — | Runway Gen-4.5 | — | Yes | Below the leaders; best granular camera control |

**Caveat on the metric:** the arena measures whether a clip is *pleasing*, not
whether it is *correct*. Compliance video fails on facts, hands, and on-screen
text far more often than on aesthetics — and on-screen text is exactly what a
policy explainer wants. An arena score does not predict that.

**The structural finding:** the with-audio leaderboard is dominated by ByteDance,
Alibaba and Kuaishou. **Best-picture and best-governance now pull in opposite
directions** — the top three models are the three with no IP indemnity and a
PRC-hosted control plane. A compliance product has to decide that trade
explicitly rather than inherit it from a leaderboard.

---

## 5. Which config fields are actually honoured

The expensive lesson from the first live run: **capability is per-API-surface,
not per-model.** The same Veo 3.1 weights accept `seed` on Vertex and reject it
on the Gemini Developer API. Some rejections are **server-side only** — no
client-side validation catches them.

| Capability | Vertex AI | Gemini Dev API | Kling 3.0 | Runway | Self-hosted Wan / LTX |
|---|---|---|---|---|---|
| `seed` — reproducible output | ✅ | ❌ rejected | ❌ | ❌ | ✅ |
| `generate_audio` | ✅ | ❌ rejected | ✅ | ✅ | ⚠️ Wan 2.5 only |
| `reference_images` | ✅ up to 4 | ❌ rejected | ⚠️ 1 | ⚠️ 1 | ✅ |
| `negative_prompt` | ✅ | ⚠️ **not on Lite** | ✅ | ✅ | ✅ |
| `duration_seconds` / `fps` | ✅ | ❌ rejected | ✅ | ✅ | ✅ |
| 4K output | ✅ | ⚠️ not on Lite | ✅ | ❌ | ⚠️ LTX-2 only |
| Auth model | Service account | API key | API key | API key | None |

Also rejected on the Developer API path (read from the SDK source, free):
`compression_quality`, `labels`, `output_gcs_uri`, `person_generation`,
`webhook_config`.

**Of the seven config fields the provider currently sends, three survive the
Developer API path:** `aspect_ratio`, `resolution`, `number_of_videos`.

### Consequence for Pramana

`VideoRequest.seed` is documented as **required once a unit is `APPROVED`**.
Reproducible regeneration of an approved course version is therefore
**unachievable on the Gemini Developer API at all.** That is an API-surface
choice, not a flag — no amount of provider-side work recovers it.

---

## 6. The compliance filter

The axis general-purpose comparisons omit, and the only one that changes the
answer for a product whose output becomes audit evidence.

| Provider | Generated-output IP indemnity | Provenance | Certifications / residency | Control plane |
|---|---|---|---|---|
| **Google Vertex AI (Veo)** | Yes — indemnified-services terms cover eligible generated output at GA | SynthID pixel watermark | Full Google Cloud compliance surface; region selectable | US / EU / selectable |
| Adobe Firefly Video | Yes — contractual, reported up to **$3M per asset**; strongest posture in the field | C2PA Content Credentials | Enterprise agreements | US / EU |
| **Synthesia** | Enterprise terms; commercial rights bundled at higher tiers | Vendor-controlled avatars | **ISO 42001, ISO 27001, SOC 2 Type II**; EU residency (Frankfurt, Dublin) | EU / US |
| HeyGen | Commercial rights on paid tiers | Vendor-controlled avatars | No published HIPAA documentation as of 2026-03 | US |
| Runway | Commercial rights at higher tiers only | — | — | US |
| Kling / Seedance / HappyHorse | **None** | — | — | PRC-hosted (BytePlus offers an intl. endpoint) |
| Self-hosted Wan 2.2 / LTX-2 | N/A — Apache 2.0 (Wan); LTX free under $10M ARR | Yours to add | Whatever your own estate holds | Your infrastructure |

### Two caveats

**Provenance survives the generator, not the internet.** Platforms routinely
strip metadata on upload. And an absent credential proves nothing either way — a
missing watermark is not evidence that content is human-made. C2PA (cryptographic
signed manifest, v2.2) and SynthID (imperceptible pixel watermark) work by
different mechanisms and carry different amounts of information; the industry has
converged on wanting both layers.

**Indemnity is narrower than the marketing.** It is conditioned on using the
service as specified, typically excludes outputs you have steered toward a known
property, and **does not transfer through a reseller or aggregator path.** Buying
Veo through fal.ai or OpenRouter is not buying Google's indemnity.

---

## 7. Two engines the cinematic table hides

### Presenter engines — Synthesia, HeyGen

Script in, avatar delivering it out.

| | HeyGen | Synthesia |
|---|---|---|
| API rate | **$0.05/s = $3/min** (Avatar V) | per-minute self-serve tiers |
| Self-serve plans | $29/mo Creator, unlimited videos | ~$18–29/mo Starter, ~$64–89/mo Creator |
| Languages | 175+ | 130+ |
| Positioning | avatar realism category leader; video translation | **enterprise training, enablement, compliance at scale** |
| Certifications | none published (no HIPAA docs as of 2026-03) | ISO 42001, ISO 27001, SOC 2 Type II |

**Deterministic in the way that matters:** the same script and avatar give the
same video. That is exactly the property a frozen approved course version needs,
and no diffusion model offers it without a seed. Language coverage becomes
relevant the moment a SOX programme spans jurisdictions.

### Deterministic rendering — already in the registry

`deterministic-renderer` (`blender-grease-pencil-v2`) is the only path with **no
generative uncertainty at all**: byte-identical output from identical input, no
watermark question, no indemnity question, no data leaving the estate, zero
marginal cost per second.

Right answer for the majority of compliance visuals that are genuinely diagrams —
approval chains, segregation-of-duties matrices, control flows. Wrong answer for
anything wanting a human on screen.

### Why self-hosting a diffusion model is not the cheap option

| Model | Licence | Notes |
|---|---|---|
| Wan 2.2 / 2.5 | **Apache 2.0**, no limits | 2.5 does 1080p ≤10s with synced dialogue/ambient/music in one pass; 4K in preview |
| LTX-2 / 2.5 | free under **$10M ARR** | native 4K with synced audio in one pass (open, Jan 2026) |
| HunyuanVideo 1.5 | Tencent terms | 16GB+ VRAM minimum |
| CogVideoX-1.5 | commercial use permitted | — |

GPU rental runs **$0.40–2.00/hr** for capable hardware. Break-even against API
pricing sits **north of ~5,000 clips/month** once the engineering to keep it
alive is counted. Self-hosting buys data residency, not savings.

---

## 8. What this implies for the registry

Mapping the survey onto `wegofwd_video/registry.py` as it stands today.

| Registry entry | State today | What the survey says |
|---|---|---|
| `veo` | **Split brain** — `base_url` is Vertex, client is Dev API | Declared capabilities (`native_audio=True`, `reference_images=4`, `4k`) are **true on Vertex and false on the path the provider actually calls**. Either target Vertex and keep them, or target the Developer API and narrow all three. |
| `kling` | **UNVERIFIED** — blank `base_url`, guessed model id | Now the **arena leader** with native audio at ~$0.084/s. Worth verifying and promoting to the drafting role — but not the approved-artefact role, on governance grounds rather than quality. Real id is `kwaivgi/kling-v3.0-std` on OpenRouter. |
| `runway` | **UNVERIFIED** | Rate confirmed at $0.12/s and `gen-4.5` is a **real** id. No seed, no indemnity, mid-table quality. Keep for camera-control work; low priority. |
| `deterministic-renderer` | verified | **Undervalued.** Should be the default for diagram-shaped content rather than a fallback. |
| *— absent —* | **gap** | **No presenter/avatar provider**, which is the shape most compliance training actually takes. A `presenter-video` role pointing at Synthesia or HeyGen is the highest-value addition on this page. |

### The decision this survey does not make

It does not break the Vertex-vs-Developer-API tie — it **raises the price of the
Developer API.** `seed` is Vertex-only, so "reproducible once `APPROVED`" is
unreachable there regardless of quota. The cost of going to Vertex is that
`build_provider(api_key=…)` no longer fits: Vertex wants a service account.

Options, unchanged from [ADR-026 open questions]:

- **Vertex** — matches `base_url`, restores seed/audio/reference-images, is what
  production wants. Different auth, so the BYOK signature needs revisiting.
- **Developer API** — keep the client, stop sending the four unsupported fields,
  narrow the registry capabilities that are wrong for this path.
- **Both, selected by config** — honest, largest change.

`wegofwd-video` is v1.0.0 with a frozen interface, so this is not a unilateral
call.

### Unchanged blocker

**Veo quota is not enabled** on the Google Cloud project. The key is valid and
sees all 54 models including all three Veo variants, but generation returns
`429 RESOURCE_EXHAUSTED`. Veo is paid; a free-tier key can see the models without
being able to invoke them, and Google's pricing page confirms **no free tier for
video** on any variant. Nothing further can be proven without this.

---

## 9. Sources

Google's rates were read from its own pricing documentation. Arena scores and
non-Google rates come from vendor pages and aggregator reporting.

- [Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing) — authoritative Veo 3.1 / Fast / Lite per-second rates; "no free tier for video"
- [OpenAI API deprecations](https://developers.openai.com/api/docs/deprecations) and [What to know about the Sora discontinuation](https://help.openai.com/en/articles/20001152-what-to-know-about-the-sora-discontinuation) — the 2026-09-24 removal
- [Google Cloud generative AI indemnified services](https://cloud.google.com/terms/generative-ai-indemnified-services) — scope and conditions of the output indemnity
- [Kling v3.0 Standard on OpenRouter](https://openrouter.ai/kwaivgi/kling-v3.0-std) and [Seedance 2.0 on OpenRouter](https://openrouter.ai/bytedance/seedance-2.0) — per-second rates with and without audio
- [Blind-vote video generation leaderboard](https://llm-stats.com/leaderboards/best-ai-for-video-generation) and [arena-score roundup](https://techsy.io/en/blog/best-ai-video-models) — Elo figures
- [Runway Gen-4 API specs and pricing](https://unifically.com/blogs/runway-gen-4) — the $0.01/credit basis behind $0.12/s
- [HeyGen pricing](https://www.eesel.ai/blog/heygen-pricing) and [HeyGen vs Synthesia](https://www.colossyan.com/posts/heygen-vs-synthesia/) — avatar API rates, certifications, language coverage
- [Self-hosting LTX / Wan / Hunyuan on GPU cloud](https://www.spheron.network/blog/image-to-video-gpu-cloud-ltx-wan-hunyuan/) and [open-source video model landscape](https://ltx.io/blog/open-source-video-generation-models-guide) — licences and break-even
- [C2PA vs SynthID](https://c2paviewer.com/articles/verify-ai-generated-image-c2pa-synthid) — how the two provenance mechanisms differ and where both fail
- [AI vendor indemnification, in practice](https://www.runtime.news/ai-vendors-promised-indemnification-against-copyright-lawsuits-the-details-are-messy/) — why the headline promise narrows on reading
- [Seedance 2.0 pricing breakdown](https://www.atlascloud.ai/blog/case-studies/seedance-2.0-pricing-full-cost-breakdown-2026) and [HappyHorse vs Seedance cost analysis](https://help.apiyi.com/en/happyhorse-pricing-vs-seedance-2-comparison-en.html) — resolution-tier rates
- [fal.ai vs Replicate](https://www.teamday.ai/blog/fal-ai-vs-replicate-comparison) and [AI video API pricing 2026](https://apiframe.ai/blog/ai-video-api-pricing-2026) — aggregator rates for Wan, Hailuo, Luma, Pika
