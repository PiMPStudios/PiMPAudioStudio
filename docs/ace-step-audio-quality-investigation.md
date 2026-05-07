# ACE-Step Audio Quality Investigation — Resumption Notes

**Last updated:** 2026-05-07
**Status:** Pipeline runs end-to-end without crashing. Output WAV is structurally valid (30s, 48 kHz, stereo, full amplitude). **Audio does not follow text prompts.** Sounds like structured stereo static / noise across all DiT variants and configurations.

> Use this doc to resume troubleshooting after a context reset. Skim "What's Verified Working" + "Hypotheses Still Open" first.

---

## Project context

Porting **ACE-Step v1.5** (audio diffusion model — text→music) from PyTorch+upstream-MLX into a native Swift + MLX-Swift macOS app (`AnvilApp`). Pipeline:

```
text prompt
  → Qwen3 text encoder + LyricEncoder + TimbreEncoder
  → ConditionEncoder fuses → encoder_states
  → cross-attention into DiT (rectified-flow diffusion, 8 or 50 steps)
  → AutoEncoderOobleck VAE decodes latents
  → 48 kHz stereo WAV
```

Upstream reference repos:
- Local Python source: `~/Developer/GitRepos/ACE-Step-1.5`
- Most relevant upstream MLX code: `acestep/models/mlx/dit_generate.py`, `acestep/models/mlx/dit_model.py`, `acestep/models/mlx/vae_model.py`
- Upstream PyTorch DiT: `acestep/models/turbo/modeling_acestep_v15_turbo.py`, `acestep/models/base/modeling_acestep_v15_base.py`, `acestep/models/sft/modeling_acestep_v15_base.py`

## Current symptom

Generation completes in ~5–60 s (depending on Turbo 8 steps vs Full/SFT 50 steps). Output WAV file:

- **30.0 s, 48000 Hz, stereo Int16, ~5.76 MB** ✓
- **Peak hits ±32767 (0 dBFS)**, RMS around -14 to -16 dBFS — full musical loudness
- **L/R correlation 0.7–0.92** (typical music range)
- **ZCR varies by run**: 1000-4800/sec (instrumental sustained → noisy edge of music)
- **Different prompts produce different audio character** — verified via amplitude/ZCR/dynamics analysis. So conditioning *is* reaching the model, but it's not strong enough to follow specific prompt content.

User listening: "static / noise, vaguely music-shaped, doesn't reflect the prompt." One lucky Turbo seed once produced a 270× per-second RMS dynamic range with strong musical structure (commit `c6ab178` era), but most runs are uniform-energy structured noise (ratio ~1.4×).

## What's verified working (don't re-investigate these)

### Weight loading (100% match for everything that has weights in v1.5 checkpoint)

Diagnostic logs in `DiTActor.loadWeights`, `VAEActor.loadWeights`, `ConditioningActor.loadWeights` print `supplied=N model=M matched=K` plus first 8 mismatches each side. Recent run:

| Module | Supplied | Model | Matched | Notes |
|---|---|---|---|---|
| DiT | 476 | 477 | 476 | Only `rotaryEmb.invFreq` unsupplied (runtime-computed) |
| Qwen3 | 310 | 311 | 310 | Same — `rotaryEmbedding.invFreq` |
| LyricEncoder | 91 | 92 | 91 | Same |
| TimbreEncoder | 47 | 73 | 47 | 26 unmatched all under `attentionPooler.layers.0..1.*` and `.norm` — those have NO weights in v1.5 checkpoint at all (only `special_token` exists at top level, shape mismatched [1,1,2048] vs our [1,5,2048] — bypass below) |
| ConditionEncoder | 1 | 1 | 1 | `text_projector.weight` → `textProjection.weight` |
| VAE | 291 | 291 | 291 | All match |

### Healthy intermediate values

`DiffusionActor` final-latent diagnostic shows abs_mean ≈ 0.85, abs_max ≈ 4.5 (Turbo) or 1.0 / 5.8 (Full). Both well within typical VAE-latent magnitude range.

`VAEActor.decode` diagnostic shows audio output abs_max ≈ 1.15 (Turbo) / 1.78 (Full). Slightly above [-1, 1] but normalized down by the WAVWriter peak-normalize fix.

### Architecture verified against upstream

- **DiT config** matches `config.json`: 24 layers, hidden_size=2048, intermediate=6144, 16 attention heads, 8 KV heads, head_dim=128, rms_norm_eps=1e-6, attention_bias=false, patch_size=2, sliding_window=128, rope_theta=1e6, audio_acoustic_hidden_dim=64, in_channels=192, layer_types alternating sliding/full
- **VAE config** matches `vae/config.json`: 48000 Hz sampling, downsampling_ratios `[2,4,4,6,10]` (total stride 1920), audio_channels=2, decoder_input_channels=64, encoder_hidden_size=128
- **Latent frame rate** matches: 48000 / 1920 = 25 fps. Both `contextLatents` and `noisyLatents` shaped `[1, 750, ...]` for 30s
- **APG implementation** byte-for-byte identical to upstream's `_mlx_apg_forward` in `acestep/models/mlx/dit_generate.py:86`: same momentum (no decay, additive accumulation), `axis=1` projection, same orthogonal-component-only formula
- **TimestepEmbedding**: scale=1000, max_period=10000, `[cos, sin]` order, silu→linear→silu→linear→silu→time_proj→reshape — matches upstream
- **scale_shift_table modulation**: `norm(x) * (1 + scale) + shift` before self-attn → `+attn*gate` residual → plain norm before cross-attn → `+cross-attn` residual (no gate or modulation on cross-attn) → `norm(x) * (1 + c_scale) + c_shift` before MLP → `+mlp*c_gate` residual. Matches upstream exactly.
- **DiT key mapping** (`mapDiTKey`): snake_case→camelCase, `decoder.` prefix strip, `proj_in.1.→proj_in.`, skip rotary_emb buffers, special-case top-level `null_condition_emb`. Verified all 476 weights resolve to real model parameter paths.
- **VAE key mapping** (`VAENameMap.mapKey`): block→blocks, snake→camel, fuse weight_g/weight_v. All 291 resolve.

## Fixes shipped (in roughly chronological order, all on `origin/main`)

Each commit message has full reasoning. Key ones for the audio path:

| Commit | Title (paraphrased) | Why |
|---|---|---|
| `f736bc2` | WAVWriter NLC stereo handling | VAE returns `[B, length, 2]`; we treated it as 1-D mono → 2× duration, alternating L/R noise |
| `07993fa` | VAE 48 kHz / `[2,4,4,6,10]` strides | Defaults were `[2,4,4,8,8]` at 44100 — wrong VAE architecture for v1.5 |
| `c6ab178` | WAVWriter peak-normalize before clip | VAE output abs_max > 1.0; hard-clip distorted audibly. Now scales by `1/abs_max` only when peak > 1.0 |
| `992be2a` | Auto step count by DiT variant | `DiffusionConfig.steps=nil` → fallback to 8-step schedule for ALL variants. Turbo→8, Full/SFT→50 |
| `d88a625` | DiT snake_case→camelCase weight key map | `mapDiTKey` only stripped `proj_in.1.`. 471 of 476 weights weren't loading |
| `6c99706` | VAE block→blocks + snake→camel weight key map | Only 11/291 VAE weights loaded before this |
| `0421733` | Wire `ConditioningActor.loadWeights` | Defined but never called — Qwen3 + lyric + timbre + condition encoders ran on default-init random weights |
| `ced3c55` | bias=true on lyric/timbre input projections, skip mismatched timbre special_token | v1.5 checkpoint provides bias for these but Linear was bias=false → load crashed |
| `fc483c9` | TimbreEncoder bypass when no reference audio | TimbreEncoder.AttentionPooler stays default-init random; running zeros through it produces 5 random tokens that polluted cross-attn. Now passes `[1, 5, 2048]` zeros directly |
| `1bc9a1c` | Drop cross-attn encoder mask | Upstream MLX hardcodes `encoder_attention_mask=None`; upstream Python shadows the param to None and rebuilds shape-only mask. Model trained without padding mask in cross-attn — applying one shifted distribution away from training |
| `9bb523b` | Bypass LyricEncoder when lyrics empty | Without cross-attn mask, LyricEncoder's 2048 PAD-input tokens get attended at full weight, drowning ~5–10 real text tokens 8:1 |
| `e8f7fe0` | shift = 3.0 | The MLX-specific upstream handler defaults shift=3.0; the 1.0 in `inference.py` is for the PyTorch backend. Tested both — same result either way |
| `34591d5` | **Use learned `null_condition_emb` for CFG unconditional branch** | We used `MLXArray.zeros(like: encHS)`. Upstream broadcasts a LEARNED `[1, 1, 2048]` token (top-level safetensors key) across encoder length. All-zeros is OOD for trained DiT; CFG `(cond - uncond)` was just adding noise on top of cond instead of isolating prompt direction. **User reported "a little bit better" after this fix** but still not matching prompt |

## Hypotheses still open

Ranked by my current likelihood estimate:

### 1. Float16 numerical drift (untested)

Generate tab UI defaults `precision: .float16`. Cross-attention softmax with very small denominators can lose precision in fp16. **Easy test the user can do:** switch the Precision setting in the UI to Float32 and re-run. Slower but rules out numerical issues. *Top priority for next session.*

### 2. Subtle math bug in DiTLayer or attention path

I've verified scale_shift_table application, residual placement, RMSNorm pre-norm position, MLP structure, and that q_proj/k_proj/v_proj have the right kNorm/qNorm pre-scaled-dot. But there could still be:

- An off-by-one in the rotary embedding application (only Q/K should get rotary in self-attn; cross-attn might or might not — need to verify)
- Wrong ordering of `transposed(0, 2, 1, 3)` in attention reshape
- `repeat_kv` for grouped-query attention — we have `nRep = numAttentionHeads / numKeyValueHeads = 16/8 = 2`; verify the broadcast direction matches upstream

**Diff approach:** dump intermediate tensor values (after each layer, after self_attn, after cross_attn, after mlp) for a fixed seed and same prompt, in both upstream MLX and ours, then compare numerically. First layer where they diverge is the bug. Multi-hour task but bulletproof.

### 3. Schedule curvature wrong

We use `t * shift / (1 + (shift - 1) * t)` — standard rectified-flow shifted schedule. Tested both shift=1.0 and shift=3.0; same result. Maybe the model expects a different schedule formula entirely? Config has `timestep_mu=-0.4, timestep_sigma=1.0` (training distribution). Inference might need a LogNormal-derived schedule, not the rectified-flow shifted one. Untested.

### 4. Cross-attention K/V uses different normalization than upstream

Worth comparing our `DiTAttention.callAsFunction` for the cross-attention path step-by-step against upstream's `MLXAttention` in `acestep/models/mlx/dit_model.py:155`. Specifically the kNorm/qNorm placement and whether the encoder_hidden_states get any treatment beyond the top-level `condition_embedder` (Linear) before being projected into K and V.

### 5. Float values of `condition_embedder.weight` post-load

Could verify the loaded weight matches the safetensors source. Add a debug log that prints `decoder.condition_embedder.weight.sum()` on both sides. If they don't match, weight conversion has a bug.

### 6. Silence-prompt baseline test

Generate with **empty text prompt + empty lyrics**. If output sounds the same as a real prompt, the model is being driven entirely by noise (conditioning never reaches it meaningfully). If output is dramatically different (e.g., near-silence or pure noise), the conditioning IS having effect, just not the right kind.

## Things explicitly ruled out

- WAV writer container correctness — fully verified
- VAE architecture — fully verified
- DiT architecture defaults — fully verified
- DiT weight loading completeness — verified 100% for everything with checkpoint weights
- Conditioning weight loading — verified 100%
- APG (CFG amplification) math — byte-identical to upstream
- TimestepEmbedding math — verified identical
- scale_shift_table modulation — verified identical
- Cross-attention mask handling — now matches upstream (no mask)
- TimbreEncoder padding contamination — bypassed
- LyricEncoder PAD-only contamination — bypassed
- CFG unconditional branch — now uses learned `null_condition_emb` (was zeros)
- Step count — auto-picks correctly per variant
- Guidance scale — defaults to 7.5, matches upstream defaults

## How to resume

1. **Open this doc first.** Read "Current symptom" and "Hypotheses still open."
2. **Run the app once** with prompt "slow dark guitar riff" on Full variant at 50 steps. Capture the diagnostic logs (DiTActor / VAEActor / ConditioningActor / DiffusionActor) and the latest WAV file.
3. **Run the amplitude analysis script** (in any prior conversation, search for "abs_max abs_mean ZCR" — there's a Python snippet that produces all the metrics).
4. **Try Float32 precision in the UI first.** That's the cheapest experiment.
5. **If Float32 doesn't change output character, dump intermediate tensors** for layer-by-layer numerical diff against upstream MLX.

## Sandbox cleanup helpers

To reset to fresh-user state for testing:

```bash
ANVIL_BASE="$HOME/Library/Containers/net.pimp.anvil.AnvilApp/Data/Library/Application Support/Anvil"
rm -rf "$ANVIL_BASE/models"
defaults delete net.pimp.anvil.AnvilApp com.pimp.AnvilApp.modelMigration.v1 2>/dev/null
```

The user's local checkpoint mirrors at `~/Developer/GitRepos/ACE-Step-1.5/checkpoints/` which can be copied into the sandbox to skip downloads:

```bash
SRC=~/Developer/GitRepos/ACE-Step-1.5/checkpoints
DST="$HOME/Library/Containers/net.pimp.anvil.AnvilApp/Data/Library/Application Support/Anvil/models"

# Qwen3 text encoder
mkdir -p "$DST/ace-step-vae"
cp "$SRC/Qwen3-Embedding-0.6B/model.safetensors" "$DST/ace-step-vae/qwen3_model.safetensors"
cp "$SRC/Qwen3-Embedding-0.6B/tokenizer.json" "$DST/ace-step-vae/tokenizer.json"
cp "$SRC/vae/diffusion_pytorch_model.safetensors" "$DST/ace-step-vae/vae.safetensors"
cp "$SRC/vae/config.json" "$DST/ace-step-vae/config.json"

# DiT (turbo)
mkdir -p "$DST/ace-step-dit-turbo"
# Note: the local turbo bundle is acestep-v15-turbo/model.safetensors which is the
# WHOLE model bundle (DiT + encoders + tokenizer + null_condition_emb).
# The HF-downloaded version is also model.safetensors at the same path.
cp "$SRC/acestep-v15-turbo/model.safetensors" "$DST/ace-step-dit-turbo/model.safetensors"
cp "$SRC/acestep-v15-turbo/config.json" "$DST/ace-step-dit-turbo/config.json"
```

Or simply let the app download from HuggingFace — Phase 2/3 download infrastructure is fully working.

## Useful diagnostic snippets

### WAV amplitude + structure analysis (Python, no numpy required)

```python
import wave, struct, math
fn = "anvil_TIMESTAMP_PROMPT.wav"
with wave.open(fn, 'rb') as w:
    n, ch, sw, sr = w.getnframes(), w.getnchannels(), w.getsampwidth(), w.getframerate()
    raw = w.readframes(n)
samples = struct.unpack(f"<{len(raw)//sw}h", raw)
left = samples[0::2]; right = samples[1::2]
mono = [(l+r)//2 for l,r in zip(left,right)]
abs_s = [abs(s) for s in samples]
print(f"peak={max(abs_s)} ({20*math.log10(max(max(abs_s),1)/32767):.1f} dBFS)")
print(f"RMS={math.sqrt(sum(s*s for s in samples)/len(samples)):.0f}")
# Per-second energy
es = [math.sqrt(sum(x*x for x in mono[i:i+sr])/sr) for i in range(0, len(mono)-sr, sr)]
zc = [sum(1 for j in range(1,sr) if (mono[i+j-1]>=0)!=(mono[i+j]>=0)) for i in range(0, len(mono)-sr, sr)]
print(f"  per-sec RMS: min={min(es):.0f} max={max(es):.0f} ratio={max(es)/max(min(es),1):.1f}")
print(f"  per-sec ZCR: mean={sum(zc)/len(zc):.0f}  (sustained tone <800; music 1000-4000; noise 5000+)")
# L/R correlation
ml = sum(left)/len(left); mr = sum(right)/len(right)
num = sum((l-ml)*(r-mr) for l,r in zip(left,right))
den = math.sqrt(sum((l-ml)**2 for l in left) * sum((r-mr)**2 for r in right))
print(f"  L/R Pearson: {num/den:.3f}  (1.0=identical, 0=independent, ~0.5–0.9 typical for music)")
```

### Inspect safetensors keys without loading

```python
import struct, json
with open("model.safetensors", 'rb') as f:
    hl = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(hl).decode())
keys = [k for k in header.keys() if k != "__metadata__"]
print(f"Total keys: {len(keys)}")
prefixes = {}
for k in keys:
    p = k.split(".")[0]
    prefixes[p] = prefixes.get(p, 0) + 1
for p, c in sorted(prefixes.items()):
    print(f"  {p}: {c}")
```

## Side-quests deferred (not blocking audio investigation)

- **User-controllable diffusion settings UI** (steps, shift, method, precision, CFG interval). User asked for this; conditioning verification was prioritized. Now that almost everything is plumbed, exposing knobs is a good follow-up.
- **Cover-song generation** (5Hz LM models). User has these downloaded locally but they're for audio→audio conditioning and require:
  - LM model download specs in `ModelRegistry.builtInSpecs`
  - Audio token tokenizer (the `tokenizer.*` subtree we currently filter out)
  - LM forward pass to produce 25Hz hints
  - Wire `precomputed_lm_hints_25Hz` into the diffusion path
- **`isModelDownloaded` accuracy bug** for the LLM (`LLMModelManager`) — currently only checks `config.json` + `tokenizer.json`, missing the safetensors. Causes "downloaded" UI state while safetensors is still streaming.
- **Background download race** for `LyricWriterManager` — UI can show "Downloading 100%" with stuck phase + active model checkmark + .failed toast simultaneously.
