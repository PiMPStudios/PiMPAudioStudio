# Cover-Song Generation — Design Spec

> Add audio→audio cover-song generation to AnvilApp's ACE-Step pipeline. The user provides an input audio clip + a text prompt; the model produces a "cover" of the source audio that follows the prompt's stylistic direction. Phase 1 (model spec download infrastructure) is already shipped; this spec covers Phase 2 (inference + UI).

**Goal:** End-to-end working cover-song generation: drop a WAV file into the UI, type a prompt, hit Generate, get a cover-styled output.

**Architecture:** Adds three new MLX-Swift modules (audio tokenizer, audio detokenizer, ACE-Step LM), wires them into a new conditioning path in `ConditioningActor.encode`, and surfaces input-audio + cover-strength controls in the Generate tab. The DiT and VAE remain unchanged — the new components plug in upstream of the diffusion loop.

**Pre-conditions:** Phase 1 already shipped (commit `1f9dc97`):
- `ModelCategory.aceStepLM` exists
- `ace-step-lm-0.6b` and `ace-step-lm-4b` specs in `ModelRegistry.builtInSpecs`
- Settings UI section for picking + downloading LM variants
- The audio tokenizer/detokenizer **weights are already in the DiT bundle** (`tokenizer.*` 32 keys, `detokenizer.*` 28 keys) — currently filtered out by `mapDiTKey` returning nil for non-`decoder.` keys. Phase 2 keeps them.

---

## 1. Scope

### In scope (Phase 2)

- Load audio tokenizer + detokenizer weights from the DiT bundle (already-downloaded files)
- Load LM weights from the user's chosen LM variant
- Run input audio → VAE encode → audio tokenizer → discrete codes → LM → 25 Hz hints → diffusion guidance
- New Generate tab UI: drop zone for input audio, cover strength slider, LM variant indicator
- One-shot inference path producing a 30s cover

### Out of scope (defer to Phase 3 if needed)

- "Repaint" / "Inpaint" / partial-audio editing modes
- Real-time / streaming inference
- Audio-codes input (user provides pre-tokenized codes instead of audio)
- Lyric-aligned covers (using the lyrics field with audio input)
- Long-form covers > 47s
- Custom timestep schedules per-cover

---

## 2. Upstream reference

The whole flow is in `acestep/models/turbo/modeling_acestep_v15_turbo.py`:

```
input_audio (waveform)
  ↓ VAE encode
  → src_latents [B, T_lat, 64]            (T_lat ≈ duration × 25)
  ↓ tokenize.forward(audio, silence_latent, attention_mask)
  → lm_hints_5Hz [B, T_lat // pool_window_size, hidden_size]
  → indices [B, T_lat // pool_window_size, num_quantizers]
  → llm_mask
  ↓ detokenize.forward(lm_hints_5Hz)
  → lm_hints_25Hz [B, T_lat, audio_acoustic_hidden_dim]
  ↓ src_latents = where(is_covers, lm_hints_25Hz, src_latents)
  → conditioning fed into diffusion
```

Key code locations to mirror:

| Component | Upstream file | Lines |
|---|---|---|
| AudioTokenizer (FSQ-based) | `acestep/models/turbo/modeling_acestep_v15_turbo.py` | ~1100–1220 |
| AudioTokenDetokenizer | same | ~870–990 |
| Tokenize/detokenize call site | same (DiT.forward) | ~1670–1690 |
| LM forward (Qwen-style) | from the LM checkpoint config | per `config.json` of `ACE-Step/acestep-5Hz-lm-0.6B` |

---

## 3. New components

### 3.1 `AudioTokenizer` (Swift module)

Mirrors upstream's `AceStepTokenizer`. FSQ (Finite Scalar Quantization) bottleneck that converts continuous audio latents → discrete codes.

```swift
public class AudioTokenizer: Module {
    let config: AceStepConfig          // brought in from VAE config.json + DiT config.json
    let inputProjection: Linear
    let layers: [AceStepEncoderLayer]   // num_audio_decoder_hidden_layers (24 in turbo)
    let norm: RMSNorm
    let rotaryEmbedding: RotaryEmbedding
    let quantizer: FSQQuantizer

    init(config: AceStepConfig)

    /// Returns:
    ///   lmHints5Hz: [B, T // pool_window_size, hidden_size]
    ///   indices: [B, T // pool_window_size, num_quantizers]
    ///   mask: [B, T // pool_window_size]
    public func callAsFunction(
        audioLatents: MLXArray,         // [B, T, audio_acoustic_hidden_dim]
        silenceLatent: MLXArray,        // [1, hidden_size] from `silence_latent.pt` in turbo dir
        attentionMask: MLXArray?
    ) -> (lmHints5Hz: MLXArray, indices: MLXArray, mask: MLXArray)
}

public class FSQQuantizer: Module {
    /// Levels per scalar: [8, 8, 8, 5, 5, 5] from config.fsq_input_levels
    let levels: [Int]
    /// Total codebook size = product(levels) = 64000 for the v1.5 turbo config
    var codebookSize: Int { levels.reduce(1, *) }

    public func quantize(_ x: MLXArray) -> (quantized: MLXArray, indices: MLXArray)
    public func getOutputFromIndices(_ indices: MLXArray) -> MLXArray
}
```

**Weights source:** the DiT bundle's `tokenizer.*` subtree (32 keys). Map via a new `AudioTokenizerNameMap.mapKey` that handles the prefix and snake→camelCase the same way `ConditionEncoderNameMap.mapKey` does.

**Silence latent source:** `silence_latent.pt` (3.84 MB) in each DiT variant's repo. This is a precomputed VAE-encoded latent of silence that the tokenizer mixes in to identify silent regions. Add to the DiT spec's file list:

```swift
ModelFileSpec(
    remotePath: "acestep-v15-turbo/silence_latent.pt",  // or root for sft/base
    localName: "silence_latent.pt",
    sizeBytes: 3_841_215
)
```

> Note: `.pt` is a PyTorch pickle file. We need to either parse it via a minimal pickle-to-safetensors conversion at runtime, or pre-convert and ship as `silence_latent.safetensors` in our app bundle. Path of least resistance: ship it pre-converted as a Swift resource since it's small + fixed.

### 3.2 `AudioTokenDetokenizer` (Swift module)

Mirrors upstream's `AudioTokenDetokenizer`. Expands 5 Hz LM hints into 25 Hz acoustic hints.

```swift
public class AudioTokenDetokenizer: Module {
    let embedTokens: Linear              // hidden_size → hidden_size
    let layers: [AceStepEncoderLayer]    // num_attention_pooler_hidden_layers (2 in turbo)
    let norm: RMSNorm
    let rotaryEmbedding: RotaryEmbedding
    let specialTokens: MLXArray          // [1, pool_window_size, hidden_size] = [1, 5, 2048]
    let projOut: Linear                  // hidden_size → audio_acoustic_hidden_dim (64)

    init(config: AceStepConfig)

    /// Input:  [B, T_5Hz, hidden_size]
    /// Output: [B, T_5Hz * pool_window_size, audio_acoustic_hidden_dim]
    public func callAsFunction(_ lmHints5Hz: MLXArray) -> MLXArray
}
```

**Weights source:** the DiT bundle's `detokenizer.*` subtree (28 keys).

> **Note:** the `special_tokens` field here is what we currently treat as `attentionPooler.queryTokens` in `TimbreEncoder` and skip in `ConditionEncoderNameMap`. After this phase lands, the `detokenizer.special_token` weight finally gets a real home (in `AudioTokenDetokenizer.specialTokens`) and the `TimbreEncoder.AttentionPooler` skip can be revisited — though that's a separate question (timbre encoder stays for the no-cover path).

### 3.3 `AceStepLM` (Swift module)

The 5 Hz audio language model. Architecture is Qwen-style decoder-only transformer (the safetensors keys match Qwen's structure: `embed_tokens`, `layers.N.{input_layernorm, post_attention_layernorm, mlp.{down,gate,up}_proj, self_attn.{q,k,v,o}_{proj,norm}}`, `norm`).

Different sizes per variant (from `config.json` in each repo):

| Variant | hidden_size | num_layers | intermediate | num_q_heads | num_kv_heads | head_dim |
|---|---|---|---|---|---|---|
| 0.6B | TBD — read at load | TBD | TBD | TBD | TBD | TBD |
| 4B | TBD — read at load | TBD | TBD | TBD | TBD | TBD |

```swift
public class AceStepLM: Module {
    let embedTokens: Embedding
    let layers: [Qwen3Layer]              // reuse our existing Qwen3Layer
    let norm: RMSNorm
    let rotaryEmbedding: RotaryEmbedding
    // No lm_head — we want hidden states, not next-token logits

    init(config: AceStepLMConfig)         // loaded from the variant's config.json

    /// Forward through the LM and return final hidden states.
    /// Input: audio_codes [B, T_5Hz] (integer indices into the FSQ codebook)
    /// Output: hidden_states [B, T_5Hz, hidden_size]
    public func callAsFunction(audioCodes: MLXArray, attentionMask: MLXArray?) -> MLXArray
}

public struct AceStepLMConfig {
    let hiddenSize: Int
    let numLayers: Int
    let intermediateSize: Int
    let numAttentionHeads: Int
    let numKeyValueHeads: Int
    let headDim: Int
    let rmsNormEps: Float
    let vocabSize: Int
    let maxPositionEmbeddings: Int
    let ropeTheta: Float

    /// Loads the config from a variant's config.json on disk.
    public static func load(from configURL: URL) throws -> AceStepLMConfig
}
```

**Weights source:** the LM variant's `model.safetensors` (or sharded `model-00001-of-NNNNN.safetensors`).

**Sharded loading:** We need a `loadSharded` path in `WeightConverter` that reads `model.safetensors.index.json` and assembles weights from multiple shards. Currently `WeightConverter.loadSafetensors(from:)` handles single-file only. The 4B variant is sharded across 2 files.

### 3.4 `CoverSongCoordinator` (new actor)

Owns the cover-generation flow end-to-end. Sits alongside `ACEStepGenerationService` / `SFXGenerationService` as a third sub-service of `GenerationService`.

```swift
@MainActor
final class CoverSongCoordinator {
    private let pipeline: MusicPipeline
    private let conditioning: ConditioningActor
    private let weightConverter: WeightConverter
    private let registry: ModelRegistry

    private var audioTokenizer: AudioTokenizer?
    private var detokenizer: AudioTokenDetokenizer?
    private var lm: AceStepLM?
    private var currentlyLoadedLMID: String?

    enum Event {
        case loadingWeights(progress: Float)
        case encodingSource         // VAE encode of input audio
        case tokenizing             // audio → codes
        case lmInference            // codes → 25Hz hints
        case generating(step: Int, totalSteps: Int, eta: TimeInterval)
        case decoding
        case result(AudioResult)
    }

    func generate(
        request: GenerationRequest,
        inputAudio: URL,
        coverStrength: Float            // 0.0 = pure prompt, 1.0 = pure source
    ) -> AsyncThrowingStream<Event, Error>
}
```

The coordinator reuses `pipeline.vae`, `pipeline.diffusion`, and `conditioning` from the existing infrastructure. The DiT and text-conditioning paths don't change at all — only the source-latent path is rerouted through tokenizer→LM→detokenizer.

---

## 4. Data flow

```
GenerateView
  → user drops input.wav, types prompt, sets coverStrength=0.7
  → GenerationRequest{ text, lyrics, inputAudio: URL, coverStrength: 0.7, … }
  ↓
GenerationService.generate
  ↓ shouldUseCoverPath(request)?           // request.inputAudio != nil
  ├─ yes → coverSongCoordinator.generate()
  └─ no  → existing aceStepService / sfxService

CoverSongCoordinator.generate (yields events)
  1. ensureWeightsLoaded()                  // loads LM if active variant changed
       - .loadingWeights events
  2. let audio = try AudioReader.load(inputAudio)
       - resamples to 48kHz stereo if needed
  3. let srcLatents = await pipeline.vae.encodeMean(audio: audio, …)
       - emit .encodingSource
  4. let (lmHints5Hz, _, _) = try audioTokenizer(srcLatents, silenceLatent, mask)
       - emit .tokenizing
  5. let lmHidden = lm(audioCodesFromIndices(lmHints5Hz), mask)
       - emit .lmInference
       - lmHidden shape: [1, T_5Hz, lm.hiddenSize]
  6. let lmHints25Hz = detokenizer(lmHidden)
       - shape [1, T_5Hz * pool_window_size, audio_acoustic_hidden_dim]
       - = [1, T_lat, 64]
  7. let blendedLatents = (1 - coverStrength) * srcLatents
                        + coverStrength * lmHints25Hz
       - or: if coverStrength == 1.0, just use lmHints25Hz directly
  8. let conditioningResult = await conditioning.encode(
         text: request.text, lyrics: request.lyrics,
         referenceAudio: nil, duration: …, precision: …,
         srcLatentsOverride: blendedLatents      // NEW parameter
     )
       - context_latents now uses blendedLatents instead of zeros
  9. forward into pipeline.generate as today (DiT diffusion + VAE decode)
       - emit .generating, .decoding, .result
```

The crucial `srcLatentsOverride` parameter on `ConditioningActor.encode` replaces the current zero-filled `contextLatents` with the blended source/LM-hint latents.

### 4.1 `ConditioningActor.encode` change

Currently:

```swift
let tFrames = Self.latentFrames(forDuration: duration)
let contextLatents = MLXArray.zeros([1, tFrames, 128])   // text-only path
```

Phase 2:

```swift
let contextLatents: MLXArray
if let override = srcLatentsOverride {
    // Pad/truncate to target frame count, concat with chunk_mask of 1s
    let padded = padToLatentFrames(override, target: tFrames)
    let chunkMask = MLXArray.ones([1, tFrames, 64])      // 1s = "use this latent"
    contextLatents = concatenated([padded, chunkMask], axis: -1)
} else {
    contextLatents = MLXArray.zeros([1, tFrames, 128])
}
```

This is closer to upstream's actual behavior — the chunk_mask channels get set to 1.0 where source latents are present and 0.0 where they're padded.

### 4.2 Cover strength interpretation

`coverStrength: Float` in [0, 1]:

- `0.0` → ignore source, run pure text-to-music (degenerates to existing path, but going through cover code)
- `1.0` → use only LM hints, ignore raw source latents
- intermediate → blend per-frame

Upstream's `audioCoverStrength` controls how many of the first `N` diffusion steps use cover-conditioned latents vs uncover-conditioned latents (line 51 in DiffusionActor.generate already references this). We can reuse that mechanism — early steps get `lmHints25Hz`, later steps switch to whatever the diffusion produces. That gives smoother style transfer.

---

## 5. UI changes

### 5.1 GenerateView additions

A new "Source Audio" subsection under the prompt:

```
┌─ Prompt ─────────────────────────────────┐
│ slow dark guitar riff                    │
└──────────────────────────────────────────┘
┌─ Lyrics ─────────────────────────── [Write Lyrics]
│ verse one la la                          │
└──────────────────────────────────────────┘
┌─ Source Audio (optional) ────────────────┐
│ ┌──────────────────────────────────────┐ │
│ │ Drop a WAV/MP3 here, or click Browse │ │
│ │   acoustic.wav · 0:30 · 48kHz stereo │ │
│ └──────────────────────────────────────┘ │
│ Cover Strength    [══════•───] 0.70      │
│ LM Variant        ACE-Step LM 0.6B  ▾    │
└──────────────────────────────────────────┘

[ Generate ]
```

When source audio is dropped:

- Cover Strength slider becomes enabled (0.0–1.0, default 0.7)
- LM Variant picker shows downloaded LM variants only; greyed out if none downloaded
- Generate button label changes to "Generate Cover"
- Existing duration / seed / guidance settings still apply

When source audio cleared, controls return to the text-to-music path.

### 5.2 GenerationRequest additions

```swift
struct GenerationRequest {
    // existing fields…
    let inputAudio: URL?           // nil = text-to-music path
    let coverStrength: Float       // 0..1, ignored when inputAudio == nil
}
```

### 5.3 GenerateView state additions

```swift
@State private var inputAudio: URL?
@State private var coverStrength: Double = 0.7
@State private var selectedLMID: String?
```

Plus a drop-target NSItemProvider handler for `.audio` UTType.

---

## 6. Phased implementation plan

### Phase 2A — Foundation (no UI yet)

1. **Add `silence_latent.pt` to DiT specs.** Each variant's repo has it; small file (~3.8 MB). Pre-convert to safetensors as a Swift bundle resource OR add minimal pickle parser. Recommend: Swift bundle resource for simplicity.
2. **Update `mapDiTKey` to keep `tokenizer.*` and `detokenizer.*` keys.** Currently filtered out. Add new entries that pass them through to a new `AudioTokenizerNameMap.mapKey` and `AudioTokenDetokenizerNameMap.mapKey`.
3. **Implement `FSQQuantizer`** (small — basically `quantize` and `get_output_from_indices`).
4. **Implement `AudioTokenizer`** module with weight loading via `WeightConverter.convertAudioTokenizerWeights`.
5. **Implement `AudioTokenDetokenizer`** module similarly.
6. **Implement `AceStepLMConfig.load(from:)`** (read JSON config from disk).
7. **Implement `AceStepLM`** module reusing `Qwen3Layer`.
8. **Add sharded safetensors loading** to `WeightConverter` (read index.json, assemble multiple shards). Needed for the 4B variant.
9. **Add `AceStepLMNameMap.mapKey`** for the LM variant's snake→camel + `embed_tokens → embeddingTable` etc. (likely identical to existing `Qwen3NameMap.mapKey`; consider parameterizing).
10. **Unit tests for each module.** Audio tokenizer round-trip (encode → decode → compare); LM forward shape check.

### Phase 2B — Coordinator + Conditioning hook

1. **Add `srcLatentsOverride` parameter to `ConditioningActor.encode`.** Gate the contextLatents construction on it.
2. **Implement `CoverSongCoordinator`** with the data flow described above.
3. **Add `inputAudio` + `coverStrength` to `GenerationRequest`.**
4. **Update `GenerationService.generate(request:)`** routing — if `request.inputAudio != nil`, route to `CoverSongCoordinator`.
5. **Wire `CoverSongCoordinator` into `AppServices.makeGenerationService()`.**
6. **Test end-to-end with a hardcoded input audio file** before any UI work.

### Phase 2C — UI

1. **Source Audio drop zone** in GenerateView with NSItemProvider handler.
2. **Cover Strength slider** + LM Variant picker.
3. **Generate button label** swaps to "Generate Cover" when input audio is set.
4. **Active LM variant tracking** — persist in `ModelRegistry.activeModels[.aceStepLM]`.

### Phase 2D — Polish

1. **Audio file readers** — expand support beyond WAV (MP3, M4A) via AVFoundation.
2. **Cover strength step-based interpretation** matching upstream's `audio_cover_strength * num_steps` behavior.
3. **Memory management** — `unloadAllWeights()` should also unload the LM (it can be 4–8 GB).
4. **CHANGELOG entry.**

---

## 7. Open questions

These need answers before Phase 2A starts; can be punted until we hit them:

1. **`silence_latent.pt` format** — confirm it's a single `[1, hidden_size]` tensor by inspecting the file. If so, ship as bundle resource; if it's more complex, parse via minimal pickle reader.
2. **`config.fsq_input_levels = [8, 8, 8, 5, 5, 5]` math** — verify FSQ codebook size 8×8×8×5×5×5 = 64000 matches the `vocab_size: 64003` in the LM config (3 extra for special tokens). If not, the indexing scheme is different.
3. **LM 4B sharded loading** — verify that index.json's `weight_map` field is sufficient to assemble. Might need to handle dtype downcasting if shards are bf16.
4. **DiT model bundle vs LM weight overlap** — does the DiT bundle's `tokenizer.*` need to come from a specific DiT variant? Or are tokenizer/detokenizer architecturally identical across turbo/full/sft? Inspect each variant's bundle to confirm.
5. **Audio resampling** — input audio might be at any sample rate. Add a resampler that goes to 48kHz stereo before VAE encode. Use AVFoundation or write a minimal resampler.
6. **Cover strength UI default** — 0.7 is a guess; verify against upstream's typical demo settings.

---

## 8. Migration / rollback

This is purely additive. No existing code paths change behavior:

- Phase 2A adds new modules + weight key handling. mapDiTKey gains tokenizer/detokenizer cases but still returns nil for everything else.
- Phase 2B adds `srcLatentsOverride: MLXArray? = nil` as an optional parameter — existing callers continue passing nothing, behavior unchanged.
- Phase 2C adds UI controls that default to "no source audio" — existing prompts work identically.
- `GenerationService` gets a third routing branch but only triggers on `request.inputAudio != nil`.

Rollback: revert each phase's commits independently. Phase 2A's module + key changes are revertable in isolation since nothing calls them yet.

---

## 9. Test plan

### Phase 2A tests

- Audio tokenizer weight load: `matched=N` count from diagnostic should equal `supplied=N` (modulo runtime buffers like rotary)
- Detokenizer weight load: same
- LM weight load (0.6B): same; total should be ~310 keys like Qwen3
- LM weight load (4B sharded): all shards merge cleanly; total key count > 0.6B
- FSQ round-trip: `quantizer.getOutputFromIndices(quantizer.quantize(x).indices)` should be approximately equal to `quantize(x).quantized` for a synthetic input
- Audio tokenizer end-to-end shape: input `[1, 750, 64]` → output `[1, 150, 2048]` (T // pool_window_size = 150)
- Detokenizer end-to-end shape: input `[1, 150, 2048]` → output `[1, 750, 64]`

### Phase 2B tests

- CoverSongCoordinator end-to-end with a known input audio + simple prompt — verify `result(AudioResult)` event fires
- Output WAV is 30s, 48kHz, stereo, full amplitude
- Latent magnitudes through the pipeline are healthy (use the same diagnostic logging from `DiffusionActor`)

### Phase 2C tests

- Drop a WAV file → UI displays filename + duration + sample rate
- Slider: dragging updates `coverStrength` in real time
- LM picker: only shows downloaded LM variants
- Generate button: label switches based on `inputAudio` presence
- Hit Generate → all the right events fire in order

### Phase 2D / acceptance

- Cover a 30s acoustic guitar WAV with prompt "rock distortion electric guitar" — output should clearly retain the source's rhythmic/melodic structure but in the new style
- Cover the same source with prompt "ambient pad" — output should be ambient texture following the source's contour
- Memory: total RAM during 4B-LM cover should fit within typical M-series Mac memory; if not, document in user-facing UI

---

## 10. Effort estimate

Rough sizing per phase, contingent on no audio quality bugs lurking in the existing pipeline (we're still chasing one):

| Phase | Effort | Risk |
|---|---|---|
| 2A | 1–2 days | Medium — sharded loading + FSQ are fresh code |
| 2B | 1 day | Low — mostly plumbing |
| 2C | 0.5 day | Low — standard SwiftUI |
| 2D | 0.5–1 day | Low — polish + docs |

**Total: 3–5 dev days for a working cover-song generation feature.** Less if the pipeline pieces compose cleanly; more if FSQ math or sharded loading hits edge cases.

> Caveat: this estimate assumes the audio-quality bug from `docs/ace-step-audio-quality-investigation.md` is resolved by then — cover generation builds on the same DiT path, so any conditioning-side bug there will also affect covers. Recommend: don't start Phase 2A until text-to-music produces prompt-aligned output.
