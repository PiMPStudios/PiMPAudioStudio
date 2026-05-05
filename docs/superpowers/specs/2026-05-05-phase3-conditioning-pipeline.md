# Phase 3: Conditioning Pipeline — Design Spec

## Overview

Implement the text/lyric/timbre conditioning pipeline in native MLX-Swift. This
phase delivers the `ConditioningActor` that transforms raw text prompts, lyrics,
and optional reference audio into the `encoderStates` and `contextLatents`
tensors that the Phase 2 `MusicPipeline` already accepts.

**Scope boundary:** Phase 3 does NOT include UI integration, prompt routing, or
lyric writing assistance. It delivers a testable actor that takes strings and
optional audio in, and produces conditioning tensors out.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Text encoder model | Qwen3-Embedding-0.6B (native port) | Same architecture as reference; no external model runtime dependency |
| Tokenizer | swift-tokenizers package | Handles BPE correctly; loads tokenizer.json directly |
| Lyric encoding | Dedicated 8-layer LyricEncoder | Lyrics get embedding-only from Qwen3, then specialized encoder |
| Timbre encoding | 4-layer TimbreEncoder + attention pooler | Extracts fixed timbre tokens from variable-length reference audio |
| Reference audio | Variable-length with silence fallback | Text-only generation works out of the box |
| Sequence packing | Simple concatenation | Correct for batch size 1; avoids unnecessary complexity |
| Architecture reuse | Shared RotaryEmbedding, RMSNorm from Phase 2 | Same base freq and head_dim; no duplication |

## Component 1: Qwen3Model (Text Encoder)

Standard decoder-architecture transformer used bidirectionally (no causal mask).
Only `last_hidden_state` is needed — no language modeling head.

### Architecture

```text
input_ids [B, T]
  → Embedding(151669, 1024)
  → 28 × Qwen3Layer (self-attention + SwiGLU MLP, pre-norm)
  → RMSNorm
  → hidden_states [B, T, 1024]
```

- **Layers:** 28
- **Hidden size:** 1024
- **Intermediate size:** 3072
- **Attention:** GQA — 16 query heads, 8 KV heads, head_dim=128
- **Position encoding:** RoPE, base=1,000,000, max_seq=32,768
- **Normalization:** RMSNorm, eps=1e-6
- **Activation:** SiLU (SwiGLU pattern)
- **Vocabulary:** 151,669 tokens
- **No causal mask** — bidirectional attention for embedding use

### Qwen3Layer

```text
x → RMSNorm → Qwen3Attention(GQA + RoPE) → residual
  → RMSNorm → SwiGLUMLP(1024 → 3072 → 1024) → residual
```

No cross-attention, no AdaLN. Simpler than DiTLayer.

### API

```swift
public class Qwen3Model: Module {
    public func callAsFunction(inputIds: MLXArray) -> MLXArray
    public func embedTokens(_ inputIds: MLXArray) -> MLXArray
}
```

- `callAsFunction` — full forward pass, returns `[B, T, 1024]`
- `embedTokens` — embedding table lookup only, returns `[B, T, 1024]`

## Component 2: LyricEncoder

Dedicated encoder for lyric token embeddings. Receives raw embeddings from
Qwen3's embedding table and processes through 8 transformer layers with an input
projection to the DiT hidden dimension.

### Architecture

```text
lyric_embeddings [B, T_lyric, 1024]
  → Linear(1024, 2048)
  → 8 × Qwen3Layer (self-attention + SwiGLU MLP, hidden_size=2048)
  → RMSNorm
  → lyric_hidden_states [B, T_lyric, 2048]
```

- **Layers:** 8
- **Hidden size:** 2048
- **Intermediate size:** 6144
- **Attention:** GQA — 16 query heads, 8 KV heads, head_dim=128
- **Position encoding:** RoPE, base=1,000,000
- **Max sequence:** 2048 tokens (lyric max length)

### API

```swift
public class LyricEncoder: Module {
    public func callAsFunction(
        embeddings: MLXArray,
        attentionMask: MLXArray?
    ) -> MLXArray
}
```

## Component 3: TimbreEncoder

Processes VAE-encoded reference audio into a compact set of timbre tokens via
self-attention followed by attention pooling.

### Architecture

```text
refer_audio_latents [B, 750, 64]
  → Linear(64, 2048)
  → 4 × Qwen3Layer (self-attention + SwiGLU MLP, hidden_size=2048)
  → AttentionPooler (2 layers, 5 learnable query tokens)
  → timbre_tokens [B, 5, 2048]
```

- **Layers:** 4 (encoder) + 2 (pooler)
- **Hidden size:** 2048
- **Intermediate size:** 6144
- **Attention:** GQA — 16 query heads, 8 KV heads, head_dim=128
- **Pool window size:** 5 (output token count)
- **Fixed input frames:** 750 (pad/truncate reference audio latents)

### Attention Pooler

The pooler uses learnable query tokens that attend to the encoded reference
sequence via cross-attention, compressing the full sequence into 5 tokens:

```text
queries: learnable [1, 5, 2048] (expanded to batch)
keys/values: encoded reference [B, 750, 2048]
→ cross-attention → [B, 5, 2048]
```

### Silence Fallback

When no reference audio is provided, pass `MLXArray.zeros([1, 750, 64])`. The
model was trained with silence as the "no timbre" conditioning signal.

### API

```swift
public class TimbreEncoder: Module {
    public func callAsFunction(
        latents: MLXArray,
        attentionMask: MLXArray?
    ) -> MLXArray
}
```

## Component 4: ConditionEncoder

Orchestrates the sub-encoders and concatenates all conditioning into a single
sequence for the DiT's cross-attention.

### Flow

```text
text_hidden [B, T_text, 1024]
  → Linear(1024, 2048) → text_projected [B, T_text, 2048]

lyric_hidden [B, T_lyric, 2048]  (from LyricEncoder)
timbre_tokens [B, 5, 2048]       (from TimbreEncoder)

Concatenate along axis=1:
  → encoder_hidden_states [B, T_text + T_lyric + 5, 2048]

Combined attention mask (concat of individual masks):
  → encoder_mask [B, T_combined]
```

**T_combined** = T_text (≤256) + T_lyric (≤2048) + 5 = up to 2309 tokens.

### API

```swift
public class ConditionEncoder: Module {
    public func callAsFunction(
        textHiddenStates: MLXArray,
        textMask: MLXArray?,
        lyricHiddenStates: MLXArray,
        lyricMask: MLXArray?,
        timbreTokens: MLXArray
    ) -> (encoderStates: MLXArray, mask: MLXArray)
}
```

## Component 5: ConditioningActor

Top-level actor that owns the full conditioning pipeline and provides the public
interface for generating conditioning tensors from raw inputs.

### Responsibilities

1. Tokenize text and lyrics (via swift-tokenizers)
2. Run Qwen3 forward pass on text tokens → text_hidden_states
3. Get lyric embeddings from Qwen3 embedding table → lyric_embeddings
4. Run LyricEncoder on lyric embeddings → lyric_hidden_states
5. VAE-encode reference audio (via VAEActor) → reference latents
6. Pad/truncate reference latents to 750 frames
7. Run TimbreEncoder → timbre_tokens
8. Run ConditionEncoder (project text + concatenate) → encoderStates
9. Prepare contextLatents (reference latents or silence)

### API

```swift
public actor ConditioningActor {
    public init(
        vae: VAEActor,
        qwen3: Qwen3Model,
        conditionEncoder: ConditionEncoder,
        lyricEncoder: LyricEncoder,
        timbreEncoder: TimbreEncoder,
        tokenizerURL: URL
    )

    public func encode(
        text: String,
        lyrics: String,
        referenceAudio: MLXArray?,
        precision: Precision
    ) async -> ConditioningResult

    public func loadWeights(
        qwen3Weights: [(String, MLXArray)],
        conditionWeights: [(String, MLXArray)]
    ) async
}

public struct ConditioningResult: @unchecked Sendable {
    public let encoderStates: MLXArray    // [B, T_combined, 2048]
    public let encoderMask: MLXArray      // [B, T_combined]
    public let contextLatents: MLXArray   // [B, T_frames, 128]
}
```

### Context Latents Construction

The DiT expects `contextLatents [B, T, 128]` which is the concatenation of:
- `src_latents [B, T, 64]` — source audio latents (zeros for text-only, LM hints
  for covers)
- `chunk_mask [B, T, 64]` — expanded mask indicating which frames are valid
  (ones for generation region, zeros for padding)

For text-only generation: `contextLatents = zeros([1, T_frames, 128])`

For cover/style transfer: `contextLatents = cat([lm_hints, chunk_mask], dim=-1)`
where `lm_hints` come from the tokenize/detokenize path (future scope — Phase 3
just needs the zeros path and basic cover path via VAE-encoded reference).

### Integration

```swift
let cond = await conditioningActor.encode(
    text: "A dreamy synthwave track with ethereal vocals",
    lyrics: "[Verse]\nFloating through the neon lights...",
    referenceAudio: nil,
    precision: .float16
)
let result = try await musicPipeline.generate(
    encoderStates: cond.encoderStates,
    contextLatents: cond.contextLatents,
    duration: 30.0,
    config: config
)
```

## Component 6: Tokenizer Wrapper

Thin wrapper around swift-tokenizers that loads the Qwen3 `tokenizer.json` and
provides typed encoding for text and lyrics.

### API

```swift
public struct AnvilTokenizer {
    public init(configURL: URL) throws

    public func encodeText(_ text: String, maxLength: Int = 256) -> (ids: MLXArray, mask: MLXArray)
    public func encodeLyrics(_ lyrics: String, maxLength: Int = 2048) -> (ids: MLXArray, mask: MLXArray)
}
```

## Weight Conversion

### Qwen3 Weights

Source: HuggingFace `Qwen3-Embedding-0.6B` checkpoint (safetensors format).

Key remapping:
- `model.embed_tokens.weight` → `embedTokens.weight`
- `model.layers.N.self_attn.{q,k,v,o}_proj.weight` → `layers.N.attention.{q,k,v,o}Proj.weight`
- `model.layers.N.mlp.{gate,up,down}_proj.weight` → `layers.N.mlp.{gate,up,down}Proj.weight`
- `model.layers.N.input_layernorm.weight` → `layers.N.inputNorm.weight`
- `model.layers.N.post_attention_layernorm.weight` → `layers.N.postNorm.weight`
- `model.norm.weight` → `norm.weight`

No axis transpositions needed — Linear weights are the same layout in PyTorch and
MLX.

### Condition Encoder Weights

Source: ACE-Step v1.5 DiT checkpoint. The LyricEncoder, TimbreEncoder, and
ConditionEncoder are part of the DiT state dict under specific prefixes:

- `condition_encoder.text_projection.*` → ConditionEncoder text projection
- `condition_encoder.lyric_encoder.*` → LyricEncoder
- `condition_encoder.timbre_encoder.*` → TimbreEncoder
- `condition_encoder.timbre_attention_pooler.*` → TimbreEncoder attention pooler

These use the same naming patterns as the DiT (Linear weights, RMSNorm, etc.) so
no axis transpositions needed.

## Package Structure

```text
AnvilCore/Sources/AnvilCore/Inference/
├── Conditioning/
│   ├── Qwen3Model.swift
│   ├── Qwen3Components.swift
│   ├── LyricEncoder.swift
│   ├── TimbreEncoder.swift
│   ├── ConditionEncoder.swift
│   ├── ConditioningActor.swift
│   └── Tokenizer.swift
├── Weights/
│   ├── Qwen3NameMap.swift
│   └── ConditionEncoderNameMap.swift
```

## Testing Strategy

1. **Qwen3Model** — Feed known token IDs, verify output shape `[B, T, 1024]`.
   Test `embedTokens` returns `[B, T, 1024]` without running full model.
2. **LyricEncoder** — Feed random embeddings `[1, 100, 1024]`, verify output
   shape `[1, 100, 2048]`.
3. **TimbreEncoder** — Feed zeros `[1, 750, 64]`, verify output shape
   `[1, 5, 2048]`.
4. **ConditionEncoder** — Feed all three inputs, verify concatenated shape and
   mask dimensions.
5. **ConditioningActor** — End-to-end: text + lyrics + nil reference → verify
   `ConditioningResult` shapes. Test silence fallback path.
6. **Tokenizer** — Verify token IDs and attention masks for known inputs.
7. **Weight conversion** — Verify key remapping and shape validation.

## Dependencies

- **mlx-swift** (existing)
- **swift-tokenizers** (new) — HuggingFace BPE tokenizer, loads tokenizer.json

## Non-Goals (Future Phases)

- UI integration (Phase 4)
- Prompt routing / IntentRouter (Phase 4)
- LyricWriter LLM assistance (Phase 4)
- DSP audio editor (Phase 5)
- AI audio editor / inpainting (Phase 6)
- LoRA fine-tuning (Phase 7)
