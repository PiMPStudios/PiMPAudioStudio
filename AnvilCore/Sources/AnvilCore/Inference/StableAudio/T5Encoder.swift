// AnvilCore/Sources/AnvilCore/Inference/StableAudio/T5Encoder.swift
//
// T5-base text encoder used by Stable Audio Open 1.0. SAO's `model_config.json`
// declares `t5_model_name = "t5-base"` with `max_length = 128`, so the shapes
// here mirror HuggingFace's `t5-base`:
//
//   vocab_size      = 32128
//   d_model         = 768
//   num_layers      = 12
//   num_heads       = 12
//   d_kv            = 64        (per-head, d_model / num_heads)
//   d_ff            = 3072
//   relative_buckets = 32
//   relative_max_distance = 128
//
// This file is the structural skeleton: every module is declared with the
// correct parameter shapes so weight loading via the snake_case→camelCase
// `WeightConverter` path resolves cleanly. The forward pass is a stub that
// returns a zero tensor of the correct output shape so the rest of the
// pipeline (projection model → DiT conditioning) can be plumbed without
// crashing on type mismatches.
//
// Real T5 forward math (relative-bias attention, gated GLU MLPs, layer
// norms, encoder stack) lands in a follow-up commit when full Stable Audio
// Open re-architecture is enabled. Until then the spec is marked
// `isRunnable: false` so users can't accidentally run a stubbed pipeline.

import Foundation
import MLX
import MLXNN

public final class T5Config: Sendable {
    public let vocabSize: Int
    public let dModel: Int
    public let dKV: Int
    public let dFF: Int
    public let numLayers: Int
    public let numHeads: Int
    public let relativeAttentionNumBuckets: Int
    public let relativeAttentionMaxDistance: Int
    public let layerNormEpsilon: Float
    public let dropoutRate: Float

    public init(
        vocabSize: Int = 32128,
        dModel: Int = 768,
        dKV: Int = 64,
        dFF: Int = 3072,
        numLayers: Int = 12,
        numHeads: Int = 12,
        relativeAttentionNumBuckets: Int = 32,
        relativeAttentionMaxDistance: Int = 128,
        layerNormEpsilon: Float = 1e-6,
        dropoutRate: Float = 0.1
    ) {
        self.vocabSize = vocabSize
        self.dModel = dModel
        self.dKV = dKV
        self.dFF = dFF
        self.numLayers = numLayers
        self.numHeads = numHeads
        self.relativeAttentionNumBuckets = relativeAttentionNumBuckets
        self.relativeAttentionMaxDistance = relativeAttentionMaxDistance
        self.layerNormEpsilon = layerNormEpsilon
        self.dropoutRate = dropoutRate
    }

    public static let t5Base = T5Config()
}

// MARK: - T5LayerNorm
//
// T5 uses a simplified RMS-norm variant: x * rsqrt(mean(x^2) + eps), with no
// bias and a learned per-feature gain. MLXNN.RMSNorm matches this exactly.

// MARK: - T5DenseGatedActDense (FFN)
//
// T5 v1.1 / mT5 uses gated-GELU FFNs:
//   wi_0(x)  = Linear(d_model, d_ff, bias=False)
//   wi_1(x)  = Linear(d_model, d_ff, bias=False)
//   gated_x  = gelu(wi_0(x)) * wi_1(x)
//   out      = wo(gated_x)   // Linear(d_ff, d_model, bias=False)
public final class T5GatedMLP: Module, UnaryLayer {
    @ModuleInfo(key: "wi_0") public var wi0: Linear
    @ModuleInfo(key: "wi_1") public var wi1: Linear
    public var wo: Linear

    public init(config: T5Config) {
        self._wi0.wrappedValue = Linear(config.dModel, config.dFF, bias: false)
        self._wi1.wrappedValue = Linear(config.dModel, config.dFF, bias: false)
        self.wo = Linear(config.dFF, config.dModel, bias: false)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // gelu approx; T5v1.1 uses GELU "new" / GELU
        wo(gelu(wi0(x)) * wi1(x))
    }
}

// MARK: - T5Attention (self-attention with relative-position bias)
//
// Skeleton: declares the four projection matrices and the relative-bias
// embedding. Real bucket-bias computation pending.
public final class T5SelfAttention: Module {
    public var q: Linear
    public var k: Linear
    public var v: Linear
    public var o: Linear
    /// Only the FIRST encoder block has this — subsequent blocks reuse the
    /// position-bias tensor computed by block 0. The skeleton still declares
    /// it Optional on every block so weight-load name maps line up; only
    /// block 0's tensor is loaded with real weights.
    @ModuleInfo(key: "relative_attention_bias")
    public var relativeAttentionBias: Embedding?

    public init(config: T5Config, hasRelativeBias: Bool) {
        let inner = config.numHeads * config.dKV
        self.q = Linear(config.dModel, inner, bias: false)
        self.k = Linear(config.dModel, inner, bias: false)
        self.v = Linear(config.dModel, inner, bias: false)
        self.o = Linear(inner, config.dModel, bias: false)
        self._relativeAttentionBias.wrappedValue = hasRelativeBias
            ? Embedding(embeddingCount: config.relativeAttentionNumBuckets, dimensions: config.numHeads)
            : nil
    }

    /// Stub forward: returns the input unchanged (identity) so downstream
    /// layers don't crash on shape mismatch. Replace with real
    /// scaled-dot-product attention + relative-position bias when enabling.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // TODO: implement T5 self-attention with relative-bias.
        // For now: q*k^T/sqrt(d_k) + bias, softmax, @v, then o-projection
        // is intentionally unwired. Identity passthrough keeps shapes valid.
        x
    }
}

// MARK: - T5Block (encoder layer = self-attn + FFN, each pre-normed)
public final class T5Block: Module {
    public var layer0_norm: RMSNorm
    public var layer0_attn: T5SelfAttention
    public var layer1_norm: RMSNorm
    public var layer1_mlp: T5GatedMLP

    public init(config: T5Config, hasRelativeBias: Bool) {
        self.layer0_norm = RMSNorm(dimensions: config.dModel, eps: config.layerNormEpsilon)
        self.layer0_attn = T5SelfAttention(config: config, hasRelativeBias: hasRelativeBias)
        self.layer1_norm = RMSNorm(dimensions: config.dModel, eps: config.layerNormEpsilon)
        self.layer1_mlp = T5GatedMLP(config: config)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Pre-norm residual blocks. Stub forward uses real RMSNorm + MLP
        // (those layers are real); attention is currently identity so this
        // is mostly an MLP-only forward. Sufficient to keep shapes correct
        // through the encoder stack.
        var h = x
        let a = layer0_attn(layer0_norm(h))
        h = h + a
        let f = layer1_mlp(layer1_norm(h))
        h = h + f
        return h
    }
}

// MARK: - T5Encoder

public final class T5Encoder: Module {
    public let config: T5Config
    @ModuleInfo(key: "shared")
    public var shared: Embedding
    public var blocks: [T5Block]
    @ModuleInfo(key: "final_layer_norm")
    public var finalLayerNorm: RMSNorm

    public init(config: T5Config = .t5Base) {
        self.config = config
        self._shared.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.dModel)
        self.blocks = (0..<config.numLayers).map { idx in
            T5Block(config: config, hasRelativeBias: idx == 0)
        }
        self._finalLayerNorm.wrappedValue = RMSNorm(dimensions: config.dModel, eps: config.layerNormEpsilon)
    }

    /// Encode a batch of token-id sequences into hidden states.
    /// - Parameters:
    ///   - inputIds: `[B, L]` Int32 token ids
    ///   - attentionMask: `[B, L]` 0/1 mask (currently unused — full T5
    ///     forward will fold this into attention).
    /// - Returns: `[B, L, dModel]` encoder hidden states.
    public func callAsFunction(inputIds: MLXArray, attentionMask: MLXArray? = nil) -> MLXArray {
        _ = attentionMask // pending real attention impl
        var h = shared(inputIds)
        for block in blocks {
            h = block(h)
        }
        h = finalLayerNorm(h)
        return h
    }
}
