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
// Full encoder-only forward pass: embedding lookup → 12 pre-norm transformer
// blocks (gated-GELU FFN, self-attention with learned relative-position bias
// propagated from block 0) → RMS final norm → [B, L, 768] hidden states.

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

// MARK: - T5DenseActDense (FFN)
//
// T5 v1.0 (the variant used by Stable Audio Open) uses a standard dense-ReLU-dense FFN:
//   wi(x)  = Linear(d_model, d_ff, bias=False)
//   out    = wo(relu(wi(x)))  // Linear(d_ff, d_model, bias=False)
//
// HF checkpoint key: DenseReluDense.wi.weight / DenseReluDense.wo.weight
// T5NameMap routes both to layer1_mlp.wi.weight / layer1_mlp.wo.weight.
public final class T5DenseActDense: Module, UnaryLayer {
    public var wi: Linear
    public var wo: Linear

    public init(config: T5Config) {
        self.wi = Linear(config.dModel, config.dFF, bias: false)
        self.wo = Linear(config.dFF, config.dModel, bias: false)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        wo(relu(wi(x)))
    }
}

// MARK: - T5Attention (self-attention with relative-position bias)
//
// T5 quirks vs standard attention:
//   • No 1/sqrt(d_k) scale on Q·Kᵀ (T5 uses raw dot products).
//   • Position information comes from a learned per-head BIAS added to
//     scores before softmax, NOT from RoPE or absolute positional
//     embeddings. The bias depends on a bucketed relative position
//     (signed bucket index for bidirectional encoder).
//   • The bias is computed ONCE per forward (in block 0, which owns
//     `relative_attention_bias`) and threaded through subsequent blocks.
//
// References:
//   transformers/src/transformers/models/t5/modeling_t5.py
//     class T5Attention.{_relative_position_bucket, compute_bias, forward}
public final class T5SelfAttention: Module {
    public let numHeads: Int
    public let dKV: Int
    public let relativeAttentionNumBuckets: Int
    public let relativeAttentionMaxDistance: Int

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
        self.numHeads = config.numHeads
        self.dKV = config.dKV
        self.relativeAttentionNumBuckets = config.relativeAttentionNumBuckets
        self.relativeAttentionMaxDistance = config.relativeAttentionMaxDistance
        let inner = config.numHeads * config.dKV
        self.q = Linear(config.dModel, inner, bias: false)
        self.k = Linear(config.dModel, inner, bias: false)
        self.v = Linear(config.dModel, inner, bias: false)
        self.o = Linear(inner, config.dModel, bias: false)
        self._relativeAttentionBias.wrappedValue = hasRelativeBias
            ? Embedding(embeddingCount: config.relativeAttentionNumBuckets, dimensions: config.numHeads)
            : nil
    }

    /// Build the `[1, numHeads, qLen, kLen]` position-bias tensor used by
    /// every block in the encoder forward. Bidirectional bucketing per
    /// HuggingFace T5; mirrors `_relative_position_bucket(bidirectional=True)`.
    public func computePositionBias(queryLength: Int, keyLength: Int) -> MLXArray {
        guard let bias = relativeAttentionBias else {
            // Caller should only invoke this on block 0 (which owns the
            // bias embedding). Fallback returns a zero tensor of the right
            // shape so callers can short-circuit instead of branching.
            return MLXArray.zeros([1, numHeads, queryLength, keyLength])
        }

        // memory_position - context_position → [qLen, kLen] signed deltas.
        let qPos = MLXArray(Array(0..<queryLength).map { Int32($0) })
            .reshaped([queryLength, 1])
        let kPos = MLXArray(Array(0..<keyLength).map { Int32($0) })
            .reshaped([1, keyLength])
        let relPos = kPos - qPos

        let buckets = Self.relativePositionBucket(
            relativePosition: relPos,
            numBuckets: relativeAttentionNumBuckets,
            maxDistance: relativeAttentionMaxDistance
        )

        // Embedding lookup: [qLen, kLen, numHeads] → permute [numHeads, qLen, kLen]
        // → expand to [1, numHeads, qLen, kLen] for batch broadcast.
        let values = bias(buckets)                       // [qLen, kLen, numHeads]
        let permuted = values.transposed(2, 0, 1)        // [numHeads, qLen, kLen]
        return permuted.expandedDimensions(axis: 0)      // [1, numHeads, qLen, kLen]
    }

    /// Bidirectional bucket map from signed `memory - context` distances to
    /// indices in `[0, numBuckets)`. Half the buckets cover positive
    /// direction, half cover negative direction, with logarithmic spacing
    /// for distances ≥ numBuckets / 4.
    public static func relativePositionBucket(
        relativePosition: MLXArray,
        numBuckets: Int = 32,
        maxDistance: Int = 128
    ) -> MLXArray {
        let halfBuckets = numBuckets / 2
        // Direction half: 0 if relPos <= 0, halfBuckets if relPos > 0.
        let directionOffset = MLX.where(
            relativePosition .> 0,
            MLXArray(Int32(halfBuckets)),
            MLXArray(Int32(0))
        )
        // Magnitude.
        let absRel = MLX.abs(relativePosition)

        // Small distances: identity mapping into the lower-half bucket.
        let maxExact = halfBuckets / 2
        let isSmall = absRel .< Int32(maxExact)

        // Large distances: log-scale spread across the upper-half bucket
        // range, clamped to numBuckets - 1.
        let absFloat = absRel.asType(.float32)
        let logRatio = log(absFloat / Float(maxExact))
            / log(Float(maxDistance) / Float(maxExact))
        let large = (
            MLXArray(Float(maxExact))
            + logRatio * Float(halfBuckets - maxExact)
        ).asType(.int32)
        let largeClamped = MLX.minimum(large, MLXArray(Int32(halfBuckets - 1)))

        let magnitudeBucket = MLX.where(isSmall, absRel.asType(.int32), largeClamped)
        return directionOffset + magnitudeBucket
    }

    /// Self-attention forward. `attentionMask` is the additive mask form
    /// (`0` for valid, large-negative for pad) shaped broadcastable to
    /// `[B, 1, qLen, kLen]`. `positionBias` may be precomputed by block 0
    /// and reused by later blocks; pass `nil` on block 0 to compute fresh.
    public func callAsFunction(
        _ x: MLXArray,
        attentionMask: MLXArray? = nil,
        positionBias: MLXArray? = nil
    ) -> (output: MLXArray, positionBias: MLXArray) {
        let bsz = x.shape[0]
        let seqL = x.shape[1]

        // Q/K/V projections, reshape to [B, numHeads, seqL, dKV].
        let qOut = q(x).reshaped([bsz, seqL, numHeads, dKV]).transposed(0, 2, 1, 3)
        let kOut = k(x).reshaped([bsz, seqL, numHeads, dKV]).transposed(0, 2, 1, 3)
        let vOut = v(x).reshaped([bsz, seqL, numHeads, dKV]).transposed(0, 2, 1, 3)

        // Resolve / compute the per-head additive bias.
        var bias = positionBias ?? computePositionBias(queryLength: seqL, keyLength: seqL)
        if let mask = attentionMask {
            bias = bias + mask
        }

        // Scores: Q · Kᵀ (NO scale — T5 quirk).
        let scores = matmul(qOut, kOut.transposed(0, 1, 3, 2)) + bias
        let attn = softmax(scores, axis: -1)
        let context = matmul(attn, vOut)
            .transposed(0, 2, 1, 3)
            .reshaped([bsz, seqL, numHeads * dKV])
        return (o(context), bias)
    }
}

// MARK: - T5Block (encoder layer = self-attn + FFN, each pre-normed)
public final class T5Block: Module {
    public var layer0_norm: RMSNorm
    public var layer0_attn: T5SelfAttention
    public var layer1_norm: RMSNorm
    public var layer1_mlp: T5DenseActDense

    public init(config: T5Config, hasRelativeBias: Bool) {
        self.layer0_norm = RMSNorm(dimensions: config.dModel, eps: config.layerNormEpsilon)
        self.layer0_attn = T5SelfAttention(config: config, hasRelativeBias: hasRelativeBias)
        self.layer1_norm = RMSNorm(dimensions: config.dModel, eps: config.layerNormEpsilon)
        self.layer1_mlp = T5DenseActDense(config: config)
    }

    /// Pre-norm residual block: norm → attn → residual → norm → MLP →
    /// residual. Position bias propagates from block 0; subsequent blocks
    /// receive the same tensor and re-add it inside their attention.
    public func callAsFunction(
        _ x: MLXArray,
        attentionMask: MLXArray?,
        positionBias: MLXArray?
    ) -> (output: MLXArray, positionBias: MLXArray) {
        var h = x
        let (a, returnedBias) = layer0_attn(
            layer0_norm(h),
            attentionMask: attentionMask,
            positionBias: positionBias
        )
        h = h + a
        let f = layer1_mlp(layer1_norm(h))
        h = h + f
        return (h, returnedBias)
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
    ///   - attentionMask: `[B, L]` 0/1 mask. Pad positions (mask=0) get a
    ///     large-negative bias added before softmax so they're effectively
    ///     ignored.
    /// - Returns: `[B, L, dModel]` encoder hidden states.
    public func callAsFunction(inputIds: MLXArray, attentionMask: MLXArray? = nil) -> MLXArray {
        var h = shared(inputIds)

        // Build the additive attention mask once. Shape: [B, 1, 1, L].
        let additiveMask: MLXArray? = attentionMask.map { mask in
            // mask is 1 for valid, 0 for pad. Convert to additive form:
            //   valid → 0.0, pad → -1e9
            let neg = (1.0 - mask.asType(.float32)) * Float(-1e9)
            return neg.expandedDimensions(axes: [1, 2])
        }

        // Block 0 owns the relative-bias embedding; its returned position
        // bias (already augmented with the additive mask) is reused by
        // blocks 1..N to avoid recomputing.
        var positionBias: MLXArray? = nil
        for block in blocks {
            let (newH, newBias) = block(
                h,
                attentionMask: additiveMask,
                positionBias: positionBias
            )
            h = newH
            positionBias = newBias
        }
        h = finalLayerNorm(h)
        return h
    }
}
