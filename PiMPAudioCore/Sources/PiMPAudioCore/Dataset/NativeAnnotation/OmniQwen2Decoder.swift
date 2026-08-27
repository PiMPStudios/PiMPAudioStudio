import Foundation
import MLX
import MLXFast
import MLXNN

/// Minimal Qwen2 decoder used by the Qwen2.5-Omni Thinker.
///
/// The structure follows the MIT-licensed MLX Swift LM Qwen2 implementation at
/// tag 2.31.3, but keeps the embedding-forward path public because Omni must
/// replace the audio placeholder span with projected audio embeddings.
public final class OmniKVCache {
    private var keys: MLXArray?
    private var values: MLXArray?

    public private(set) var offset = 0
    public let allocationStep: Int

    public init(allocationStep: Int = 256) {
        self.allocationStep = max(1, allocationStep)
    }

    public func update(
        keys newKeys: MLXArray,
        values newValues: MLXArray
    ) -> (MLXArray, MLXArray) {
        let previous = offset
        let newTokenCount = newKeys.dim(2)
        let requiresGrowth =
            keys == nil || previous + newTokenCount > (keys?.dim(2) ?? 0)

        if requiresGrowth {
            let batch = newKeys.dim(0)
            let heads = newKeys.dim(1)
            let keyDimension = newKeys.dim(3)
            let valueDimension = newValues.dim(3)
            let growthSteps =
                (allocationStep + newTokenCount - 1) / allocationStep
            let growth = growthSteps * allocationStep
            let extraKeys = MLXArray.zeros(
                [batch, heads, growth, keyDimension],
                dtype: newKeys.dtype
            )
            let extraValues = MLXArray.zeros(
                [batch, heads, growth, valueDimension],
                dtype: newValues.dtype
            )

            if var currentKeys = keys, var currentValues = values {
                if previous % allocationStep != 0 {
                    currentKeys = currentKeys[.ellipsis, ..<previous, 0...]
                    currentValues = currentValues[.ellipsis, ..<previous, 0...]
                }
                keys = concatenated([currentKeys, extraKeys], axis: 2)
                values = concatenated([currentValues, extraValues], axis: 2)
            } else {
                keys = extraKeys
                values = extraValues
            }
        }

        offset += newTokenCount
        keys?[.ellipsis, previous ..< offset, 0...] = newKeys
        values?[.ellipsis, previous ..< offset, 0...] = newValues

        guard let keys, let values else {
            return (newKeys, newValues)
        }
        return (
            keys[.ellipsis, ..<offset, 0...],
            values[.ellipsis, ..<offset, 0...]
        )
    }

    public func reset() {
        keys = nil
        values = nil
        offset = 0
    }
}

public final class OmniQwen2Attention: Module {
    public let configuration: OmniTextConfiguration
    public let scale: Float

    @ModuleInfo(key: "q_proj") public var queryProjection: Linear
    @ModuleInfo(key: "k_proj") public var keyProjection: Linear
    @ModuleInfo(key: "v_proj") public var valueProjection: Linear
    @ModuleInfo(key: "o_proj") public var outputProjection: Linear

    public let rope: RoPE

    public init(configuration: OmniTextConfiguration) {
        self.configuration = configuration
        scale = pow(Float(configuration.headDimension), -0.5)
        _queryProjection.wrappedValue = Linear(
            configuration.hiddenSize,
            configuration.attentionHeads * configuration.headDimension,
            bias: true
        )
        _keyProjection.wrappedValue = Linear(
            configuration.hiddenSize,
            configuration.keyValueHeads * configuration.headDimension,
            bias: true
        )
        _valueProjection.wrappedValue = Linear(
            configuration.hiddenSize,
            configuration.keyValueHeads * configuration.headDimension,
            bias: true
        )
        _outputProjection.wrappedValue = Linear(
            configuration.attentionHeads * configuration.headDimension,
            configuration.hiddenSize,
            bias: false
        )
        rope = RoPE(
            dimensions: configuration.headDimension,
            traditional: configuration.ropeTraditional,
            base: configuration.ropeTheta
        )
    }

    public func callAsFunction(
        _ input: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: OmniKVCache?
    ) -> MLXArray {
        let batch = input.dim(0)
        let sequence = input.dim(1)
        var queries = queryProjection(input)
        var keys = keyProjection(input)
        var values = valueProjection(input)

        queries = queries
            .reshaped(
                batch,
                sequence,
                configuration.attentionHeads,
                configuration.headDimension
            )
            .transposed(0, 2, 1, 3)
        keys = keys
            .reshaped(
                batch,
                sequence,
                configuration.keyValueHeads,
                configuration.headDimension
            )
            .transposed(0, 2, 1, 3)
        values = values
            .reshaped(
                batch,
                sequence,
                configuration.keyValueHeads,
                configuration.headDimension
            )
            .transposed(0, 2, 1, 3)

        if let cache {
            queries = rope(queries, offset: cache.offset)
            keys = rope(keys, offset: cache.offset)
        } else {
            queries = rope(queries)
            keys = rope(keys)
        }

        let updated: (MLXArray, MLXArray) = if let cache {
            cache.update(keys: keys, values: values)
        } else {
            (keys, values)
        }
        let attended = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: updated.0,
            values: updated.1,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(
            batch,
            sequence,
            configuration.attentionHeads * configuration.headDimension
        )
        return outputProjection(attended)
    }
}

public final class OmniQwen2MLP: Module {
    @ModuleInfo(key: "gate_proj") public var gateProjection: Linear
    @ModuleInfo(key: "up_proj") public var upProjection: Linear
    @ModuleInfo(key: "down_proj") public var downProjection: Linear

    public init(configuration: OmniTextConfiguration) {
        _gateProjection.wrappedValue = Linear(
            configuration.hiddenSize,
            configuration.intermediateSize,
            bias: false
        )
        _upProjection.wrappedValue = Linear(
            configuration.hiddenSize,
            configuration.intermediateSize,
            bias: false
        )
        _downProjection.wrappedValue = Linear(
            configuration.intermediateSize,
            configuration.hiddenSize,
            bias: false
        )
    }

    public func callAsFunction(_ input: MLXArray) -> MLXArray {
        downProjection(silu(gateProjection(input)) * upProjection(input))
    }
}

public final class OmniQwen2TransformerBlock: Module {
    @ModuleInfo(key: "self_attn") public var attention: OmniQwen2Attention
    public let mlp: OmniQwen2MLP
    @ModuleInfo(key: "input_layernorm") public var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm")
    public var postAttentionLayerNorm: RMSNorm

    public init(configuration: OmniTextConfiguration) {
        _attention.wrappedValue = OmniQwen2Attention(
            configuration: configuration
        )
        mlp = OmniQwen2MLP(configuration: configuration)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize,
            eps: configuration.rmsNormEpsilon
        )
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize,
            eps: configuration.rmsNormEpsilon
        )
    }

    public func callAsFunction(
        _ input: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: OmniKVCache?
    ) -> MLXArray {
        let attended = attention(
            inputLayerNorm(input),
            mask: mask,
            cache: cache
        )
        let residual = input + attended
        return residual + mlp(postAttentionLayerNorm(residual))
    }
}

public final class OmniQwen2DecoderModel: Module {
    public let configuration: OmniTextConfiguration
    @ModuleInfo(key: "embed_tokens") public var embedTokens: Embedding
    public let layers: [OmniQwen2TransformerBlock]
    public let norm: RMSNorm

    public init(configuration: OmniTextConfiguration) {
        self.configuration = configuration
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: configuration.vocabularySize,
            dimensions: configuration.hiddenSize
        )
        layers = (0 ..< configuration.hiddenLayers).map { _ in
            OmniQwen2TransformerBlock(configuration: configuration)
        }
        norm = RMSNorm(
            dimensions: configuration.hiddenSize,
            eps: configuration.rmsNormEpsilon
        )
    }

    public func embeddings(for tokenIDs: MLXArray) -> MLXArray {
        embedTokens(tokenIDs)
    }

    public func callAsFunction(
        embeddings: MLXArray,
        cache: [OmniKVCache]? = nil,
        cancellationCheck: OmniCancellationCheck = { try Task.checkCancellation() }
    ) throws -> MLXArray {
        var hidden = embeddings
        let mask = Self.attentionMask(
            sequenceLength: hidden.dim(1),
            cacheOffset: cache?.first?.offset ?? 0
        )
        for (index, layer) in layers.enumerated() {
            try cancellationCheck()
            hidden = layer(hidden, mask: mask, cache: cache?[index])
        }
        return norm(hidden)
    }

    private static func attentionMask(
        sequenceLength: Int,
        cacheOffset: Int
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        if sequenceLength == 1 {
            return .none
        }
        if cacheOffset == 0 {
            return .causal
        }
        var right = MLXArray(
            Int32(0) ..< Int32(cacheOffset + sequenceLength)
        )
        var left = MLXArray(
            Int32(cacheOffset) ..< Int32(cacheOffset + sequenceLength)
        )
        left = left[0..., .newAxis]
        right = right[.newAxis]
        return .array(left .>= right)
    }
}

public final class OmniQwen2ForCausalLM: Module {
    public let configuration: OmniTextConfiguration
    public let model: OmniQwen2DecoderModel
    @ModuleInfo(key: "lm_head") public var languageModelHead: Linear?

    public init(configuration: OmniTextConfiguration) {
        self.configuration = configuration
        model = OmniQwen2DecoderModel(configuration: configuration)
        if configuration.tieWordEmbeddings {
            _languageModelHead.wrappedValue = nil
        } else {
            _languageModelHead.wrappedValue = Linear(
                configuration.hiddenSize,
                configuration.vocabularySize,
                bias: false
            )
        }
    }

    public func newCache(allocationStep: Int = 256) -> [OmniKVCache] {
        (0 ..< configuration.hiddenLayers).map { _ in
            OmniKVCache(allocationStep: allocationStep)
        }
    }

    public func logits(
        tokenIDs: MLXArray,
        cache: [OmniKVCache]? = nil,
        cancellationCheck: OmniCancellationCheck = { try Task.checkCancellation() }
    ) throws -> MLXArray {
        try logits(
            embeddings: model.embeddings(for: tokenIDs),
            cache: cache,
            cancellationCheck: cancellationCheck
        )
    }

    public func logits(
        embeddings: MLXArray,
        cache: [OmniKVCache]? = nil,
        cancellationCheck: OmniCancellationCheck = { try Task.checkCancellation() }
    ) throws -> MLXArray {
        var hidden = try model(
            embeddings: embeddings,
            cache: cache,
            cancellationCheck: cancellationCheck
        )
        if let languageModelHead {
            hidden = languageModelHead(hidden)
        } else {
            hidden = matmul(hidden, model.embedTokens.weight.T)
        }
        return hidden
    }
}
