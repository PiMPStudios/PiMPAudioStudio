// AnvilCore/Sources/AnvilCore/Inference/StableAudio/CLAPEncoder.swift
import Foundation
import MLX
import MLXNN
import MLXFast

public enum CLAPError: Error, Sendable {
    case notLoaded
    case invalidWeights(String)
}

public actor CLAPEncoder {
    private let embeddingDim: Int
    private let numLayers: Int
    private let numHeads: Int
    private let vocabSize: Int
    private var model: CLAPTextModel?
    private var loaded = false

    public init(embeddingDim: Int, numLayers: Int, numHeads: Int, vocabSize: Int) {
        self.embeddingDim = embeddingDim
        self.numLayers = numLayers
        self.numHeads = numHeads
        self.vocabSize = vocabSize
    }

    public var isLoaded: Bool { loaded }

    public func loadWeights(_ weights: [(String, MLXArray)]) {
        let m = CLAPTextModel(
            embeddingDim: embeddingDim,
            numLayers: numLayers,
            numHeads: numHeads,
            vocabSize: vocabSize
        )
        m.update(parameters: ModuleParameters.unflattened(weights))
        eval(m.parameters())
        self.model = m
        self.loaded = true
    }

    public func encode(text: String, tokenizer: CLAPTokenizer) throws -> MLXArray {
        guard let model, loaded else { throw CLAPError.notLoaded }

        let tokens = tokenizer.encode(text: text)
        let inputIds = MLXArray(tokens).reshaped([1, tokens.count])

        let embedding = model(inputIds)
        eval(embedding)
        return embedding
    }
}

class CLAPTextModel: Module {
    let tokenEmbedding: Embedding
    let positionEmbedding: Embedding
    let layers: [CLAPTransformerLayer]
    let finalLn: LayerNorm
    let projection: Linear

    init(embeddingDim: Int, numLayers: Int, numHeads: Int, vocabSize: Int) {
        self.tokenEmbedding = Embedding(embeddingCount: vocabSize, dimensions: embeddingDim)
        self.positionEmbedding = Embedding(embeddingCount: 77, dimensions: embeddingDim)
        self.layers = (0..<numLayers).map { _ in
            CLAPTransformerLayer(embeddingDim: embeddingDim, numHeads: numHeads)
        }
        self.finalLn = LayerNorm(dimensions: embeddingDim)
        self.projection = Linear(embeddingDim, embeddingDim, bias: false)
    }

    func callAsFunction(_ inputIds: MLXArray) -> MLXArray {
        let seqLen = inputIds.shape[1]
        let positions = MLXArray(Array(0..<seqLen)).reshaped([1, seqLen])

        var hidden = tokenEmbedding(inputIds) + positionEmbedding(positions)

        for layer in layers {
            hidden = layer(hidden)
        }

        hidden = finalLn(hidden)

        // Pool: take the EOS token position (last non-pad)
        let pooled = hidden[0..., seqLen - 1]
        let projected = projection(pooled)
        return projected.reshaped([1, -1])
    }
}

class CLAPTransformerLayer: Module {
    let attention: CLAPAttention
    let ln1: LayerNorm
    let ln2: LayerNorm
    let mlp: CLAPMLP

    init(embeddingDim: Int, numHeads: Int) {
        self.attention = CLAPAttention(embeddingDim: embeddingDim, numHeads: numHeads)
        self.ln1 = LayerNorm(dimensions: embeddingDim)
        self.ln2 = LayerNorm(dimensions: embeddingDim)
        self.mlp = CLAPMLP(embeddingDim: embeddingDim)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x + attention(ln1(x))
        h = h + mlp(ln2(h))
        return h
    }
}

class CLAPAttention: Module {
    let query: Linear
    let key: Linear
    let value: Linear
    let out: Linear
    let numHeads: Int
    let headDim: Int

    init(embeddingDim: Int, numHeads: Int) {
        self.numHeads = numHeads
        self.headDim = embeddingDim / numHeads
        self.query = Linear(embeddingDim, embeddingDim, bias: false)
        self.key = Linear(embeddingDim, embeddingDim, bias: false)
        self.value = Linear(embeddingDim, embeddingDim, bias: false)
        self.out = Linear(embeddingDim, embeddingDim, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let B = x.shape[0]
        let L = x.shape[1]

        let q = query(x).reshaped([B, L, numHeads, headDim]).transposed(0, 2, 1, 3)
        let k = key(x).reshaped([B, L, numHeads, headDim]).transposed(0, 2, 1, 3)
        let v = value(x).reshaped([B, L, numHeads, headDim]).transposed(0, 2, 1, 3)

        let scale = Float(1.0 / sqrt(Float(headDim)))
        let attn = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: nil)

        let merged = attn.transposed(0, 2, 1, 3).reshaped([B, L, -1])
        return out(merged)
    }
}

class CLAPMLP: Module {
    let fc1: Linear
    let fc2: Linear

    init(embeddingDim: Int) {
        self.fc1 = Linear(embeddingDim, embeddingDim * 4)
        self.fc2 = Linear(embeddingDim * 4, embeddingDim)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = fc1(x)
        h = gelu(h)
        return fc2(h)
    }
}

private func gelu(_ x: MLXArray) -> MLXArray {
    x * (1.0 + MLX.tanh(0.7978845608 * (x + 0.044715 * x * x * x))) * 0.5
}
