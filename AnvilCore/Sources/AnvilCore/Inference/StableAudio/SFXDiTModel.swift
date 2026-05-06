// AnvilCore/Sources/AnvilCore/Inference/StableAudio/SFXDiTModel.swift
import Foundation
import MLX
import MLXNN
import MLXFast

public class SFXDiTModel: Module {
    let projIn: Linear
    let timeEmbed: SFXTimestepEmbedding
    let condProj: Linear
    let blocks: [SFXDiTBlock]
    let finalNorm: LayerNorm
    let projOut: Linear
    let config: SFXDiTConfig

    public init(config: SFXDiTConfig) {
        self.config = config
        self.projIn = Linear(config.latentDim, config.hiddenSize)
        self.timeEmbed = SFXTimestepEmbedding(hiddenSize: config.hiddenSize)
        self.condProj = Linear(config.conditioningDim, config.hiddenSize)
        self.blocks = (0..<config.numLayers).map { _ in
            SFXDiTBlock(hiddenSize: config.hiddenSize, numHeads: config.numHeads, condDim: config.hiddenSize)
        }
        self.finalNorm = LayerNorm(dimensions: config.hiddenSize)
        self.projOut = Linear(config.hiddenSize, config.latentDim)
    }

    public func forward(latents: MLXArray, timestep: MLXArray, conditioning: MLXArray) -> MLXArray {
        var hidden = projIn(latents)
        let tEmb = timeEmbed(timestep)
        let condEmb = condProj(conditioning)

        for block in blocks {
            hidden = block(hidden, timeEmbedding: tEmb, conditioning: condEmb)
        }

        hidden = finalNorm(hidden)
        return projOut(hidden)
    }
}

class SFXTimestepEmbedding: Module {
    let fc1: Linear
    let fc2: Linear
    let hiddenSize: Int

    init(hiddenSize: Int) {
        self.hiddenSize = hiddenSize
        self.fc1 = Linear(hiddenSize, hiddenSize)
        self.fc2 = Linear(hiddenSize, hiddenSize)
    }

    func callAsFunction(_ t: MLXArray) -> MLXArray {
        let halfDim = hiddenSize / 2
        let freqs = exp(
            MLXArray(Array(0..<halfDim).map { Float($0) }) * (-log(Float(10000.0)) / Float(halfDim))
        )
        let angles = t.reshaped([-1, 1]) * freqs.reshaped([1, -1])
        let emb = concatenated([sin(angles), cos(angles)], axis: 1)
        var h = fc1(emb)
        h = sfxSilu(h)
        return fc2(h)
    }
}

class SFXDiTBlock: Module {
    let norm1: LayerNorm
    let attn: SFXSelfAttention
    let norm2: LayerNorm
    let crossAttn: SFXCrossAttention
    let norm3: LayerNorm
    let mlp: SFXMLP
    let adaLN: Linear

    init(hiddenSize: Int, numHeads: Int, condDim: Int) {
        self.norm1 = LayerNorm(dimensions: hiddenSize)
        self.attn = SFXSelfAttention(hiddenSize: hiddenSize, numHeads: numHeads)
        self.norm2 = LayerNorm(dimensions: hiddenSize)
        self.crossAttn = SFXCrossAttention(hiddenSize: hiddenSize, numHeads: numHeads, condDim: condDim)
        self.norm3 = LayerNorm(dimensions: hiddenSize)
        self.mlp = SFXMLP(hiddenSize: hiddenSize)
        self.adaLN = Linear(hiddenSize, hiddenSize * 3)
    }

    func callAsFunction(_ x: MLXArray, timeEmbedding: MLXArray, conditioning: MLXArray) -> MLXArray {
        let adaln = adaLN(timeEmbedding)
        let chunks = adaln.reshaped([adaln.shape[0], 3, -1])
        let scale1 = chunks[0..., 0]
        let shift1 = chunks[0..., 1]
        let gate = sigmoid(chunks[0..., 2])

        var h = norm1(x) * (1.0 + scale1.expandedDimensions(axis: 1)) + shift1.expandedDimensions(axis: 1)
        h = x + attn(h)
        h = h + crossAttn(norm2(h), conditioning: conditioning)
        h = h + gate.expandedDimensions(axis: 1) * mlp(norm3(h))
        return h
    }
}

class SFXSelfAttention: Module {
    let qkv: Linear
    let out: Linear
    let numHeads: Int
    let headDim: Int

    init(hiddenSize: Int, numHeads: Int) {
        self.numHeads = numHeads
        self.headDim = hiddenSize / numHeads
        self.qkv = Linear(hiddenSize, hiddenSize * 3, bias: false)
        self.out = Linear(hiddenSize, hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let B = x.shape[0]
        let L = x.shape[1]
        let qkvOut = qkv(x).reshaped([B, L, 3, numHeads, headDim])
        let q = qkvOut[0..., 0..., 0].transposed(0, 2, 1, 3)
        let k = qkvOut[0..., 0..., 1].transposed(0, 2, 1, 3)
        let v = qkvOut[0..., 0..., 2].transposed(0, 2, 1, 3)

        let scale = Float(1.0 / sqrt(Float(headDim)))
        let attn = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: nil)
        let merged = attn.transposed(0, 2, 1, 3).reshaped([B, L, -1])
        return out(merged)
    }
}

class SFXCrossAttention: Module {
    let q: Linear
    let kv: Linear
    let out: Linear
    let numHeads: Int
    let headDim: Int

    init(hiddenSize: Int, numHeads: Int, condDim: Int) {
        self.numHeads = numHeads
        self.headDim = hiddenSize / numHeads
        self.q = Linear(hiddenSize, hiddenSize, bias: false)
        self.kv = Linear(condDim, hiddenSize * 2, bias: false)
        self.out = Linear(hiddenSize, hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray, conditioning: MLXArray) -> MLXArray {
        let B = x.shape[0]
        let L = x.shape[1]

        let cond = conditioning.expandedDimensions(axis: 1)
        let condLen = cond.shape[1]

        let queries = q(x).reshaped([B, L, numHeads, headDim]).transposed(0, 2, 1, 3)
        let kvOut = kv(cond).reshaped([B, condLen, 2, numHeads, headDim])
        let keys = kvOut[0..., 0..., 0].transposed(0, 2, 1, 3)
        let values = kvOut[0..., 0..., 1].transposed(0, 2, 1, 3)

        let scale = Float(1.0 / sqrt(Float(headDim)))
        let attn = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale, mask: nil
        )
        let merged = attn.transposed(0, 2, 1, 3).reshaped([B, L, -1])
        return out(merged)
    }
}

class SFXMLP: Module {
    let fc1: Linear
    let fc2: Linear

    init(hiddenSize: Int) {
        self.fc1 = Linear(hiddenSize, hiddenSize * 4)
        self.fc2 = Linear(hiddenSize * 4, hiddenSize)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = fc1(x)
        h = sfxSilu(h)
        return fc2(h)
    }
}

private func sfxSilu(_ x: MLXArray) -> MLXArray {
    x * sigmoid(x)
}
