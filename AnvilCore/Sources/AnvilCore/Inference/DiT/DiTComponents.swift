import Foundation
import MLX
import MLXNN
import MLXFast

// MARK: - Rotary Position Embedding

public class RotaryEmbedding: Module {
    let headDim: Int
    let maxLen: Int
    let cosTable: MLXArray
    let sinTable: MLXArray

    public init(headDim: Int, maxLen: Int = 32768, base: Float = 1_000_000.0) {
        self.headDim = headDim
        self.maxLen = maxLen

        let invFreq = 1.0 / pow(
            MLXArray(base),
            MLXArray(stride(from: 0, to: headDim, by: 2).map { Float($0) / Float(headDim) })
        )
        let positions = MLXArray(Array(0..<maxLen).map { Float($0) })
        let freqs = positions.expandedDimensions(axis: 1) * invFreq.expandedDimensions(axis: 0)
        let freqsFull = concatenated([freqs, freqs], axis: 1)
        self.cosTable = cos(freqsFull)
        self.sinTable = sin(freqsFull)
    }

    public func forward(seqLen: Int) -> (cos: MLXArray, sin: MLXArray) {
        let c = cosTable[0..<seqLen].reshaped([1, 1, seqLen, headDim])
        let s = sinTable[0..<seqLen].reshaped([1, 1, seqLen, headDim])
        return (c, s)
    }
}

// MARK: - Rotary helpers

func rotateHalf(_ x: MLXArray) -> MLXArray {
    let half = x.shape.last! / 2
    let x1 = x[.ellipsis, 0..<half]
    let x2 = x[.ellipsis, half...]
    return concatenated([-x2, x1], axis: -1)
}

func applyRotaryPosEmb(
    q: MLXArray, k: MLXArray, cos: MLXArray, sin: MLXArray
) -> (MLXArray, MLXArray) {
    let qEmbed = (q * cos) + (rotateHalf(q) * sin)
    let kEmbed = (k * cos) + (rotateHalf(k) * sin)
    return (qEmbed, kEmbed)
}

// MARK: - Sliding Window Mask

public func createSlidingWindowMask(seqLen: Int, windowSize: Int, dtype: DType = .float32)
    -> MLXArray
{
    let indices = MLXArray(Array(0..<seqLen).map { Float($0) })
    let diff = abs(
        indices.expandedDimensions(axis: 1) - indices.expandedDimensions(axis: 0))
    let zeros = MLXArray.zeros([seqLen, seqLen])
    let negInf = MLXArray.full([seqLen, seqLen], values: MLXArray(Float(-1e9)))
    let mask = which(diff .< Float(windowSize + 1), zeros, negInf)
    return mask.reshaped([1, 1, seqLen, seqLen])
}

// MARK: - SwiGLU MLP

public class SwiGLUMLP: Module, UnaryLayer {
    let gateProj: Linear
    let upProj: Linear
    let downProj: Linear

    public init(hiddenSize: Int, intermediateSize: Int) {
        self.gateProj = Linear(hiddenSize, intermediateSize, bias: false)
        self.upProj = Linear(hiddenSize, intermediateSize, bias: false)
        self.downProj = Linear(intermediateSize, hiddenSize, bias: false)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

// MARK: - Attention

public class DiTAttention: Module {
    let hiddenSize: Int
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let nRep: Int
    let scale: Float
    let layerIdx: Int
    let isCrossAttention: Bool
    let slidingWindow: Int?

    let qProj: Linear
    let kProj: Linear
    let vProj: Linear
    let oProj: Linear
    let qNorm: RMSNorm
    let kNorm: RMSNorm

    public init(
        hiddenSize: Int,
        numAttentionHeads: Int,
        numKeyValueHeads: Int,
        headDim: Int,
        rmsNormEps: Float,
        attentionBias: Bool,
        layerIdx: Int,
        isCrossAttention: Bool = false,
        slidingWindow: Int? = nil
    ) {
        self.hiddenSize = hiddenSize
        self.numHeads = numAttentionHeads
        self.numKVHeads = numKeyValueHeads
        self.headDim = headDim
        self.nRep = numAttentionHeads / numKeyValueHeads
        self.scale = pow(Float(headDim), -0.5)
        self.layerIdx = layerIdx
        self.isCrossAttention = isCrossAttention
        self.slidingWindow = slidingWindow

        self.qProj = Linear(hiddenSize, numAttentionHeads * headDim, bias: attentionBias)
        self.kProj = Linear(hiddenSize, numKeyValueHeads * headDim, bias: attentionBias)
        self.vProj = Linear(hiddenSize, numKeyValueHeads * headDim, bias: attentionBias)
        self.oProj = Linear(numAttentionHeads * headDim, hiddenSize, bias: attentionBias)
        self.qNorm = RMSNorm(dimensions: headDim, eps: rmsNormEps)
        self.kNorm = RMSNorm(dimensions: headDim, eps: rmsNormEps)
    }

    private func repeatKV(_ x: MLXArray, nRep: Int) -> MLXArray {
        if nRep == 1 { return x }
        let (b, nKV, seqL, d) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        let expanded = x.expandedDimensions(axis: 2)
        let broadcasted = broadcast(expanded, to: [b, nKV, nRep, seqL, d])
        return broadcasted.reshaped([b, nKV * nRep, seqL, d])
    }

    public func callAsFunction(
        hiddenStates: MLXArray,
        positionCosSin: (cos: MLXArray, sin: MLXArray)?,
        attentionMask: MLXArray?,
        encoderHiddenStates: MLXArray?,
        cache: CrossAttentionCache?,
        useCache: Bool
    ) -> MLXArray {
        let b = hiddenStates.shape[0]
        let seqL = hiddenStates.shape[1]

        var q = qProj(hiddenStates)
        q = qNorm(q.reshaped([b, seqL, numHeads, headDim]))
        q = q.transposed(0, 2, 1, 3)

        var k: MLXArray
        var v: MLXArray

        if isCrossAttention, let enc = encoderHiddenStates {
            if let cache = cache, cache.isUpdated(layerIdx: layerIdx) {
                let cached = cache.get(layerIdx: layerIdx)!
                k = cached.key
                v = cached.value
            } else {
                let encL = enc.shape[1]
                k = kProj(enc)
                k = kNorm(k.reshaped([b, encL, numKVHeads, headDim]))
                k = k.transposed(0, 2, 1, 3)
                v = vProj(enc).reshaped([b, encL, numKVHeads, headDim]).transposed(0, 2, 1, 3)
                if let cache = cache, useCache {
                    cache.update(key: k, value: v, layerIdx: layerIdx)
                }
            }
        } else {
            k = kProj(hiddenStates)
            k = kNorm(k.reshaped([b, seqL, numKVHeads, headDim]))
            k = k.transposed(0, 2, 1, 3)
            v = vProj(hiddenStates).reshaped([b, seqL, numKVHeads, headDim])
                .transposed(0, 2, 1, 3)

            if let (cosVal, sinVal) = positionCosSin {
                (q, k) = applyRotaryPosEmb(q: q, k: k, cos: cosVal, sin: sinVal)
            }
        }

        k = repeatKV(k, nRep: nRep)
        v = repeatKV(v, nRep: nRep)

        let attnOut = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: attentionMask
        )

        let merged = attnOut.transposed(0, 2, 1, 3).reshaped([b, seqL, -1])
        return oProj(merged)
    }
}

// MARK: - Timestep Embedding

public class TimestepEmbedding: Module {
    let inChannels: Int
    let tsScale: Float
    let linear1: Linear
    let linear2: Linear
    let timeProj: Linear

    public init(inChannels: Int = 256, timeEmbedDim: Int = 2048, scale: Float = 1000.0) {
        self.inChannels = inChannels
        self.tsScale = scale
        self.linear1 = Linear(inChannels, timeEmbedDim, bias: true)
        self.linear2 = Linear(timeEmbedDim, timeEmbedDim, bias: true)
        self.timeProj = Linear(timeEmbedDim, timeEmbedDim * 6, bias: true)
    }

    private func sinusoidalEmbedding(_ t: MLXArray, dim: Int, maxPeriod: Int = 10000) -> MLXArray {
        let scaled = t * tsScale
        let half = dim / 2
        let freqs = exp(
            -log(Float(maxPeriod))
                * MLXArray(Array(0..<half).map { Float($0) / Float(half) })
        )
        let args = scaled.expandedDimensions(axis: 1) * freqs.expandedDimensions(axis: 0)
        var embedding = concatenated([cos(args), sin(args)], axis: -1)
        if dim % 2 != 0 {
            embedding = concatenated(
                [embedding, MLXArray.zeros([embedding.shape[0], 1])], axis: -1)
        }
        return embedding
    }

    public func callAsFunction(_ t: MLXArray) -> (temb: MLXArray, timestepProj: MLXArray) {
        let tFreq = sinusoidalEmbedding(t, dim: inChannels)
        var temb = linear1(tFreq)
        temb = silu(temb)
        temb = linear2(temb)
        let proj = timeProj(silu(temb))
        let timestepProj = proj.reshaped([proj.shape[0], 6, -1])
        return (temb, timestepProj)
    }
}

// MARK: - DiT Layer

public class DiTLayer: Module {
    let layerType: String
    let selfAttnNorm: RMSNorm
    let selfAttn: DiTAttention
    let crossAttnNorm: RMSNorm
    let crossAttn: DiTAttention
    let mlpNorm: RMSNorm
    let mlp: SwiGLUMLP
    let scaleShiftTable: MLXArray

    public init(
        hiddenSize: Int,
        intermediateSize: Int,
        numAttentionHeads: Int,
        numKeyValueHeads: Int,
        headDim: Int,
        rmsNormEps: Float,
        attentionBias: Bool,
        layerIdx: Int,
        layerType: String,
        slidingWindow: Int = 128
    ) {
        self.layerType = layerType
        let sw: Int? = layerType == "sliding_attention" ? slidingWindow : nil

        self.selfAttnNorm = RMSNorm(dimensions: hiddenSize, eps: rmsNormEps)
        self.selfAttn = DiTAttention(
            hiddenSize: hiddenSize, numAttentionHeads: numAttentionHeads,
            numKeyValueHeads: numKeyValueHeads, headDim: headDim,
            rmsNormEps: rmsNormEps, attentionBias: attentionBias,
            layerIdx: layerIdx, isCrossAttention: false, slidingWindow: sw
        )
        self.crossAttnNorm = RMSNorm(dimensions: hiddenSize, eps: rmsNormEps)
        self.crossAttn = DiTAttention(
            hiddenSize: hiddenSize, numAttentionHeads: numAttentionHeads,
            numKeyValueHeads: numKeyValueHeads, headDim: headDim,
            rmsNormEps: rmsNormEps, attentionBias: attentionBias,
            layerIdx: layerIdx, isCrossAttention: true, slidingWindow: nil
        )
        self.mlpNorm = RMSNorm(dimensions: hiddenSize, eps: rmsNormEps)
        self.mlp = SwiGLUMLP(hiddenSize: hiddenSize, intermediateSize: intermediateSize)
        self.scaleShiftTable = MLXArray.zeros([1, 6, hiddenSize])
    }

    public func callAsFunction(
        hiddenStates: MLXArray,
        positionCosSin: (cos: MLXArray, sin: MLXArray),
        temb: MLXArray,
        selfAttnMask: MLXArray?,
        encoderHiddenStates: MLXArray?,
        encoderAttentionMask: MLXArray?,
        cache: CrossAttentionCache?,
        useCache: Bool
    ) -> MLXArray {
        let modulation = scaleShiftTable + temb
        let parts = modulation.split(parts: 6, axis: 1)
        let (shiftMSA, scaleMSA, gateMSA) = (parts[0], parts[1], parts[2])
        let (cShiftMSA, cScaleMSA, cGateMSA) = (parts[3], parts[4], parts[5])

        var h = hiddenStates
        var normed = selfAttnNorm(h)
        normed = normed * (1.0 + scaleMSA) + shiftMSA
        let attnOut = selfAttn(
            hiddenStates: normed, positionCosSin: positionCosSin,
            attentionMask: selfAttnMask, encoderHiddenStates: nil,
            cache: nil, useCache: false
        )
        h = h + attnOut * gateMSA

        normed = crossAttnNorm(h)
        let crossOut = crossAttn(
            hiddenStates: normed, positionCosSin: nil,
            attentionMask: encoderAttentionMask,
            encoderHiddenStates: encoderHiddenStates,
            cache: cache, useCache: useCache
        )
        h = h + crossOut

        normed = mlpNorm(h)
        normed = normed * (1.0 + cScaleMSA) + cShiftMSA
        let ffOut = mlp(normed)
        h = h + ffOut * cGateMSA

        return h
    }
}
