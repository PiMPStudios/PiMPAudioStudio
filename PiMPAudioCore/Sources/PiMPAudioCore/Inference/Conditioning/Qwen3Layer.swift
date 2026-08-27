import Foundation
import MLX
import MLXFast
import MLXNN

public class Qwen3Attention: Module {
    let hiddenSize: Int
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let nRep: Int
    let scale: Float

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
        rmsNormEps: Float
    ) {
        self.hiddenSize = hiddenSize
        numHeads = numAttentionHeads
        numKVHeads = numKeyValueHeads
        self.headDim = headDim
        nRep = numAttentionHeads / numKeyValueHeads
        scale = pow(Float(headDim), -0.5)

        qProj = Linear(hiddenSize, numAttentionHeads * headDim, bias: false)
        kProj = Linear(hiddenSize, numKeyValueHeads * headDim, bias: false)
        vProj = Linear(hiddenSize, numKeyValueHeads * headDim, bias: false)
        oProj = Linear(numAttentionHeads * headDim, hiddenSize, bias: false)
        qNorm = RMSNorm(dimensions: headDim, eps: rmsNormEps)
        kNorm = RMSNorm(dimensions: headDim, eps: rmsNormEps)
    }

    private func repeatKV(_ x: MLXArray, nRep: Int) -> MLXArray {
        if nRep == 1 {
            return x
        }
        let (b, nKV, seqL, d) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        let expanded = x.expandedDimensions(axis: 2)
        let broadcasted = broadcast(expanded, to: [b, nKV, nRep, seqL, d])
        return broadcasted.reshaped([b, nKV * nRep, seqL, d])
    }

    public func callAsFunction(
        hiddenStates: MLXArray,
        positionCosSin: (cos: MLXArray, sin: MLXArray),
        attentionMask: MLXArray?
    ) -> MLXArray {
        let b = hiddenStates.shape[0]
        let seqL = hiddenStates.shape[1]

        var q = qProj(hiddenStates)
        q = qNorm(q.reshaped([b, seqL, numHeads, headDim]))
        q = q.transposed(0, 2, 1, 3)

        var k = kProj(hiddenStates)
        k = kNorm(k.reshaped([b, seqL, numKVHeads, headDim]))
        k = k.transposed(0, 2, 1, 3)

        let v = vProj(hiddenStates).reshaped([b, seqL, numKVHeads, headDim])
            .transposed(0, 2, 1, 3)

        (q, k) = applyRotaryPosEmb(q: q, k: k, cos: positionCosSin.cos, sin: positionCosSin.sin)

        let kExpanded = repeatKV(k, nRep: nRep)
        let vExpanded = repeatKV(v, nRep: nRep)

        let attnOut = MLXFast.scaledDotProductAttention(
            queries: q, keys: kExpanded, values: vExpanded, scale: scale, mask: attentionMask
        )

        let merged = attnOut.transposed(0, 2, 1, 3).reshaped([b, seqL, -1])
        return oProj(merged)
    }
}

public class Qwen3Layer: Module {
    let inputNorm: RMSNorm
    let attention: Qwen3Attention
    let postNorm: RMSNorm
    let mlp: SwiGLUMLP

    public init(
        hiddenSize: Int,
        intermediateSize: Int,
        numAttentionHeads: Int,
        numKeyValueHeads: Int,
        headDim: Int,
        rmsNormEps: Float
    ) {
        inputNorm = RMSNorm(dimensions: hiddenSize, eps: rmsNormEps)
        attention = Qwen3Attention(
            hiddenSize: hiddenSize,
            numAttentionHeads: numAttentionHeads,
            numKeyValueHeads: numKeyValueHeads,
            headDim: headDim,
            rmsNormEps: rmsNormEps
        )
        postNorm = RMSNorm(dimensions: hiddenSize, eps: rmsNormEps)
        mlp = SwiGLUMLP(hiddenSize: hiddenSize, intermediateSize: intermediateSize)
    }

    public func callAsFunction(
        hiddenStates: MLXArray,
        positionCosSin: (cos: MLXArray, sin: MLXArray),
        attentionMask: MLXArray?
    ) -> MLXArray {
        var h = hiddenStates
        let normed = inputNorm(h)
        let attnOut = attention(
            hiddenStates: normed, positionCosSin: positionCosSin, attentionMask: attentionMask
        )
        h = h + attnOut

        let postNormed = postNorm(h)
        let ffOut = mlp(postNormed)
        h = h + ffOut

        return h
    }
}
