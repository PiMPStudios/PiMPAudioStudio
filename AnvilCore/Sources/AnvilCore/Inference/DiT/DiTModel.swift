import Foundation
import MLX
import MLXNN

public class DiTDecoder: Module {
    let hiddenSize: Int
    let patchSize: Int
    let rotaryEmb: RotaryEmbedding
    let projIn: Conv1d
    let timeEmbed: TimestepEmbedding
    let timeEmbedR: TimestepEmbedding
    let conditionEmbedder: Linear
    let layers: [DiTLayer]
    let normOut: RMSNorm
    let projOut: ConvTransposed1d
    let scaleShiftTable: MLXArray
    let layerTypes: [String]
    let slidingWindow: Int

    private var slidingMasks: [Int: MLXArray] = [:]

    public init(
        hiddenSize: Int = 2048,
        intermediateSize: Int = 6144,
        numHiddenLayers: Int = 24,
        numAttentionHeads: Int = 16,
        numKeyValueHeads: Int = 8,
        headDim: Int = 128,
        rmsNormEps: Float = 1e-6,
        attentionBias: Bool = false,
        inChannels: Int = 192,
        audioAcousticHiddenDim: Int = 64,
        patchSize: Int = 2,
        slidingWindow: Int = 128,
        layerTypes: [String]? = nil,
        ropeTheta: Float = 1_000_000.0,
        maxPositionEmbeddings: Int = 32768
    ) {
        self.hiddenSize = hiddenSize
        self.patchSize = patchSize
        self.slidingWindow = slidingWindow

        let resolvedLayerTypes = layerTypes ?? (0..<numHiddenLayers).map { i in
            (i + 1) % 2 != 0 ? "sliding_attention" : "full_attention"
        }
        self.layerTypes = resolvedLayerTypes

        self.rotaryEmb = RotaryEmbedding(
            headDim: headDim, maxLen: maxPositionEmbeddings, base: ropeTheta
        )
        self.projIn = Conv1d(
            inputChannels: inChannels, outputChannels: hiddenSize,
            kernelSize: patchSize, stride: patchSize, padding: 0
        )
        self.timeEmbed = TimestepEmbedding(inChannels: 256, timeEmbedDim: hiddenSize)
        self.timeEmbedR = TimestepEmbedding(inChannels: 256, timeEmbedDim: hiddenSize)
        self.conditionEmbedder = Linear(hiddenSize, hiddenSize, bias: true)

        self.layers = (0..<numHiddenLayers).map { i in
            DiTLayer(
                hiddenSize: hiddenSize,
                intermediateSize: intermediateSize,
                numAttentionHeads: numAttentionHeads,
                numKeyValueHeads: numKeyValueHeads,
                headDim: headDim,
                rmsNormEps: rmsNormEps,
                attentionBias: attentionBias,
                layerIdx: i,
                layerType: resolvedLayerTypes[i],
                slidingWindow: slidingWindow
            )
        }

        self.normOut = RMSNorm(dimensions: hiddenSize, eps: rmsNormEps)
        self.projOut = ConvTransposed1d(
            inputChannels: hiddenSize, outputChannels: audioAcousticHiddenDim,
            kernelSize: patchSize, stride: patchSize, padding: 0
        )
        self.scaleShiftTable = MLXArray.zeros([1, 2, hiddenSize])
    }

    private func getSlidingMask(seqLen: Int) -> MLXArray {
        if let cached = slidingMasks[seqLen] { return cached }
        let mask = createSlidingWindowMask(seqLen: seqLen, windowSize: slidingWindow)
        slidingMasks[seqLen] = mask
        return mask
    }

    public func callAsFunction(
        hiddenStates: MLXArray,
        timestep: MLXArray,
        timestepR: MLXArray,
        encoderHiddenStates: MLXArray,
        contextLatents: MLXArray,
        cache: CrossAttentionCache?,
        useCache: Bool
    ) -> (MLXArray, CrossAttentionCache?) {
        // Timestep embeddings
        let (tembT, projT) = timeEmbed(timestep)
        let (tembR, projR) = timeEmbedR(timestep - timestepR)
        let temb = tembT + tembR
        let timestepProj = projT + projR

        // Concatenate context with hidden states along channel dimension
        var h = concatenated([contextLatents, hiddenStates], axis: -1)
        let originalSeqLen = h.shape[1]

        // Pad to multiple of patchSize
        var padLength = 0
        if h.shape[1] % patchSize != 0 {
            padLength = patchSize - (h.shape[1] % patchSize)
            let padding = MLXArray.zeros([h.shape[0], padLength, h.shape[2]])
            h = concatenated([h, padding], axis: 1)
        }

        // Patch embedding via strided convolution
        h = projIn(h)

        // Project encoder states
        let encStates = conditionEmbedder(encoderHiddenStates)

        let seqLen = h.shape[1]

        // Position embeddings
        let (cos, sin) = rotaryEmb.forward(seqLen: seqLen)

        // Sliding mask
        let hasSlidingLayer = layerTypes.contains("sliding_attention")
        let slidingMask: MLXArray? = hasSlidingLayer ? getSlidingMask(seqLen: seqLen) : nil

        // Process through transformer layers
        for layer in layers {
            let selfAttnMask = layer.layerType == "sliding_attention" ? slidingMask : nil
            h = layer(
                hiddenStates: h,
                positionCosSin: (cos, sin),
                temb: timestepProj,
                selfAttnMask: selfAttnMask,
                encoderHiddenStates: encStates,
                encoderAttentionMask: nil,
                cache: cache,
                useCache: useCache
            )
        }

        // Output adaptive layer norm
        let expandedTemb = temb.expandedDimensions(axis: 1)
        let outMod = scaleShiftTable + expandedTemb
        let outParts = outMod.split(parts: 2, axis: 1)
        let (shift, scale) = (outParts[0], outParts[1])
        h = normOut(h) * (1.0 + scale) + shift

        // De-patchify via transposed convolution
        h = projOut(h)

        // Crop back to original sequence length
        h = h[0..., 0..<originalSeqLen, 0...]

        return (h, cache)
    }
}
