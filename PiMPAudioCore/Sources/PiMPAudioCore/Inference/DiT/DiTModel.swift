import Foundation
import MLX
import MLXNN

public struct DiTConfig: Codable, Sendable, Equatable {
    public let hiddenSize: Int
    public let encoderHiddenSize: Int
    public let intermediateSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let headDim: Int
    public let rmsNormEps: Float
    public let attentionBias: Bool
    public let inChannels: Int
    public let audioAcousticHiddenDim: Int
    public let patchSize: Int
    public let slidingWindow: Int
    public let layerTypes: [String]
    public let ropeTheta: Float
    public let maxPositionEmbeddings: Int
    public let textHiddenDim: Int
    public let timbreHiddenDim: Int
    public let isTurbo: Bool

    private enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case encoderHiddenSize = "encoder_hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case attentionBias = "attention_bias"
        case inChannels = "in_channels"
        case audioAcousticHiddenDim = "audio_acoustic_hidden_dim"
        case patchSize = "patch_size"
        case slidingWindow = "sliding_window"
        case layerTypes = "layer_types"
        case ropeTheta = "rope_theta"
        case maxPositionEmbeddings = "max_position_embeddings"
        case textHiddenDim = "text_hidden_dim"
        case timbreHiddenDim = "timbre_hidden_dim"
        case isTurbo = "is_turbo"
    }

    public static let v15Base = DiTConfig(
        hiddenSize: 2048,
        encoderHiddenSize: 2048,
        intermediateSize: 6144,
        numHiddenLayers: 24,
        numAttentionHeads: 16,
        numKeyValueHeads: 8,
        headDim: 128,
        rmsNormEps: 1e-6,
        attentionBias: false,
        inChannels: 192,
        audioAcousticHiddenDim: 64,
        patchSize: 2,
        slidingWindow: 128,
        layerTypes: (0..<24).map { ($0 + 1) % 2 != 0 ? "sliding_attention" : "full_attention" },
        ropeTheta: 1_000_000,
        maxPositionEmbeddings: 32768,
        textHiddenDim: 1024,
        timbreHiddenDim: 64,
        isTurbo: false
    )

    public init(
        hiddenSize: Int,
        encoderHiddenSize: Int,
        intermediateSize: Int,
        numHiddenLayers: Int,
        numAttentionHeads: Int,
        numKeyValueHeads: Int,
        headDim: Int,
        rmsNormEps: Float,
        attentionBias: Bool,
        inChannels: Int,
        audioAcousticHiddenDim: Int,
        patchSize: Int,
        slidingWindow: Int,
        layerTypes: [String],
        ropeTheta: Float,
        maxPositionEmbeddings: Int,
        textHiddenDim: Int,
        timbreHiddenDim: Int,
        isTurbo: Bool
    ) {
        self.hiddenSize = hiddenSize
        self.encoderHiddenSize = encoderHiddenSize
        self.intermediateSize = intermediateSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.headDim = headDim
        self.rmsNormEps = rmsNormEps
        self.attentionBias = attentionBias
        self.inChannels = inChannels
        self.audioAcousticHiddenDim = audioAcousticHiddenDim
        self.patchSize = patchSize
        self.slidingWindow = slidingWindow
        self.layerTypes = layerTypes.count == numHiddenLayers
            ? layerTypes
            : (0..<numHiddenLayers).map { ($0 + 1) % 2 != 0 ? "sliding_attention" : "full_attention" }
        self.ropeTheta = ropeTheta
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.textHiddenDim = textHiddenDim
        self.timbreHiddenDim = timbreHiddenDim
        self.isTurbo = isTurbo
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let defaults = Self.v15Base
        let hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize)
            ?? defaults.hiddenSize
        let numHiddenLayers = try container.decodeIfPresent(Int.self, forKey: .numHiddenLayers)
            ?? defaults.numHiddenLayers
        let layerTypes = try container.decodeIfPresent([String].self, forKey: .layerTypes)
            ?? (0..<numHiddenLayers).map { ($0 + 1) % 2 != 0 ? "sliding_attention" : "full_attention" }

        self.init(
            hiddenSize: hiddenSize,
            encoderHiddenSize: try container.decodeIfPresent(Int.self, forKey: .encoderHiddenSize)
                ?? defaults.encoderHiddenSize,
            intermediateSize: try container.decodeIfPresent(Int.self, forKey: .intermediateSize)
                ?? defaults.intermediateSize,
            numHiddenLayers: numHiddenLayers,
            numAttentionHeads: try container.decodeIfPresent(Int.self, forKey: .numAttentionHeads)
                ?? defaults.numAttentionHeads,
            numKeyValueHeads: try container.decodeIfPresent(Int.self, forKey: .numKeyValueHeads)
                ?? defaults.numKeyValueHeads,
            headDim: try container.decodeIfPresent(Int.self, forKey: .headDim)
                ?? defaults.headDim,
            rmsNormEps: try container.decodeIfPresent(Float.self, forKey: .rmsNormEps)
                ?? defaults.rmsNormEps,
            attentionBias: try container.decodeIfPresent(Bool.self, forKey: .attentionBias)
                ?? defaults.attentionBias,
            inChannels: try container.decodeIfPresent(Int.self, forKey: .inChannels)
                ?? defaults.inChannels,
            audioAcousticHiddenDim: try container.decodeIfPresent(Int.self, forKey: .audioAcousticHiddenDim)
                ?? defaults.audioAcousticHiddenDim,
            patchSize: try container.decodeIfPresent(Int.self, forKey: .patchSize)
                ?? defaults.patchSize,
            slidingWindow: try container.decodeIfPresent(Int.self, forKey: .slidingWindow)
                ?? defaults.slidingWindow,
            layerTypes: layerTypes,
            ropeTheta: try container.decodeIfPresent(Float.self, forKey: .ropeTheta)
                ?? defaults.ropeTheta,
            maxPositionEmbeddings: try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings)
                ?? defaults.maxPositionEmbeddings,
            textHiddenDim: try container.decodeIfPresent(Int.self, forKey: .textHiddenDim)
                ?? defaults.textHiddenDim,
            timbreHiddenDim: try container.decodeIfPresent(Int.self, forKey: .timbreHiddenDim)
                ?? defaults.timbreHiddenDim,
            isTurbo: try container.decodeIfPresent(Bool.self, forKey: .isTurbo)
                ?? defaults.isTurbo
        )
    }

    public static func load(from url: URL) throws -> DiTConfig {
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(DiTConfig.self, from: data)
    }

    public var isSupportedBySwiftRuntime: Bool {
        encoderHiddenSize == 2048
            && textHiddenDim == 1024
            && timbreHiddenDim == 64
            && inChannels == 192
            && audioAcousticHiddenDim == 64
            && patchSize == 2
    }

    public var runtimeSupportNote: String? {
        guard !isSupportedBySwiftRuntime else { return nil }
        return "This ACE-Step DiT can be downloaded, but this build cannot run it yet. It declares encoder_hidden_size=\(encoderHiddenSize), text_hidden_dim=\(textHiddenDim), timbre_hidden_dim=\(timbreHiddenDim), in_channels=\(inChannels), audio_acoustic_hidden_dim=\(audioAcousticHiddenDim), patch_size=\(patchSize); the current Swift conditioning runtime requires 2048/1024/64/192/64/2."
    }

    public var variantFamily: String {
        hiddenSize >= 2560 || numHiddenLayers >= 32 ? "ace-step-1.5-xl" : "ace-step-1.5"
    }
}

public struct DiTTraceOutput {
    public let output: MLXArray
    public let tensors: [String: MLXArray]

    public init(output: MLXArray, tensors: [String: MLXArray]) {
        self.output = output
        self.tensors = tensors
    }
}

public class DiTDecoder: Module {
    let hiddenSize: Int
    let encoderHiddenSize: Int
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
    let nullConditionEmb: MLXArray   // [1, 1, hidden_size] learned "no conditioning" token for CFG
    let layerTypes: [String]
    let slidingWindow: Int

    private var slidingMasks: [Int: MLXArray] = [:]

    public init(
        hiddenSize: Int = 2048,
        encoderHiddenSize: Int = 2048,
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
        self.encoderHiddenSize = encoderHiddenSize
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
        self.conditionEmbedder = Linear(encoderHiddenSize, hiddenSize, bias: true)

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
        self.nullConditionEmb = MLXArray.zeros([1, 1, encoderHiddenSize])
    }

    public convenience init(config: DiTConfig) {
        self.init(
            hiddenSize: config.hiddenSize,
            encoderHiddenSize: config.encoderHiddenSize,
            intermediateSize: config.intermediateSize,
            numHiddenLayers: config.numHiddenLayers,
            numAttentionHeads: config.numAttentionHeads,
            numKeyValueHeads: config.numKeyValueHeads,
            headDim: config.headDim,
            rmsNormEps: config.rmsNormEps,
            attentionBias: config.attentionBias,
            inChannels: config.inChannels,
            audioAcousticHiddenDim: config.audioAcousticHiddenDim,
            patchSize: config.patchSize,
            slidingWindow: config.slidingWindow,
            layerTypes: config.layerTypes,
            ropeTheta: config.ropeTheta,
            maxPositionEmbeddings: config.maxPositionEmbeddings
        )
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
        encoderAttentionMask: MLXArray?,
        cache: CrossAttentionCache?,
        useCache: Bool
    ) -> (MLXArray, CrossAttentionCache?) {
        let result = run(
            hiddenStates: hiddenStates,
            timestep: timestep,
            timestepR: timestepR,
            encoderHiddenStates: encoderHiddenStates,
            contextLatents: contextLatents,
            encoderAttentionMask: encoderAttentionMask,
            cache: cache,
            useCache: useCache,
            traceLayerIndices: nil
        )
        return (result.output, result.cache)
    }

    public func trace(
        hiddenStates: MLXArray,
        timestep: MLXArray,
        timestepR: MLXArray,
        encoderHiddenStates: MLXArray,
        contextLatents: MLXArray,
        encoderAttentionMask: MLXArray?,
        cache: CrossAttentionCache?,
        useCache: Bool,
        layerIndices: Set<Int>
    ) -> DiTTraceOutput {
        let result = run(
            hiddenStates: hiddenStates,
            timestep: timestep,
            timestepR: timestepR,
            encoderHiddenStates: encoderHiddenStates,
            contextLatents: contextLatents,
            encoderAttentionMask: encoderAttentionMask,
            cache: cache,
            useCache: useCache,
            traceLayerIndices: layerIndices
        )
        return DiTTraceOutput(output: result.output, tensors: result.trace)
    }

    private func run(
        hiddenStates: MLXArray,
        timestep: MLXArray,
        timestepR: MLXArray,
        encoderHiddenStates: MLXArray,
        contextLatents: MLXArray,
        encoderAttentionMask: MLXArray?,
        cache: CrossAttentionCache?,
        useCache: Bool,
        traceLayerIndices: Set<Int>?
    ) -> (output: MLXArray, cache: CrossAttentionCache?, trace: [String: MLXArray]) {
        var trace: [String: MLXArray] = [:]
        let shouldTrace = traceLayerIndices != nil

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
        if shouldTrace { trace["proj_in"] = h }

        // Project encoder states
        let encStates = conditionEmbedder(encoderHiddenStates)
        if shouldTrace { trace["condition_embed"] = encStates }

        let seqLen = h.shape[1]

        // Position embeddings
        let (cos, sin) = rotaryEmb.forward(seqLen: seqLen)

        // Sliding mask
        let hasSlidingLayer = layerTypes.contains("sliding_attention")
        let slidingMask: MLXArray? = hasSlidingLayer ? getSlidingMask(seqLen: seqLen) : nil

        // Match upstream: do NOT pass an encoder_attention_mask to cross-
        // attention. Upstream MLX (acestep/models/mlx/dit_model.py:591)
        // hardcodes encoder_attention_mask=None when calling each layer;
        // upstream Python (modeling_acestep_v15_turbo.py:1380-1382) shadows
        // its forward parameter to None and rebuilds a shape-only 4D mask
        // with attention_mask=None (no padding info). The model was trained
        // without padding mask in cross-attention, so applying one at
        // inference shifts the attention distribution away from training
        // and degrades prompt-following.
        let _ = encoderAttentionMask  // intentionally unused

        // Process through transformer layers
        let tracedLayers = traceLayerIndices ?? []
        for (layerIndex, layer) in layers.enumerated() {
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
            if tracedLayers.contains(layerIndex) {
                trace["layers.\(layerIndex)"] = h
            }
        }

        // Output adaptive layer norm
        let expandedTemb = temb.expandedDimensions(axis: 1)
        let outMod = scaleShiftTable + expandedTemb
        let outParts = outMod.split(parts: 2, axis: 1)
        let (shift, scale) = (outParts[0], outParts[1])
        h = normOut(h) * (1.0 + scale) + shift
        if shouldTrace { trace["post_norm_out"] = h }

        // De-patchify via transposed convolution
        h = projOut(h)

        // Crop back to original sequence length
        h = h[0..., 0..<originalSeqLen, 0...]
        if shouldTrace { trace["output"] = h }

        return (h, cache, trace)
    }
}
