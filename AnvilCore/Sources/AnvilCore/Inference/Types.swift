import Foundation
import MLX

public enum Precision: String, Sendable, CaseIterable {
    case float32
    case float16

    public var mlxDtype: DType {
        switch self {
        case .float32: return .float32
        case .float16: return .float16
        }
    }
}

public enum InferMethod: String, Sendable {
    case ode
    case sde
}

public enum ModelComponent: Sendable {
    case vae
    case dit
    case qwen3
    case conditionEncoder
}

public struct DiffusionConfig: Sendable {
    public var seed: Int?
    public var method: InferMethod
    public var shift: Float
    public var steps: Int?
    public var guidanceScale: Float
    public var cfgIntervalStart: Float
    public var cfgIntervalEnd: Float
    public var audioCoverStrength: Float
    public var precision: Precision

    public init(
        seed: Int? = nil,
        method: InferMethod = .ode,
        // Upstream MLX-specific handler (acestep/core/generation/handler/
        // diffusion.py) defaults shift=3.0 — that's the actual MLX inference
        // default, used end-to-end in production. (The Python-only
        // inference.py defaults 1.0, but that's a different code path.)
        // The pre-baked 8-step Turbo schedule in shiftTimesteps[3.0] also
        // assumes 3.0.
        shift: Float = 3.0,
        steps: Int? = nil,
        guidanceScale: Float = 1.0,
        cfgIntervalStart: Float = 0.0,
        cfgIntervalEnd: Float = 1.0,
        audioCoverStrength: Float = 1.0,
        precision: Precision = .float32
    ) {
        self.seed = seed
        self.method = method
        self.shift = shift
        self.steps = steps
        self.guidanceScale = guidanceScale
        self.cfgIntervalStart = cfgIntervalStart
        self.cfgIntervalEnd = cfgIntervalEnd
        self.audioCoverStrength = audioCoverStrength
        self.precision = precision
    }
}

public struct TimeCosts: Sendable {
    public let diffusionTime: TimeInterval
    public let perStepTime: TimeInterval
    public let totalTime: TimeInterval

    public init(diffusionTime: TimeInterval, perStepTime: TimeInterval, totalTime: TimeInterval) {
        self.diffusionTime = diffusionTime
        self.perStepTime = perStepTime
        self.totalTime = totalTime
    }
}

public struct DiffusionProgress: Sendable {
    public let currentStep: Int
    public let totalSteps: Int
    public let elapsedTime: TimeInterval

    public init(currentStep: Int, totalSteps: Int, elapsedTime: TimeInterval) {
        self.currentStep = currentStep
        self.totalSteps = totalSteps
        self.elapsedTime = elapsedTime
    }
}

public struct ConversionProgress: Sendable {
    public let current: Int
    public let total: Int
    public let currentTensorName: String

    public init(current: Int, total: Int, currentTensorName: String) {
        self.current = current
        self.total = total
        self.currentTensorName = currentTensorName
    }
}

public enum PipelineState: Sendable, Equatable {
    case unloaded
    case loading
    case ready
    case generating(step: Int, totalSteps: Int)
    case error(String)
}

public enum GenerationProgress: Sendable {
    case loadingWeights(Float)
    case diffusion(DiffusionProgress)
    case decoding
    case complete
}

public struct AudioResult: @unchecked Sendable {
    public let samples: MLXArray
    public let sampleRate: Int
    public let duration: TimeInterval
    public let timeCosts: TimeCosts

    public init(samples: MLXArray, sampleRate: Int = 44100, duration: TimeInterval, timeCosts: TimeCosts) {
        self.samples = samples
        self.sampleRate = sampleRate
        self.duration = duration
        self.timeCosts = timeCosts
    }
}

public struct DiffusionResult: @unchecked Sendable {
    public let latents: MLXArray
    public let timeCosts: TimeCosts

    public init(latents: MLXArray, timeCosts: TimeCosts) {
        self.latents = latents
        self.timeCosts = timeCosts
    }
}

public struct ConditioningResult: @unchecked Sendable {
    public let encoderStates: MLXArray
    public let encoderMask: MLXArray
    public let contextLatents: MLXArray

    public init(encoderStates: MLXArray, encoderMask: MLXArray, contextLatents: MLXArray) {
        self.encoderStates = encoderStates
        self.encoderMask = encoderMask
        self.contextLatents = contextLatents
    }
}

public struct SFXVAEConfig: Codable, Sendable {
    public let latentDim: Int
    public let outputChannels: Int
    public let sampleRate: Int
    public let upsampleRatios: [Int]

    public init(latentDim: Int, outputChannels: Int, sampleRate: Int, upsampleRatios: [Int]) {
        self.latentDim = latentDim
        self.outputChannels = outputChannels
        self.sampleRate = sampleRate
        self.upsampleRatios = upsampleRatios
    }
}

public struct SFXDiffusionConfig: Sendable {
    public let steps: Int
    public let guidanceScale: Float
    public let seed: Int?
    public let latentShape: [Int]

    public init(steps: Int = 100, guidanceScale: Float = 7.0, seed: Int? = nil, latentShape: [Int]) {
        self.steps = steps
        self.guidanceScale = guidanceScale
        self.seed = seed
        self.latentShape = latentShape
    }
}

public struct SFXDiTConfig: Codable, Sendable {
    public let hiddenSize: Int
    public let numLayers: Int
    public let numHeads: Int
    public let latentDim: Int
    public let conditioningDim: Int

    public init(hiddenSize: Int, numLayers: Int, numHeads: Int, latentDim: Int, conditioningDim: Int) {
        self.hiddenSize = hiddenSize
        self.numLayers = numLayers
        self.numHeads = numHeads
        self.latentDim = latentDim
        self.conditioningDim = conditioningDim
    }
}
