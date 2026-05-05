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
