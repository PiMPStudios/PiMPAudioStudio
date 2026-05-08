// AnvilApp/Services/SFXGenerationService.swift
import Foundation
import AnvilCore
import OSLog

private let logger = Logger(subsystem: "com.pimp.AnvilApp", category: "SFXGenerationService")

/// Handles Stable Audio / SFX generation. Internal to the GenerationService facade.
/// Emits typed events via AsyncThrowingStream; the facade maps them to GenerationState.
@MainActor
final class SFXGenerationService {
    enum Event {
        case downloadingModel(progress: Float)
        case generating(step: Int, totalSteps: Int, eta: TimeInterval)
        case decoding
        case result(AudioResult)
    }

    private let sfxModelManager: SFXModelManager

    init(sfxModelManager: SFXModelManager) {
        self.sfxModelManager = sfxModelManager
    }

    func generate(request: GenerationRequest) -> AsyncThrowingStream<Event, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try await sfxModelManager.ensureActiveModel()
                    if !sfxModelManager.isReady {
                        continuation.yield(.downloadingModel(progress: 0))
                        try await sfxModelManager.downloadIfNeeded()
                        try await sfxModelManager.loadPipeline()
                    }
                    guard let pipeline = sfxModelManager.pipeline else {
                        throw GenerationError.pipelineNotReady
                    }
                    guard let tokenizer = sfxModelManager.tokenizer else {
                        throw GenerationError.pipelineNotReady
                    }

                    try Task.checkCancellation()

                    let totalSteps = 100
                    let clampedDuration = min(request.duration, 47.0)
                    let config = SFXDiffusionConfig(
                        steps: totalSteps,
                        guidanceScale: request.guidanceScale > 1.0 ? request.guidanceScale : 7.0,
                        seed: request.seed,
                        latentShape: [1, Int(ceil(clampedDuration * 44100.0 / 2048.0)), 64]
                    )

                    continuation.yield(.generating(step: 0, totalSteps: totalSteps, eta: 0))

                    let progressStream = await pipeline.makeProgressStream()
                    let progressTask = Task {
                        for await progress in progressStream {
                            if case .diffusion(let dp) = progress {
                                let perStep = dp.currentStep > 0
                                    ? dp.elapsedTime / Double(dp.currentStep) : 0
                                let eta = perStep * Double(dp.totalSteps - dp.currentStep)
                                continuation.yield(.generating(
                                    step: dp.currentStep,
                                    totalSteps: dp.totalSteps,
                                    eta: eta
                                ))
                            }
                        }
                    }
                    defer { progressTask.cancel() }

                    let audioResult = try await pipeline.generate(
                        text: request.text,
                        duration: clampedDuration,
                        config: config,
                        tokenizer: tokenizer
                    )

                    continuation.yield(.decoding)
                    continuation.yield(.result(audioResult))
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}
