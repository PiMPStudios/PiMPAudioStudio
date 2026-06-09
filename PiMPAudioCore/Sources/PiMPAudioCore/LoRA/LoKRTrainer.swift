import Foundation
import MLX
import MLXNN
import MLXOptimizers
import MLXRandom

/// Flow-matching training loop for experimental LoKR adapters.
public final class LoKRTrainer {
    private let model: DiTDecoder
    private let config: LoKRConfig
    private let injectedPaths: [String]
    private var bestLoss: Float = .infinity
    private var bestWeights: [(String, MLXArray)]?
    private var isCancelled: Bool = false

    private var globalStep: Int = 0
    private var currentEpoch: Int = 0
    private var lossHistory: [Float] = []
    private let scheduler: CosineScheduler
    private let totalSteps: Int

    public init(
        model: DiTDecoder,
        config: LoKRConfig,
        injectedPaths: [String],
        datasetSize: Int,
        resumeFrom checkpoint: TrainingCheckpoint? = nil,
        initialBestWeights: [(String, MLXArray)]? = nil
    ) {
        self.model = model
        self.config = config
        self.injectedPaths = injectedPaths

        let updatesPerEpoch = LoRATrainer.optimizerStepsPerEpoch(
            datasetSize: datasetSize,
            gradientAccumulationSteps: config.gradientAccumulationSteps
        )
        self.totalSteps = updatesPerEpoch * config.maxEpochs
        self.scheduler = CosineScheduler(
            baseLR: config.learningRate,
            warmupFraction: config.warmupFraction,
            totalSteps: totalSteps
        )

        if let checkpoint {
            self.globalStep = checkpoint.globalStep
            self.currentEpoch = checkpoint.epoch
            self.lossHistory = checkpoint.lossHistory
            self.bestLoss = checkpoint.bestLoss ?? .infinity
            if checkpoint.bestLoss != nil {
                self.bestWeights = initialBestWeights
                    ?? LoKRInjector.extractWeights(model: model, paths: injectedPaths)
            }
        }
    }

    public func train(
        dataset: [TrainingSample],
        onProgress: @escaping @Sendable (TrainingProgress) -> Void
    ) async throws -> [(String, MLXArray)] {
        AudioDiagnostics.note(
            "training/trainer/begin",
            "kind=lokr dataset=\(dataset.count) paths=\(injectedPaths.count) maxEpochs=\(config.maxEpochs) gradAccum=\(config.gradientAccumulationSteps) weightDecompose=\(config.weightDecompose)"
        )
        AudioDiagnostics.memory("training/trainer/begin")
        guard !dataset.isEmpty else {
            AudioDiagnostics.note("training/trainer/emptyDataset", "kind=lokr")
            return []
        }

        let accumSteps = max(1, min(config.gradientAccumulationSteps, dataset.count))
        let optimizer = AdamW(learningRate: scheduler.learningRate(atStep: globalStep))
        let startTime = Date()

        setTrainingMode(true)
        AudioDiagnostics.note(
            "training/trainer/trainingMode",
            "kind=lokr enabled=true accumSteps=\(accumSteps)"
        )
        defer {
            setTrainingMode(false)
            AudioDiagnostics.note("training/trainer/trainingMode", "kind=lokr enabled=false")
        }

        var hasLoggedGradientClosure = false
        let trainingModel = model
        let trainableParameterKeys = model.trainableParameters().flattened().map(\.0)
        let trainableArgumentCount = trainableParameterKeys.count
        let trainableParameterKeyPreview = Array(trainableParameterKeys.prefix(24))
        let omittedTrainableParameterKeyCount = max(0, trainableParameterKeys.count - trainableParameterKeyPreview.count)
        let gradientArgumentNumbers = Array(0..<trainableArgumentCount)
        let lossGrad = valueAndGrad({ (allArgs: [MLXArray]) -> [MLXArray] in
            let shouldLogGradientClosure = !hasLoggedGradientClosure
            if shouldLogGradientClosure {
                hasLoggedGradientClosure = true
                AudioDiagnostics.note(
                    "training/trainer/gradClosure",
                    "kind=lokr paramCount=\(trainableParameterKeys.count) paramKeysPreview=\(trainableParameterKeyPreview) omitted=\(omittedTrainableParameterKeyCount)"
                )
            }
            let parameterValues = Array(allArgs.prefix(trainableArgumentCount))
            let parameterPairs = zip(trainableParameterKeys, parameterValues).map { ($0.0, $0.1) }
            trainingModel.update(parameters: ModuleParameters.unflattened(parameterPairs))

            let argOffset = trainableArgumentCount
            let xt = allArgs[argOffset]
            let t = allArgs[argOffset + 1]
            let encoderStates = allArgs[argOffset + 2]
            let contextLatents = allArgs[argOffset + 3]
            let vTarget = allArgs[argOffset + 4]
            let encoderMask: MLXArray? = allArgs.count > argOffset + 5
                ? allArgs[argOffset + 5]
                : nil

            let (vPred, _) = trainingModel(
                hiddenStates: xt,
                timestep: t,
                timestepR: t,
                encoderHiddenStates: encoderStates,
                contextLatents: contextLatents,
                encoderAttentionMask: encoderMask,
                cache: nil,
                useCache: false
            )
            let diff = (vPred - vTarget).asType(.float32)
            if shouldLogGradientClosure {
                AudioDiagnostics.note(
                    "training/trainer/gradClosureReady",
                    "kind=lokr xt=\(xt.shape) encoder=\(encoderStates.shape) context=\(contextLatents.shape) mask=\(encoderMask?.shape ?? [])"
                )
                AudioDiagnostics.memory("training/trainer/gradClosureReady")
            }
            return [(diff * diff).mean()]
        }, argumentNumbers: gradientArgumentNumbers)
        AudioDiagnostics.note(
            "training/trainer/valueAndGradReady",
            "kind=lokr gradientRoute=flatAdapterParameters trainableArgumentCount=\(trainableArgumentCount)"
        )
        let usesExplicitDropoutKeys = config.dropout > 0
        let adapterDropoutKeyCount = max(1, injectedPaths.count * 2)
        AudioDiagnostics.note(
            "training/trainer/dropoutRNG",
            "kind=lokr mode=\(usesExplicitDropoutKeys ? "explicitKey" : "disabled") dropout=\(config.dropout) keyCount=\(usesExplicitDropoutKeys ? adapterDropoutKeyCount : 0)"
        )

        var hasLoggedFirstProcessedSample = false
        for epoch in (currentEpoch + 1)...config.maxEpochs {
            guard !isCancelled else { break }
            currentEpoch = epoch

            var epochLoss: Float = 0
            var epochSamples = 0
            var windowLoss: Float = 0
            var windowSamples = 0
            var accumulatedFlatGrads: [MLXArray]?
            let shuffledIndices = Array(0..<dataset.count).shuffled()
            AudioDiagnostics.note(
                "training/trainer/epochBegin",
                "kind=lokr epoch=\(epoch)/\(config.maxEpochs) shuffled=\(shuffledIndices.count)"
            )

            for idx in shuffledIndices {
                guard !isCancelled else { break }
                try Task.checkCancellation()
                let sample = dataset[idx]
                let shouldLogSample = !hasLoggedFirstProcessedSample
                if shouldLogSample {
                    hasLoggedFirstProcessedSample = true
                    AudioDiagnostics.note(
                        "training/trainer/firstSample",
                        "kind=lokr idx=\(idx) latent=\(sample.latent.shape) context=\(sample.contextLatents.shape) encoder=\(sample.encoderStates.shape) mask=\(sample.encoderMask?.shape ?? []) paramCount=\(trainableParameterKeys.count) paramKeysPreview=\(trainableParameterKeyPreview) omitted=\(omittedTrainableParameterKeyCount)"
                    )
                }

                let t = MLXRandom.uniform(0..<1, [1])
                let noise = MLXRandom.normal(sample.latent.shape)
                let xt = (1.0 - t) * sample.latent + t * noise
                let vTarget = noise - sample.latent

                var args = [xt, t, sample.encoderStates, sample.contextLatents, vTarget]
                if let mask = sample.encoderMask {
                    args.append(mask)
                }
                if shouldLogSample {
                    AudioDiagnostics.note(
                        "training/trainer/firstValueAndGrad",
                        "kind=lokr argShapes=\(args.map { $0.shape })"
                    )
                }

                let lossGradStart = Date()
                if shouldLogSample {
                    AudioDiagnostics.note(
                        "training/trainer/firstLossGradBegin",
                        "kind=lokr args=\(args.count) trainableArgumentCount=\(trainableArgumentCount)"
                    )
                }
                let firstBackwardWatchdog = shouldLogSample
                    ? Self.startFirstBackwardWatchdog(
                        kind: "lokr",
                        argsCount: args.count,
                        trainableArgumentCount: trainableArgumentCount,
                        startedAt: lossGradStart
                    )
                    : nil
                let trainableValues = model.trainableParameters().flattened().map(\.1)
                let dropoutKeyStream = usesExplicitDropoutKeys
                    ? AdapterDropoutKeyContext.makeKeyStream(
                        seed: AdapterDropoutKeyContext.seed(
                            epoch: epoch,
                            globalStep: globalStep,
                            sampleIndex: idx,
                            pathCount: injectedPaths.count
                        ),
                        count: adapterDropoutKeyCount
                    )
                    : nil
                let (lossValues, flatGrads) = AdapterDropoutKeyContext.withKeyStream(dropoutKeyStream) {
                    lossGrad(trainableValues + args)
                }
                firstBackwardWatchdog?.cancel()
                let lossGradElapsed = Date().timeIntervalSince(lossGradStart)
                if shouldLogSample {
                    AudioDiagnostics.note(
                        "training/trainer/firstLossGradEnd",
                        "kind=lokr elapsed=\(Self.formatSeconds(lossGradElapsed)) values=\(lossValues.count) flatGrads=\(flatGrads.count)"
                    )
                }
                let lossItemStart = Date()
                if shouldLogSample {
                    AudioDiagnostics.note("training/trainer/firstLossItemBegin", "kind=lokr")
                }
                let lossValue = lossValues[0].item(Float.self)
                let lossItemElapsed = Date().timeIntervalSince(lossItemStart)
                if shouldLogSample {
                    AudioDiagnostics.note(
                        "training/trainer/firstLoss",
                        "kind=lokr loss=\(lossValue) flatGrads=\(flatGrads.count) lossGradElapsed=\(Self.formatSeconds(lossGradElapsed)) itemElapsed=\(Self.formatSeconds(lossItemElapsed))"
                    )
                }

                if let current = accumulatedFlatGrads {
                    accumulatedFlatGrads = Self.addFlatGradients(current, flatGrads)
                } else {
                    accumulatedFlatGrads = flatGrads
                }

                epochLoss += lossValue
                epochSamples += 1
                windowLoss += lossValue
                windowSamples += 1

                if windowSamples >= accumSteps,
                   let windowFlatGrads = accumulatedFlatGrads {
                    updateOptimizerForAccumulationWindow(
                        optimizer: optimizer,
                        trainableParameterKeys: trainableParameterKeys,
                        accumulatedFlatGrads: windowFlatGrads,
                        windowLoss: windowLoss,
                        windowSamples: windowSamples,
                        epoch: epoch,
                        startTime: startTime,
                        onProgress: onProgress
                    )
                    windowLoss = 0
                    windowSamples = 0
                    accumulatedFlatGrads = nil
                }
            }

            if windowSamples > 0,
               let windowFlatGrads = accumulatedFlatGrads {
                updateOptimizerForAccumulationWindow(
                    optimizer: optimizer,
                    trainableParameterKeys: trainableParameterKeys,
                    accumulatedFlatGrads: windowFlatGrads,
                    windowLoss: windowLoss,
                    windowSamples: windowSamples,
                    epoch: epoch,
                    startTime: startTime,
                    onProgress: onProgress
                )
            }

            let avgLoss = epochSamples > 0 ? epochLoss / Float(epochSamples) : .infinity
            if avgLoss < bestLoss {
                bestLoss = avgLoss
                bestWeights = LoKRInjector.extractWeights(model: model, paths: injectedPaths)
            }
            AudioDiagnostics.note(
                "training/trainer/epochEnd",
                "kind=lokr epoch=\(epoch)/\(config.maxEpochs) samples=\(epochSamples) avgLoss=\(avgLoss) bestLoss=\(bestLoss) globalStep=\(globalStep)"
            )
            AudioDiagnostics.memory("training/trainer/epochEnd")
            await Task.yield()
        }

        AudioDiagnostics.note(
            "training/trainer/end",
            "kind=lokr cancelled=\(isCancelled) globalStep=\(globalStep) bestLoss=\(bestWeights != nil ? String(bestLoss) : "nil")"
        )
        AudioDiagnostics.memory("training/trainer/end")
        return bestWeights ?? LoKRInjector.extractWeights(model: model, paths: injectedPaths)
    }

    public func cancel() {
        isCancelled = true
    }

    public var lowestLoss: Float? {
        bestWeights != nil ? bestLoss : nil
    }

    public func makeCheckpoint(
        weightsURL: URL,
        optimizerStateURL: URL,
        baseModelID: String
    ) -> TrainingCheckpoint {
        TrainingCheckpoint(
            epoch: currentEpoch,
            globalStep: globalStep,
            totalSteps: totalSteps,
            bestLoss: bestWeights != nil ? bestLoss : nil,
            lossHistory: lossHistory,
            weightsURL: weightsURL,
            optimizerStateURL: optimizerStateURL,
            config: config.loraCompatibleConfig,
            baseModelID: baseModelID
        )
    }

    public var currentLossHistory: [Float] { lossHistory }

    public var currentGlobalStep: Int { globalStep }

    private static func movingAverage(_ values: [Float], window: Int = 10) -> Float? {
        let recent = values.suffix(max(1, window))
        guard !recent.isEmpty else { return nil }
        return recent.reduce(0, +) / Float(recent.count)
    }

    private static func addFlatGradients(_ lhs: [MLXArray], _ rhs: [MLXArray]) -> [MLXArray] {
        zip(lhs, rhs).map { pair in pair.0 + pair.1 }
    }

    private static func formatSeconds(_ value: TimeInterval) -> String {
        String(format: "%.3fs", value)
    }

    private static func startFirstBackwardWatchdog(
        kind: String,
        argsCount: Int,
        trainableArgumentCount: Int,
        startedAt: Date
    ) -> Task<Void, Never> {
        Task.detached(priority: .utility) {
            var previousThreshold = 0
            for threshold in [30, 60, 120, 300] {
                let delay = max(0, threshold - previousThreshold)
                previousThreshold = threshold
                try? await Task.sleep(for: .seconds(delay))
                guard !Task.isCancelled else { return }
                let elapsed = Date().timeIntervalSince(startedAt)
                AudioDiagnostics.note(
                    "training/trainer/firstLossGradWatchdog",
                    "kind=\(kind) elapsed=\(Self.formatSeconds(elapsed)) args=\(argsCount) trainableArgumentCount=\(trainableArgumentCount)"
                )
                AudioDiagnostics.memory("training/trainer/firstLossGradWatchdog")
            }
        }
    }

    private func updateOptimizerForAccumulationWindow(
        optimizer: AdamW,
        trainableParameterKeys: [String],
        accumulatedFlatGrads: [MLXArray],
        windowLoss: Float,
        windowSamples: Int,
        epoch: Int,
        startTime: Date,
        onProgress: @escaping @Sendable (TrainingProgress) -> Void
    ) {
        let currentLR = scheduler.learningRate(atStep: globalStep)
        optimizer.learningRate = currentLR

        let scale = 1.0 / Float(max(1, windowSamples))
        let averagedFlatGrads = accumulatedFlatGrads.map { $0 * scale }
        let beforeUpdate = TrainingMetricMath.materializedSnapshot(model.trainableParameters().flattened().map(\.1))
        let gradPairs = zip(trainableParameterKeys, averagedFlatGrads).map { ($0.0, $0.1) }
        optimizer.update(model: model, gradients: ModuleParameters.unflattened(gradPairs))
        eval(model.trainableParameters(), optimizer.innerState())
        let afterUpdate = model.trainableParameters().flattened().map(\.1)
        let gradientNorm = TrainingMetricMath.l2Norm(averagedFlatGrads)
        let parameterNorm = TrainingMetricMath.l2Norm(afterUpdate)
        let updateNorm = TrainingMetricMath.l2NormOfDeltas(before: beforeUpdate, after: afterUpdate)

        globalStep += 1

        let avgWindowLoss = windowLoss / Float(max(1, windowSamples))
        lossHistory.append(avgWindowLoss)
        let lossMovingAverage = Self.movingAverage(lossHistory)

        let elapsed = Date().timeIntervalSince(startTime)
        let stepsRemaining = max(0, totalSteps - globalStep)
        let eta: TimeInterval? = globalStep > 0
            ? elapsed / Double(globalStep) * Double(stepsRemaining)
            : nil

        onProgress(TrainingProgress(
            epoch: epoch,
            totalEpochs: config.maxEpochs,
            step: globalStep,
            loss: avgWindowLoss,
            learningRate: currentLR,
            lossMovingAverage: lossMovingAverage,
            gradientNorm: gradientNorm,
            parameterNorm: parameterNorm,
            updateNorm: updateNorm,
            estimatedTimeRemaining: eta,
            lossHistory: lossHistory,
            elapsedTime: elapsed
        ))
    }

    private func setTrainingMode(_ training: Bool) {
        for path in injectedPaths {
            guard let (layerIdx, subPath) = parsePath(path),
                  layerIdx < model.layers.count else { continue }

            let layer = model.layers[layerIdx]
            if let lokrLinear = resolveLoKRLinear(in: layer, path: subPath) {
                lokrLinear.isTraining = training
            }
        }
    }

    private func parsePath(_ path: String) -> (Int, String)? {
        let components = path.split(separator: ".")
        guard components.count >= 3,
              components[0] == "layers",
              let layerIdx = Int(components[1]) else { return nil }
        let subPath = components[2...].joined(separator: ".")
        return (layerIdx, subPath)
    }

    private func resolveLoKRLinear(in layer: DiTLayer, path: String) -> LoKRLinear? {
        let parts = path.split(separator: ".")
        guard parts.count == 2 else { return nil }
        let moduleName = String(parts[0])
        let projName = String(parts[1])

        let linear: Linear?
        switch moduleName {
        case "selfAttn":
            linear = resolveAttentionProj(layer.selfAttn, proj: projName)
        case "crossAttn":
            linear = resolveAttentionProj(layer.crossAttn, proj: projName)
        case "mlp":
            linear = resolveMLPProj(layer.mlp, proj: projName)
        default:
            linear = nil
        }

        return linear as? LoKRLinear
    }

    private func resolveAttentionProj(_ attn: DiTAttention, proj: String) -> Linear? {
        switch proj {
        case "qProj": return attn.qProj
        case "kProj": return attn.kProj
        case "vProj": return attn.vProj
        case "oProj": return attn.oProj
        default: return nil
        }
    }

    private func resolveMLPProj(_ mlp: SwiGLUMLP, proj: String) -> Linear? {
        switch proj {
        case "gateProj": return mlp.gateProj
        case "upProj": return mlp.upProj
        case "downProj": return mlp.downProj
        default: return nil
        }
    }
}
