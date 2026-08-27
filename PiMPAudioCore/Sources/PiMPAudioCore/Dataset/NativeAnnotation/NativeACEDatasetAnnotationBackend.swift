import Foundation
import MLX

public actor NativeACEDatasetAnnotationBackend: DatasetAnnotationBackend {
    public nonisolated let backendID = "native-ace-step-annotation"
    public static let backendVersion = "1"
    public static let promptVersion = "1"
    public static let captionerDirectoryName = "ace-captioner-4bit-v1"
    public static let transcriberDirectoryName = "ace-transcriber-4bit-v1"

    private let captionerDirectory: URL
    private let transcriberDirectory: URL
    private let physicalMemoryBytes: UInt64
    private var engine: OmniAnnotationEngine?
    private var loadedTask: DatasetAnnotationTask?
    private var loadedManifest: NativeModelPackManifest?

    public init(
        captionerDirectory: URL,
        transcriberDirectory: URL,
        physicalMemoryBytes: UInt64 = ProcessInfo.processInfo.physicalMemory
    ) {
        self.captionerDirectory = captionerDirectory
        self.transcriberDirectory = transcriberDirectory
        self.physicalMemoryBytes = physicalMemoryBytes
    }

    public init(
        modelsDirectory: URL,
        physicalMemoryBytes: UInt64 = ProcessInfo.processInfo.physicalMemory
    ) {
        self.init(
            captionerDirectory: modelsDirectory.appendingPathComponent(
                Self.captionerDirectoryName,
                isDirectory: true
            ),
            transcriberDirectory: modelsDirectory.appendingPathComponent(
                Self.transcriberDirectoryName,
                isDirectory: true
            ),
            physicalMemoryBytes: physicalMemoryBytes
        )
    }

    public func availability() -> DatasetAnnotationBackendAvailability {
        DatasetAnnotationBackendAvailability(
            backendID: backendID,
            hardwareTier: DatasetAnnotationHardwareTier.classify(
                physicalMemoryBytes: physicalMemoryBytes
            ),
            models: [
                inspectInstallation(task: .caption),
                inspectInstallation(task: .transcribe),
            ]
        )
    }

    public func annotate(
        requests: [DatasetAnnotationRequest],
        task: DatasetAnnotationTask,
        options: DatasetAnnotationGenerationOptions = .greedy,
        onProgress: @Sendable @escaping (DatasetAnnotationProgress) -> Void = { _ in }
    ) async throws -> [DatasetAnnotationBackendResult] {
        guard !requests.isEmpty else { return [] }
        try Task.checkCancellation()

        let total = requests.count
        onProgress(
            DatasetAnnotationProgress(
                task: task,
                stage: .validatingModelPack,
                completedCount: 0,
                totalCount: total
            )
        )

        do {
            let manifest = try await loadEngineIfNeeded(
                for: task,
                totalCount: total,
                onProgress: onProgress
            )
            guard let engine else {
                throw NativeACEDatasetAnnotationBackendError.engineUnavailable
            }

            var results: [DatasetAnnotationBackendResult] = []
            results.reserveCapacity(total)
            for (index, request) in requests.enumerated() {
                try Task.checkCancellation()
                onProgress(
                    DatasetAnnotationProgress(
                        task: task,
                        stage: .annotating,
                        completedCount: index,
                        totalCount: total,
                        currentRequestID: request.id
                    )
                )
                let nativeResult = try engine.annotate(
                    audioURL: request.audioURL,
                    task: task.omniTask,
                    options: options.omniOptions,
                    cancellationCheck: { try Task.checkCancellation() }
                )
                results.append(
                    DatasetAnnotationBackendResult(
                        requestID: request.id,
                        task: task,
                        text: nativeResult.text,
                        rawText: nativeResult.rawText,
                        language: nativeResult.language,
                        descriptor: descriptor(for: manifest),
                        settings: options.persistedSettings,
                        generatedTokenCount: nativeResult.generatedTokenIDs.count,
                        promptTokenCount: nativeResult.promptTokenCount,
                        elapsedSeconds: nativeResult.elapsedSeconds,
                        diagnostics: OmniOutputDiagnostics(text: nativeResult.text)
                    )
                )
            }

            onProgress(
                DatasetAnnotationProgress(
                    task: task,
                    stage: .evictingModel,
                    completedCount: total,
                    totalCount: total
                )
            )
            evictLoadedEngine()
            return results
        } catch {
            evictLoadedEngine()
            throw error
        }
    }

    public func evict() {
        evictLoadedEngine()
    }

    private func loadEngineIfNeeded(
        for task: DatasetAnnotationTask,
        totalCount: Int,
        onProgress: @Sendable (DatasetAnnotationProgress) -> Void
    ) async throws -> NativeModelPackManifest {
        if loadedTask == task, let loadedManifest, engine?.isLoaded == true {
            return loadedManifest
        }

        evictLoadedEngine()
        try Task.checkCancellation()
        onProgress(
            DatasetAnnotationProgress(
                task: task,
                stage: .loadingModel,
                completedCount: 0,
                totalCount: totalCount
            )
        )

        let candidate = try OmniAnnotationEngine()
        do {
            let directory = modelDirectory(for: task)
            try await candidate.load(
                from: directory,
                cancellationCheck: { try Task.checkCancellation() }
            )
            let manifest = try NativeModelPackValidator.loadManifestMetadata(
                at: directory
            )
            guard manifest.task == task.omniTask else {
                throw NativeACEDatasetAnnotationBackendError.taskMismatch(
                    expected: task,
                    actual: manifest.task.datasetTask
                )
            }
            engine = candidate
            loadedTask = task
            loadedManifest = manifest
            return manifest
        } catch {
            candidate.evict()
            throw error
        }
    }

    private func evictLoadedEngine() {
        engine?.evict()
        engine = nil
        loadedTask = nil
        loadedManifest = nil
        Memory.clearCache()
    }

    private func modelDirectory(for task: DatasetAnnotationTask) -> URL {
        switch task {
        case .caption:
            captionerDirectory
        case .transcribe:
            transcriberDirectory
        }
    }

    private func inspectInstallation(
        task: DatasetAnnotationTask
    ) -> DatasetAnnotationModelAvailability {
        let directory = modelDirectory(for: task)
        do {
            let manifest = try NativeModelPackValidator.loadManifestMetadata(
                at: directory
            )
            guard manifest.task == task.omniTask else {
                return DatasetAnnotationModelAvailability(
                    task: task,
                    directory: directory,
                    isInstalled: false,
                    unavailableReason: "The installed pack is for \(manifest.task.rawValue), not \(task.rawValue)."
                )
            }
            for record in manifest.files {
                let fileURL = directory.appendingPathComponent(record.path)
                let values = try fileURL.resourceValues(
                    forKeys: [.fileSizeKey, .isRegularFileKey]
                )
                guard values.isRegularFile == true,
                      Int64(values.fileSize ?? -1) == record.byteCount
                else {
                    return DatasetAnnotationModelAvailability(
                        task: task,
                        directory: directory,
                        isInstalled: false,
                        unavailableReason: "\(record.path) is missing or incomplete."
                    )
                }
            }
            return DatasetAnnotationModelAvailability(
                task: task,
                directory: directory,
                isInstalled: true,
                descriptor: descriptor(for: manifest)
            )
        } catch {
            return DatasetAnnotationModelAvailability(
                task: task,
                directory: directory,
                isInstalled: false,
                unavailableReason: error.localizedDescription
            )
        }
    }

    private func descriptor(
        for manifest: NativeModelPackManifest
    ) -> DatasetAnnotationBackendDescriptor {
        DatasetAnnotationBackendDescriptor(
            id: backendID,
            version: Self.backendVersion,
            modelID: manifest.upstreamModelID,
            modelRevision: manifest.upstreamRevision,
            promptVersion: Self.promptVersion,
            quantization: "q\(manifest.quantization.bits)-\(manifest.quantization.mode)-g\(manifest.quantization.groupSize)"
        )
    }
}

public enum NativeACEDatasetAnnotationBackendError: LocalizedError {
    case engineUnavailable
    case taskMismatch(
        expected: DatasetAnnotationTask,
        actual: DatasetAnnotationTask
    )

    public var errorDescription: String? {
        switch self {
        case .engineUnavailable:
            "The native ACE annotation engine did not finish loading."
        case let .taskMismatch(expected, actual):
            "The \(actual.rawValue) model pack cannot run a \(expected.rawValue) request."
        }
    }
}

private extension DatasetAnnotationTask {
    var omniTask: OmniAnnotationTask {
        switch self {
        case .caption:
            .caption
        case .transcribe:
            .transcribe
        }
    }
}

private extension OmniAnnotationTask {
    var datasetTask: DatasetAnnotationTask {
        switch self {
        case .caption:
            .caption
        case .transcribe:
            .transcribe
        }
    }
}

private extension DatasetAnnotationGenerationOptions {
    var omniOptions: OmniGenerationOptions {
        OmniGenerationOptions(
            maxTokens: maxTokens,
            temperature: temperature,
            topP: topP,
            repetitionPenalty: repetitionPenalty
        )
    }
}
