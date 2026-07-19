import Accelerate
import AVFoundation
import Foundation
import PiMPAudioCore
import MLX
import OSLog

private let enrichmentMemoryLogger = Logger(
    subsystem: "com.pimpstudios.PiMPAudioStudio",
    category: "LocalDatasetEnrichment.memory"
)

enum LocalDatasetEnrichmentStep: String, Equatable {
    case preparingMemory
    case separatingStems
    case transcribingVocals
    case generatingCaptions
    case loadingNativeAnnotation
    case captioningFullMixes
    case transcribingFullMixes
    case exportingVocalsBundle
    case exportingInstrumentalBundle
    case refreshingMetadata

    var title: String {
        switch self {
        case .preparingMemory:
            return "Preparing memory"
        case .separatingStems:
            return "Separating stems"
        case .transcribingVocals:
            return "Transcribing vocals"
        case .generatingCaptions:
            return "Generating metadata captions"
        case .loadingNativeAnnotation:
            return "Loading native ACE annotation"
        case .captioningFullMixes:
            return "Captioning full mixes"
        case .transcribingFullMixes:
            return "Transcribing full mixes"
        case .exportingVocalsBundle:
            return "Exporting vocal bundle"
        case .exportingInstrumentalBundle:
            return "Exporting instrumental bundle"
        case .refreshingMetadata:
            return "Refreshing package metadata"
        }
    }
}

enum LocalDatasetEnrichmentPhase: Equatable {
    case idle
    case running(step: LocalDatasetEnrichmentStep)
    case complete(LocalDatasetEnrichmentResult)
    case failed(String)

    var isRunning: Bool {
        if case .running = self { return true }
        return false
    }
}

struct LocalDatasetEnrichmentResult: Equatable {
    let datasetDirectory: URL
    let separatedClipCount: Int
    let vocalBundleExists: Bool
    let instrumentalBundleExists: Bool
    let bpmEstimateCount: Int
    let averageBPM: Double?
    let keyEstimateCount: Int
    let transcribedClipCount: Int
    let autoCaptionedClipCount: Int

    var summary: String {
        var parts: [String] = []
        if separatedClipCount > 0 {
            parts.append(
                "Separated \(separatedClipCount) clip\(separatedClipCount == 1 ? "" : "s")"
            )
        }
        if vocalBundleExists && instrumentalBundleExists {
            parts.append("wrote vocal and instrumental bundles")
        } else if vocalBundleExists {
            parts.append("wrote vocal bundle")
        } else if instrumentalBundleExists {
            parts.append("wrote instrumental bundle")
        }
        if let averageBPM, bpmEstimateCount > 0 {
            let formatted = averageBPM.formatted(.number.precision(.fractionLength(1)))
            parts.append("\(bpmEstimateCount) BPM estimate\(bpmEstimateCount == 1 ? "" : "s"), avg \(formatted)")
        }
        parts.append(transcribedClipCount > 0 ? "\(transcribedClipCount) transcript\(transcribedClipCount == 1 ? "" : "s") found" : "transcription pending")
        parts.append(keyEstimateCount > 0 ? "\(keyEstimateCount) key estimate\(keyEstimateCount == 1 ? "" : "s")" : "key analysis pending")
        if autoCaptionedClipCount > 0 {
            parts.append("\(autoCaptionedClipCount) auto caption\(autoCaptionedClipCount == 1 ? "" : "s")")
        }
        return parts.joined(separator: "; ") + "."
    }
}

enum LocalDatasetAnnotationBackendChoice: String, CaseIterable, Hashable, Identifiable {
    case automatic
    case lightweight
    case nativeACE

    var id: String { rawValue }

    var title: String {
        switch self {
        case .automatic:
            return "Automatic"
        case .lightweight:
            return "Lightweight Native"
        case .nativeACE:
            return "Native ACE (Advanced)"
        }
    }
}

struct LocalDatasetEnrichmentOptions: Equatable {
    var createStems = false
    var annotationBackend: LocalDatasetAnnotationBackendChoice = .automatic
}

enum LocalDatasetEnrichmentError: LocalizedError {
    case invalidCaptions(String)
    case missingClip(String)
    case nativeACEUnavailable(String)

    var errorDescription: String? {
        switch self {
        case .invalidCaptions(let message):
            return "Dataset captions are not valid: \(message)"
        case .missingClip(let path):
            return "Dataset clip is missing: \(path)"
        case .nativeACEUnavailable(let reason):
            return "Native ACE annotation is unavailable: \(reason)"
        }
    }
}

@MainActor
final class LocalDatasetEnrichmentService: ObservableObject {
    @Published var phase: LocalDatasetEnrichmentPhase = .idle

    private var task: Task<Void, Never>?
    private var taskRunID: UUID?
    private var activeRunID: UUID?

    func enrich(
        datasetDirectory: URL,
        force: Bool = false,
        options: LocalDatasetEnrichmentOptions = LocalDatasetEnrichmentOptions(),
        prepareMemory: (() async -> Void)? = nil
    ) {
        let previousTask = task
        previousTask?.cancel()
        let runID = UUID()
        taskRunID = runID
        activeRunID = runID
        let initialStep: LocalDatasetEnrichmentStep
        if prepareMemory != nil {
            initialStep = .preparingMemory
        } else if options.createStems {
            initialStep = .separatingStems
        } else if options.annotationBackend == .lightweight {
            initialStep = .transcribingFullMixes
        } else {
            initialStep = .loadingNativeAnnotation
        }
        phase = .running(step: initialStep)

        task = Task { [weak self, previousTask] in
            await previousTask?.value
            guard let self else { return }
            defer { self.finishRun(runID) }
            guard self.isCurrentRun(runID), !Task.isCancelled else { return }
            do {
                if let prepareMemory {
                    await MainActor.run {
                        guard self.isCurrentRun(runID) else { return }
                        self.phase = .running(step: .preparingMemory)
                    }
                    await prepareMemory()
                    try Task.checkCancellation()
                    guard self.isCurrentRun(runID) else { return }
                }
                let result = try await Self.run(
                    datasetDirectory: datasetDirectory,
                    force: force,
                    options: options,
                    onStep: { [self] step in
                        await MainActor.run {
                            guard self.isCurrentRun(runID) else { return }
                            self.phase = .running(step: step)
                        }
                    }
                )
                guard self.isCurrentRun(runID) else { return }
                phase = .complete(result)
            } catch is CancellationError {
                guard self.isCurrentRun(runID) else { return }
                phase = .idle
            } catch {
                guard self.isCurrentRun(runID) else { return }
                phase = .failed(error.localizedDescription)
            }
        }
    }

    func cancel() {
        task?.cancel()
        activeRunID = nil
        phase = .idle
    }

    func reset() {
        cancel()
        phase = .idle
    }

    private func isCurrentRun(_ runID: UUID) -> Bool {
        activeRunID == runID
    }

    private func finishRun(_ runID: UUID) {
        if activeRunID == runID {
            activeRunID = nil
        }
        if taskRunID == runID {
            taskRunID = nil
            task = nil
        }
    }

    nonisolated static func run(
        datasetDirectory: URL,
        force: Bool = false,
        options: LocalDatasetEnrichmentOptions = LocalDatasetEnrichmentOptions(),
        lyricsTranscriber: any DatasetLyricsTranscribing = LyricsTranscriptionService(),
        annotationBackend: (any DatasetAnnotationBackend)? = nil,
        onStep: @Sendable @escaping (LocalDatasetEnrichmentStep) async -> Void
    ) async throws -> LocalDatasetEnrichmentResult {
        let didStartAccessing = datasetDirectory.startAccessingSecurityScopedResource()
        defer {
            if didStartAccessing {
                datasetDirectory.stopAccessingSecurityScopedResource()
            }
        }

        do {
            var records = try loadCaptionRecords(from: datasetDirectory)

            if options.createStems {
                let sourceSeparator = SourceSeparationService()
                await onStep(.separatingStems)
                for index in records.indices {
                    try Task.checkCancellation()
                    if !force, hasCachedInstrumentalSeparation(
                        record: records[index].object,
                        datasetDirectory: datasetDirectory
                    ) {
                        continue
                    }
                    records[index].object = try await separateClip(
                        record: records[index],
                        datasetDirectory: datasetDirectory,
                        sourceSeparator: sourceSeparator,
                        force: force
                    )
                    compactMLXCache(
                        context: "after separation \(index + 1)/\(records.count)"
                    )
                }
                compactMLXCache(context: "before annotation")
            }

            let nativeBackend = try await nativeAnnotationBackend(
                preference: options.annotationBackend,
                suppliedBackend: annotationBackend
            )
            let usedNativeAnnotationBackend = nativeBackend != nil
            if let nativeBackend {
                records = try await annotateWithNativeACE(
                    records: records,
                    datasetDirectory: datasetDirectory,
                    backend: nativeBackend,
                    lyricsAligner: lyricsTranscriber
                        as? any DatasetLyricsAligning,
                    force: force,
                    onStep: onStep
                )
            } else {
                await onStep(
                    options.createStems
                        ? .transcribingVocals
                        : .transcribingFullMixes
                )
                for index in records.indices {
                    try Task.checkCancellation()
                    records[index].object = try await transcribeClipVocals(
                        record: records[index],
                        datasetDirectory: datasetDirectory,
                        lyricsTranscriber: lyricsTranscriber,
                        preferVocals: options.createStems,
                        force: force
                    )
                    compactMLXCache(
                        context: "after transcription \(index + 1)/\(records.count)"
                    )
                }

                if options.createStems {
                    for index in records.indices {
                        try Task.checkCancellation()
                        records[index].object = try applyDerivedAudioMetadata(
                            record: records[index],
                            datasetDirectory: datasetDirectory,
                            force: force
                        )
                    }
                }

                await onStep(.generatingCaptions)
                for index in records.indices {
                    try Task.checkCancellation()
                    records[index].object = try applyAutomaticCaptionMetadata(
                        record: records[index],
                        datasetDirectory: datasetDirectory,
                        force: force
                    )
                }
            }

            if usedNativeAnnotationBackend, options.createStems {
                for index in records.indices {
                    try Task.checkCancellation()
                    records[index].object = try applyDerivedAudioMetadata(
                        record: records[index],
                        datasetDirectory: datasetDirectory,
                        force: force
                    )
                }
            }

            try writeJSONObjectArray(
                records.map(\.object),
                to: datasetDirectory.appendingPathComponent(TrainingDatasetPackageWriter.captionsFileName)
            )
            try TrainingDatasetPackageWriter.refreshTrainingBundles(at: datasetDirectory)

            if options.createStems {
                await onStep(.exportingVocalsBundle)
                _ = try writeStemTrainingBundle(
                    datasetDirectory: datasetDirectory,
                    records: records.map(\.object),
                    role: .vocals
                )

                await onStep(.exportingInstrumentalBundle)
                _ = try writeStemTrainingBundle(
                    datasetDirectory: datasetDirectory,
                    records: records.map(\.object),
                    role: .instrumental
                )
            }

            try Task.checkCancellation()
            await onStep(.refreshingMetadata)
            let result = inspectResult(datasetDirectory: datasetDirectory)
            try refreshDatasetMetadata(datasetDirectory: datasetDirectory, result: result)
            try TrainingDatasetPackageWriter.refreshValidationReport(at: datasetDirectory)
            let finalResult = inspectResult(datasetDirectory: datasetDirectory)
            await evictEnrichmentModels(lyricsTranscriber: lyricsTranscriber)
            return finalResult
        } catch {
            await evictEnrichmentModels(lyricsTranscriber: lyricsTranscriber)
            throw error
        }
    }

    nonisolated private static func evictEnrichmentModels(
        lyricsTranscriber: any DatasetLyricsTranscribing
    ) async {
        let before = Memory.snapshot()
        enrichmentMemoryLogger.info("[enrich.evict] before active=\(before.activeMemory) cache=\(before.cacheMemory) peak=\(before.peakMemory)")

        if let evictingTranscriber = lyricsTranscriber as? any DatasetLyricsTranscriptionEvicting {
            await evictingTranscriber.evictModels()
        }
        Memory.clearCache()

        let after = Memory.snapshot()
        enrichmentMemoryLogger.info("[enrich.evict] after  active=\(after.activeMemory) cache=\(after.cacheMemory) peak=\(after.peakMemory)")
        enrichmentMemoryLogger.info("[enrich.evict] delta active=-\(before.activeMemory - after.activeMemory) cache=-\(before.cacheMemory - after.cacheMemory)")
    }

    nonisolated private static func compactMLXCache(context: String) {
        let before = Memory.snapshot()
        Memory.clearCache()
        let after = Memory.snapshot()
        enrichmentMemoryLogger.info(
            "[enrich.compact] \(context) active=\(after.activeMemory) cache=\(after.cacheMemory) releasedCache=\(max(0, before.cacheMemory - after.cacheMemory)) peak=\(after.peakMemory)"
        )
    }

    @discardableResult
    nonisolated static func exportStemTrainingBundles(datasetDirectory: URL) throws -> (vocalCount: Int, instrumentalCount: Int) {
        let records = try loadCaptionRecords(from: datasetDirectory).map(\.object)
        let vocalCount = try writeStemTrainingBundle(
            datasetDirectory: datasetDirectory,
            records: records,
            role: .vocals
        )
        let instrumentalCount = try writeStemTrainingBundle(
            datasetDirectory: datasetDirectory,
            records: records,
            role: .instrumental
        )
        try TrainingDatasetPackageWriter.refreshTrainingBundles(at: datasetDirectory)
        try TrainingDatasetPackageWriter.refreshPreprocessCacheManifest(at: datasetDirectory)
        return (vocalCount, instrumentalCount)
    }

    nonisolated private static func nativeAnnotationBackend(
        preference: LocalDatasetAnnotationBackendChoice,
        suppliedBackend: (any DatasetAnnotationBackend)?
    ) async throws -> (any DatasetAnnotationBackend)? {
        guard preference != .lightweight else { return nil }
        let backend: any DatasetAnnotationBackend = suppliedBackend
            ?? NativeACEDatasetAnnotationBackend(
                modelsDirectory: AppStorageLocations.modelsDirectory
            )
        let availability = await backend.availability()

        if availability.hardwareTier == .unsupported {
            if preference == .nativeACE {
                throw LocalDatasetEnrichmentError.nativeACEUnavailable(
                    "this Mac has less than the supported 24 GB unified-memory minimum"
                )
            }
            return nil
        }
        guard availability.isFullyInstalled else {
            if preference == .nativeACE {
                let reasons = availability.models
                    .filter { !$0.isInstalled }
                    .compactMap(\.unavailableReason)
                    .joined(separator: " ")
                throw LocalDatasetEnrichmentError.nativeACEUnavailable(
                    reasons.isEmpty
                        ? "the captioner and transcriber model packs are not installed"
                        : reasons
                )
            }
            return nil
        }
        return backend
    }

    nonisolated private static func annotateWithNativeACE(
        records: [LocalDatasetCaptionRecord],
        datasetDirectory: URL,
        backend: any DatasetAnnotationBackend,
        lyricsAligner: (any DatasetLyricsAligning)?,
        force: Bool,
        onStep: @Sendable @escaping (LocalDatasetEnrichmentStep) async -> Void
    ) async throws -> [LocalDatasetCaptionRecord] {
        var records = records
        let indices = annotationRecordIndices(in: records)
        guard !indices.isEmpty else { return records }

        let captionIndices = indices.filter {
            force || !hasCompletedNativeAnnotation(
                task: .caption,
                record: records[$0].object,
                datasetDirectory: datasetDirectory
            )
        }
        if !captionIndices.isEmpty {
            await onStep(.loadingNativeAnnotation)
            let requests = try annotationRequests(
                indices: captionIndices,
                records: records,
                datasetDirectory: datasetDirectory
            )
            let results = try await backend.annotate(
                requests: requests,
                task: .caption,
                options: DatasetAnnotationGenerationOptions(maxTokens: 220),
                onProgress: { progress in
                    guard progress.stage == .annotating else { return }
                    Task {
                        await onStep(.captioningFullMixes)
                    }
                }
            )
            let indexByID = Dictionary(
                uniqueKeysWithValues: captionIndices.map {
                    (annotationRequestID(for: records[$0], index: $0), $0)
                }
            )
            for result in results {
                try Task.checkCancellation()
                guard let index = indexByID[result.requestID] else { continue }
                records[index].object = try applyNativeAnnotationResult(
                    result,
                    to: records[index],
                    datasetDirectory: datasetDirectory
                )
            }
            compactMLXCache(context: "after native ACE captioning")
        }

        let transcriptionIndices = indices.filter {
            force || !hasCompletedNativeAnnotation(
                task: .transcribe,
                record: records[$0].object,
                datasetDirectory: datasetDirectory
            )
        }
        if !transcriptionIndices.isEmpty {
            await onStep(.loadingNativeAnnotation)
            let requests = try annotationRequests(
                indices: transcriptionIndices,
                records: records,
                datasetDirectory: datasetDirectory
            )
            let results = try await backend.annotate(
                requests: requests,
                task: .transcribe,
                options: DatasetAnnotationGenerationOptions(maxTokens: 1_024),
                onProgress: { progress in
                    guard progress.stage == .annotating else { return }
                    Task {
                        await onStep(.transcribingFullMixes)
                    }
                }
            )
            let indexByID = Dictionary(
                uniqueKeysWithValues: transcriptionIndices.map {
                    (annotationRequestID(for: records[$0], index: $0), $0)
                }
            )
            for result in results {
                try Task.checkCancellation()
                guard let index = indexByID[result.requestID] else { continue }
                records[index].object = try applyNativeAnnotationResult(
                    result,
                    to: records[index],
                    datasetDirectory: datasetDirectory
                )
            }
            if let lyricsAligner {
                for index in transcriptionIndices {
                    try Task.checkCancellation()
                    records[index].object = try await alignNativeLyrics(
                        record: records[index],
                        datasetDirectory: datasetDirectory,
                        lyricsAligner: lyricsAligner,
                        force: force
                    )
                }
            }
            compactMLXCache(context: "after native ACE transcription")
        }

        await backend.evict()
        return records
    }

    nonisolated private static func annotationRecordIndices(
        in records: [LocalDatasetCaptionRecord]
    ) -> [Int] {
        let fullMixes = records.indices.filter {
            let role = stringValue(records[$0].object["asset_role"])
                ?? stringValue(records[$0].object["role"])
            return role == DatasetAssetRoleV2.fullMix.rawValue
        }
        return fullMixes.isEmpty ? Array(records.indices) : fullMixes
    }

    nonisolated private static func annotationRequests(
        indices: [Int],
        records: [LocalDatasetCaptionRecord],
        datasetDirectory: URL
    ) throws -> [DatasetAnnotationRequest] {
        try indices.map { index in
            let record = records[index]
            let audioURL = resolve(record.file, relativeTo: datasetDirectory)
            guard FileManager.default.fileExists(
                atPath: audioURL.path(percentEncoded: false)
            ) else {
                throw LocalDatasetEnrichmentError.missingClip(record.file)
            }
            return DatasetAnnotationRequest(
                id: annotationRequestID(for: record, index: index),
                audioURL: audioURL
            )
        }
    }

    nonisolated private static func annotationRequestID(
        for record: LocalDatasetCaptionRecord,
        index: Int
    ) -> String {
        stringValue(record.object["asset_id"])
            ?? stringValue(record.object["id"])
            ?? "asset-\(index + 1)"
    }

    nonisolated private static func hasCompletedNativeAnnotation(
        task: DatasetAnnotationTask,
        record: [String: Any],
        datasetDirectory: URL
    ) -> Bool {
        let annotation = record["annotation"] as? [String: Any] ?? [:]
        let rawOutputs = annotation["raw_output_files"] as? [String: Any] ?? [:]
        let key = task == .caption ? "caption" : "lyrics"
        guard let rawPath = stringValue(rawOutputs[key]),
              FileManager.default.fileExists(
                atPath: resolve(
                    rawPath,
                    relativeTo: datasetDirectory
                ).path(percentEncoded: false)
              )
        else {
            return false
        }
        switch task {
        case .caption:
            return stringValue(record["caption_source"]) == "ace-step-captioner"
        case .transcribe:
            return stringValue(record["lyrics_source"]) == "ace-step-transcriber"
        }
    }

    nonisolated private static func applyNativeAnnotationResult(
        _ result: DatasetAnnotationBackendResult,
        to record: LocalDatasetCaptionRecord,
        datasetDirectory: URL
    ) throws -> [String: Any] {
        let artifactDirectory = annotationArtifactDirectory(
            for: record,
            datasetDirectory: datasetDirectory
        ).appendingPathComponent("native-ace", isDirectory: true)
        try FileManager.default.createDirectory(
            at: artifactDirectory,
            withIntermediateDirectories: true
        )
        let taskName = result.task == .caption ? "caption" : "lyrics"
        let rawURL = artifactDirectory.appendingPathComponent(
            "\(taskName).raw.txt"
        )
        try result.rawText.write(
            to: rawURL,
            atomically: true,
            encoding: .utf8
        )
        let resultURL = artifactDirectory.appendingPathComponent(
            "\(taskName).result.json"
        )
        let resultPayload: [String: Any] = [
            "schema_version": "1.0",
            "generated_at": ISO8601DateFormatter().string(from: Date()),
            "task": result.task.rawValue,
            "backend_id": result.descriptor.id,
            "backend_version": result.descriptor.version,
            "model_id": result.descriptor.modelID,
            "model_revision": result.descriptor.modelRevision,
            "prompt_version": result.descriptor.promptVersion,
            "quantization": result.descriptor.quantization,
            "generation_settings": result.settings,
            "elapsed_seconds": rounded(result.elapsedSeconds, places: 3),
            "prompt_token_count": result.promptTokenCount,
            "generated_token_count": result.generatedTokenCount,
            "language": result.language ?? "",
            "text": result.text,
            "raw_text": result.rawText,
            "diagnostics": [
                "is_empty": result.diagnostics.isEmpty,
                "line_count": result.diagnostics.lineCount,
                "unique_line_count": result.diagnostics.uniqueLineCount,
                "maximum_repeated_line_count":
                    result.diagnostics.maximumRepeatedLineCount,
                "pathological_repetition":
                    result.diagnostics.isPathologicalRepetition,
            ],
        ]
        try writeJSONObject(resultPayload, to: resultURL)

        let rawRelative = relativePath(for: rawURL, from: datasetDirectory)
        let resultRelative = relativePath(
            for: resultURL,
            from: datasetDirectory
        )
        let updatedAt = ISO8601DateFormatter().string(from: Date())
        var updated = record.object
        var annotation = updated["annotation"] as? [String: Any] ?? [:]
        var rawOutputs = annotation["raw_output_files"] as? [String: Any] ?? [:]
        rawOutputs[taskName] = rawRelative
        rawOutputs["\(taskName)_result"] = resultRelative
        annotation["schema_version"] =
            TrainingDatasetPackageWriter.annotationSchemaVersion
        annotation["updated_at"] = updatedAt
        annotation["review_status"] = "needs_review"
        annotation["raw_output_files"] = rawOutputs

        switch result.task {
        case .caption:
            updated["caption"] = result.text
            updated["prompt"] = result.text
            updated["caption_source"] = "ace-step-captioner"
            annotation["annotation_backend"] = result.descriptor.id
            annotation["annotation_backend_version"] =
                result.descriptor.version
            annotation["caption_source"] = "ace-step-captioner"
            annotation["caption_present"] = !result.text.isEmpty
            annotation["model_id"] = result.descriptor.modelID
            annotation["model_revision"] = result.descriptor.modelRevision
            annotation["prompt_version"] = result.descriptor.promptVersion
            annotation["quantization"] = result.descriptor.quantization
            annotation["generation_settings"] = result.settings
            annotation["elapsed_seconds"] = rounded(
                result.elapsedSeconds,
                places: 3
            )
        case .transcribe:
            updated["lyrics"] = result.text
            updated["transcript"] = result.text
            updated["lyrics_source"] = "ace-step-transcriber"
            updated["language"] = result.language ?? ""
            annotation["lyrics_source"] = "ace-step-transcriber"
            annotation["lyrics_present"] = !result.text.isEmpty
            annotation["language"] = result.language ?? ""
            updated["transcription"] = [
                "status": result.text.isEmpty ? "empty" : "complete",
                "text": result.text,
                "source": "ace-step-transcriber",
                "backend": result.descriptor.id,
                "backend_version": result.descriptor.version,
                "model": result.descriptor.modelID,
                "model_revision": result.descriptor.modelRevision,
                "prompt_version": result.descriptor.promptVersion,
                "quantization": result.descriptor.quantization,
                "generation_settings": result.settings,
                "elapsed_seconds": rounded(
                    result.elapsedSeconds,
                    places: 3
                ),
                "language": result.language ?? "",
                "lyrics_file": rawRelative,
                "raw_result_file": resultRelative,
            ]
        }
        updated["annotation"] = annotation

        let audioURL = resolve(record.file, relativeTo: datasetDirectory)
        let sidecarURL = audioURL.deletingPathExtension()
            .appendingPathExtension("json")
        var sidecar = readJSONObject(from: sidecarURL) ?? [:]
        sidecar["caption"] = updated["caption"]
        sidecar["prompt"] = updated["prompt"]
        sidecar["lyrics"] = updated["lyrics"]
        sidecar["transcript"] = updated["transcript"]
        sidecar["caption_source"] = updated["caption_source"]
        sidecar["lyrics_source"] = updated["lyrics_source"]
        sidecar["language"] = updated["language"]
        sidecar["annotation"] = annotation
        sidecar["transcription"] = updated["transcription"]
        try writeJSONObject(sidecar, to: sidecarURL)
        return updated
    }

    nonisolated private static func annotationArtifactDirectory(
        for record: LocalDatasetCaptionRecord,
        datasetDirectory: URL
    ) -> URL {
        let identifier = stringValue(record.object["asset_id"])
            ?? stringValue(record.object["id"])
            ?? URL(fileURLWithPath: record.file)
                .deletingPathExtension()
                .lastPathComponent
        let safeIdentifier = identifier
            .lowercased()
            .map { character -> Character in
                character.isLetter || character.isNumber ? character : "-"
            }
            .reduce(into: "") { value, character in
                if character != "-" || !value.hasSuffix("-") {
                    value.append(character)
                }
            }
            .trimmingCharacters(in: CharacterSet(charactersIn: "-"))
        return datasetDirectory
            .appendingPathComponent("annotations", isDirectory: true)
            .appendingPathComponent(
                safeIdentifier.isEmpty ? "asset" : safeIdentifier,
                isDirectory: true
            )
    }

    nonisolated private static func alignNativeLyrics(
        record: LocalDatasetCaptionRecord,
        datasetDirectory: URL,
        lyricsAligner: any DatasetLyricsAligning,
        force: Bool
    ) async throws -> [String: Any] {
        let lyrics = stringValue(record.object["lyrics"])
            ?? stringValue(record.object["transcript"])
            ?? ""
        guard !lyrics.isEmpty else { return record.object }

        let audioURL = resolve(record.file, relativeTo: datasetDirectory)
        let outputDirectory = annotationArtifactDirectory(
            for: record,
            datasetDirectory: datasetDirectory
        ).appendingPathComponent("native-ace", isDirectory: true)
        let alignment = try await lyricsAligner.align(
            lyrics: lyrics,
            audioURL: audioURL,
            outputDirectory: outputDirectory,
            force: force
        )

        var updated = record.object
        var transcription = updated["transcription"] as? [String: Any] ?? [:]
        transcription["aligner_backend"] = alignment.backend
        transcription["aligner_model"] = alignment.alignerModel ?? ""
        transcription["alignment_elapsed_seconds"] = rounded(
            alignment.elapsedSeconds,
            places: 3
        )
        transcription["lyrics_file"] = relativePath(
            for: alignment.lyricsURL,
            from: datasetDirectory
        )
        transcription["timed_lyrics_file"] = relativePath(
            for: alignment.timedLyricsURL,
            from: datasetDirectory
        )
        transcription["lrc_file"] = relativePath(
            for: alignment.lrcURL,
            from: datasetDirectory
        )
        updated["transcription"] = transcription

        var assets = updated["assets"] as? [String: Any] ?? [:]
        assets["lyrics"] = transcription["lyrics_file"]
        assets["timed_lyrics"] = transcription["timed_lyrics_file"]
        assets["lyrics_lrc"] = transcription["lrc_file"]
        updated["assets"] = assets

        var annotation = updated["annotation"] as? [String: Any] ?? [:]
        var rawOutputs = annotation["raw_output_files"] as? [String: Any] ?? [:]
        rawOutputs["lyrics_text"] = transcription["lyrics_file"]
        rawOutputs["timed_lyrics"] = transcription["timed_lyrics_file"]
        rawOutputs["lyrics_lrc"] = transcription["lrc_file"]
        annotation["raw_output_files"] = rawOutputs
        updated["annotation"] = annotation

        let sidecarURL = audioURL.deletingPathExtension()
            .appendingPathExtension("json")
        var sidecar = readJSONObject(from: sidecarURL) ?? [:]
        sidecar["transcription"] = transcription
        sidecar["annotation"] = annotation
        try writeJSONObject(sidecar, to: sidecarURL)
        return updated
    }

    nonisolated private static func separateClip(
        record: LocalDatasetCaptionRecord,
        datasetDirectory: URL,
        sourceSeparator: SourceSeparationService,
        force: Bool
    ) async throws -> [String: Any] {
        let clipURL = resolve(record.file, relativeTo: datasetDirectory)
        guard FileManager.default.fileExists(atPath: clipURL.path(percentEncoded: false)) else {
            throw LocalDatasetEnrichmentError.missingClip(record.file)
        }

        let stemName = clipURL.deletingPathExtension().lastPathComponent
        let outputDirectory = datasetDirectory
            .appendingPathComponent("stems", isDirectory: true)
            .appendingPathComponent(stemName, isDirectory: true)
        if force, FileManager.default.fileExists(atPath: outputDirectory.path) {
            try FileManager.default.removeItem(at: outputDirectory)
        }
        try FileManager.default.createDirectory(at: outputDirectory, withIntermediateDirectories: true)

        let input = try separatorInputURL(
            clipURL: clipURL,
            outputDirectory: outputDirectory,
            force: force
        )

        let startedAt = Date()
        let separation = try await sourceSeparator.separate(
            inputURL: input.url,
            outputDirectory: outputDirectory,
            force: force
        )
        let elapsed = Date().timeIntervalSince(startedAt)
        let warnings = (input.warning.map { [$0] } ?? []) + separation.warnings
        let payload = separationPayload(
            datasetDirectory: datasetDirectory,
            sourceClip: record.file,
            backend: separation.backend,
            backendVersion: separation.backendVersion,
            model: separation.model,
            elapsedSeconds: elapsed,
            stems: separation.stems,
            warnings: warnings
        )
        let separationURL = outputDirectory.appendingPathComponent("separation.json")
        try writeJSONObject(payload, to: separationURL)

        var sidecar = readJSONObject(from: clipURL.deletingPathExtension().appendingPathExtension("json")) ?? [:]
        sidecar["separation"] = payload
        try writeJSONObject(sidecar, to: clipURL.deletingPathExtension().appendingPathExtension("json"))

        var updated = record.object
        let stemPaths = separation.stems.mapValues { relativePath(for: $0, from: datasetDirectory) }
        updated["separation"] = [
            "mode": "instrumental",
            "backend": separation.backend,
            "model": separation.model,
            "stems": stemPaths,
            "separation_file": relativePath(for: separationURL, from: datasetDirectory),
            "cached": false,
            "warnings": warnings,
        ]
        var assets = updated["assets"] as? [String: Any] ?? [:]
        assets["full_mix"] = record.file
        assets["vocals"] = stemPaths["vocals"]
        assets["instrumental"] = stemPaths["instrumental"]
        updated["assets"] = assets
        return updated
    }

    nonisolated private static func transcribeClipVocals(
        record: LocalDatasetCaptionRecord,
        datasetDirectory: URL,
        lyricsTranscriber: any DatasetLyricsTranscribing,
        preferVocals: Bool,
        force: Bool
    ) async throws -> [String: Any] {
        let audioPath = preferVocals
            ? stemPath(for: .vocals, in: record.object)
            : record.file
        guard let audioPath else { return record.object }
        if !force, hasCompletedTranscription(record.object) {
            return record.object
        }

        let audioURL = resolve(audioPath, relativeTo: datasetDirectory)
        guard FileManager.default.fileExists(atPath: audioURL.path(percentEncoded: false)) else {
            throw LocalDatasetEnrichmentError.missingClip(audioPath)
        }

        let outputDirectory = preferVocals
            ? audioURL.deletingLastPathComponent()
            : annotationArtifactDirectory(
                for: record,
                datasetDirectory: datasetDirectory
            ).appendingPathComponent("lightweight", isDirectory: true)
        let transcription = try await lyricsTranscriber.transcribe(
            vocalsURL: audioURL,
            outputDirectory: outputDirectory,
            force: force
        )

        var payload: [String: Any] = [
            "status": transcription.text.isEmpty ? "empty" : "complete",
            "text": transcription.text,
            "source": transcription.source,
            "backend": transcription.backend,
            "model": transcription.model,
            "word_count": transcription.wordCount,
            "elapsed_seconds": rounded(transcription.elapsedSeconds, places: 3),
            "lyrics_file": relativePath(for: transcription.lyricsURL, from: datasetDirectory),
            "timed_lyrics_file": relativePath(for: transcription.timedLyricsURL, from: datasetDirectory),
            "lrc_file": relativePath(for: transcription.lrcURL, from: datasetDirectory),
        ]
        if let alignerModel = transcription.alignerModel {
            payload["aligner_model"] = alignerModel
        }

        var updated = record.object
        updated["lyrics"] = transcription.text
        updated["transcript"] = transcription.text
        updated["transcription"] = payload
        updated["vocal_overlap"] = !transcription.text.isEmpty

        var assets = updated["assets"] as? [String: Any] ?? [:]
        assets["lyrics"] = relativePath(for: transcription.lyricsURL, from: datasetDirectory)
        assets["timed_lyrics"] = relativePath(for: transcription.timedLyricsURL, from: datasetDirectory)
        assets["lyrics_lrc"] = relativePath(for: transcription.lrcURL, from: datasetDirectory)
        updated["assets"] = assets

        let lyricsSource = transcription.source == "qwen3-asr" ? "native-qwen3-asr" : transcription.source
        updated["lyrics_source"] = transcription.text.isEmpty ? "none" : lyricsSource
        if updated["caption_source"] == nil {
            updated["caption_source"] = promptText(from: updated).isEmpty ? "none" : "manual"
        }

        var rawOutputs = (updated["annotation"] as? [String: Any])?["raw_output_files"] as? [String: Any] ?? [:]
        rawOutputs["lyrics"] = relativePath(for: transcription.lyricsURL, from: datasetDirectory)
        rawOutputs["timed_lyrics"] = relativePath(for: transcription.timedLyricsURL, from: datasetDirectory)
        rawOutputs["lyrics_lrc"] = relativePath(for: transcription.lrcURL, from: datasetDirectory)
        var annotation = updated["annotation"] as? [String: Any] ?? [:]
        annotation["schema_version"] = TrainingDatasetPackageWriter.annotationSchemaVersion
        annotation["updated_at"] = ISO8601DateFormatter().string(from: Date())
        annotation["annotation_backend"] = transcription.backend
        annotation["caption_source"] = updated["caption_source"] as? String ?? "manual"
        annotation["lyrics_source"] = updated["lyrics_source"] as? String ?? "none"
        annotation["model_id"] = transcription.model
        annotation["model_revision"] = annotation["model_revision"] ?? ""
        annotation["aligner_model"] = transcription.alignerModel ?? ""
        annotation["language"] = annotation["language"] ?? ""
        annotation["caption_present"] = !promptText(from: updated).isEmpty
        annotation["lyrics_present"] = !transcription.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        annotation["review_status"] = annotation["review_status"] ?? "needs_review"
        annotation["raw_output_files"] = rawOutputs
        updated["annotation"] = annotation

        let clipURL = resolve(record.file, relativeTo: datasetDirectory)
        var sidecar = readJSONObject(from: clipURL.deletingPathExtension().appendingPathExtension("json")) ?? [:]
        sidecar["transcription"] = payload
        sidecar["annotation"] = annotation
        try writeJSONObject(sidecar, to: clipURL.deletingPathExtension().appendingPathExtension("json"))

        return updated
    }

    nonisolated private static func separatorInputURL(
        clipURL: URL,
        outputDirectory: URL,
        force: Bool
    ) throws -> (url: URL, warning: String?) {
        guard clipURL.pathExtension.lowercased() != "wav" else {
            return (clipURL, nil)
        }

        let normalizedURL = outputDirectory.appendingPathComponent("source.wav")
        if FileManager.default.fileExists(atPath: normalizedURL.path), !force {
            return (normalizedURL, "normalized non-WAV source for separation: \(clipURL.lastPathComponent)")
        }
        try writeWAVCopy(from: clipURL, to: normalizedURL)
        return (normalizedURL, "normalized non-WAV source for separation: \(clipURL.lastPathComponent)")
    }

    nonisolated private static func writeWAVCopy(from sourceURL: URL, to destinationURL: URL) throws {
        if FileManager.default.fileExists(atPath: destinationURL.path) {
            try FileManager.default.removeItem(at: destinationURL)
        }
        let inputFile = try AVAudioFile(forReading: sourceURL)
        let format = inputFile.processingFormat
        let outputFile = try AVAudioFile(forWriting: destinationURL, settings: format.settings)
        inputFile.framePosition = 0

        var remainingFrames = inputFile.length
        let maxBufferFrames: AVAudioFrameCount = 262_144
        while remainingFrames > 0 {
            try Task.checkCancellation()
            let framesToRead = AVAudioFrameCount(min(Int64(maxBufferFrames), remainingFrames))
            guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: framesToRead) else {
                throw LocalFilesDatasetError.audioBufferCreationFailed
            }
            try inputFile.read(into: buffer, frameCount: framesToRead)
            guard buffer.frameLength > 0 else { break }
            try outputFile.write(from: buffer)
            remainingFrames -= AVAudioFramePosition(buffer.frameLength)
        }
    }

    nonisolated private static func separationPayload(
        datasetDirectory: URL,
        sourceClip: String,
        backend: String,
        backendVersion: String,
        model: String,
        elapsedSeconds: TimeInterval,
        stems: [String: URL],
        warnings: [String]
    ) -> [String: Any] {
        [
            "pimp_audio_dataset_version": "1.0",
            "generated_at": ISO8601DateFormatter().string(from: Date()),
            "backend": backend,
            "backend_version": backendVersion,
            "model": model,
            "mode": "instrumental",
            "source_clip": sourceClip,
            "elapsed_seconds": rounded(elapsedSeconds, places: 3),
            "cached": false,
            "warnings": warnings,
            "stems": stems.reduce(into: [String: Any]()) { result, item in
                result[item.key] = [
                    "file": relativePath(for: item.value, from: datasetDirectory),
                    "analysis": analyzeAudioFile(item.value, role: item.key),
                ]
            },
        ]
    }

    nonisolated private static func applyDerivedAudioMetadata(
        record: LocalDatasetCaptionRecord,
        datasetDirectory: URL,
        force: Bool
    ) throws -> [String: Any] {
        let refreshedAnalyses = try refreshCachedStemAnalysisIfNeeded(
            record: record.object,
            datasetDirectory: datasetDirectory,
            force: force
        )
        let instrumentalAnalysis = refreshedAnalyses[.instrumental]
            ?? stemAnalysis(for: .instrumental, in: record.object, datasetDirectory: datasetDirectory)
        let vocalAnalysis = refreshedAnalyses[.vocals]
            ?? stemAnalysis(for: .vocals, in: record.object, datasetDirectory: datasetDirectory)

        var updated = record.object
        var analysis = updated["analysis"] as? [String: Any] ?? [:]

        if missingBPM(in: updated, analysis: analysis),
           let bpm = preferredBPM(instrumental: instrumentalAnalysis, vocal: vocalAnalysis, existingAnalysis: analysis) {
            updated["bpm"] = bpm
            analysis["bpm"] = bpm
            analysis["tempo_bpm_estimate"] = bpm
            analysis["bpm_status"] = "estimated_from_stems"
            analysis["bpm_source"] = bpmSource(instrumental: instrumentalAnalysis, vocal: vocalAnalysis) ?? "stem_analysis"
        } else if !missingBPM(in: updated, analysis: analysis), stringValue(analysis["bpm_status"]) == nil {
            analysis["bpm_status"] = "available_from_metadata"
        }

        if missingKey(in: updated, analysis: analysis),
           let key = preferredKey(instrumental: instrumentalAnalysis, existingAnalysis: analysis) {
            updated["key"] = key
            updated["keyscale"] = key
            analysis["key"] = key
            analysis["keyscale"] = key
            analysis["key_status"] = stringValue(instrumentalAnalysis?["key_status"]) ?? "estimated_from_audio"
            analysis["key_source"] = stringValue(instrumentalAnalysis?["key_source"]) ?? "native-dsp-instrumental"
            if let confidence = doubleValue(instrumentalAnalysis?["key_confidence"]) {
                analysis["key_confidence"] = confidence
            }
            if let runnerUp = stringValue(instrumentalAnalysis?["key_runner_up"]) {
                analysis["key_runner_up"] = runnerUp
            }
        } else if !missingKey(in: updated, analysis: analysis), stringValue(analysis["key_status"]) == nil {
            analysis["key_status"] = "available_from_metadata"
        }

        if !analysis.isEmpty {
            updated["analysis"] = analysis
        }

        let clipURL = resolve(record.file, relativeTo: datasetDirectory)
        var sidecar = readJSONObject(from: clipURL.deletingPathExtension().appendingPathExtension("json")) ?? [:]
        sidecar["analysis"] = analysis
        if let bpm = updated["bpm"] {
            sidecar["bpm"] = bpm
        }
        if let key = updated["key"] {
            sidecar["key"] = key
            sidecar["keyscale"] = updated["keyscale"] ?? key
        }
        if let separation = updated["separation"] {
            sidecar["separation"] = separation
        }
        try writeJSONObject(sidecar, to: clipURL.deletingPathExtension().appendingPathExtension("json"))

        return updated
    }

    nonisolated private static func applyAutomaticCaptionMetadata(
        record: LocalDatasetCaptionRecord,
        datasetDirectory: URL,
        force: Bool
    ) throws -> [String: Any] {
        guard shouldGenerateNativeCaption(for: record.object, force: force) else {
            return record.object
        }

        let captionSource = captionSource(in: record.object)
        let previousCaption = shouldUseCaptionAsNativeSeed(
            record: record.object,
            captionSource: captionSource,
            datasetDirectory: datasetDirectory
        ) ? explicitCaptionText(from: record.object) : ""
        guard let generatedCaption = generatedMetadataCaption(for: record.object, seedCaption: previousCaption) else {
            return record.object
        }

        var updated = record.object
        updated["caption"] = generatedCaption
        updated["prompt"] = generatedCaption
        updated["caption_source"] = "native"

        var annotation = updated["annotation"] as? [String: Any] ?? [:]
        let rawOutputs = annotation["raw_output_files"] as? [String: Any] ?? [:]
        annotation["schema_version"] = TrainingDatasetPackageWriter.annotationSchemaVersion
        annotation["updated_at"] = ISO8601DateFormatter().string(from: Date())
        annotation["caption_source"] = "native"
        annotation["caption_present"] = true
        annotation["caption_backend"] = "native-metadata-captioner"
        annotation["caption_model_id"] = "pimp-audio-metadata-caption-v1"
        annotation["caption_seed"] = previousCaption
        annotation["caption_inputs"] = nativeCaptionInputs(in: updated, usedSeed: !previousCaption.isEmpty)
        annotation["raw_output_files"] = rawOutputs
        updated["annotation"] = annotation

        let clipURL = resolve(record.file, relativeTo: datasetDirectory)
        var sidecar = readJSONObject(from: clipURL.deletingPathExtension().appendingPathExtension("json")) ?? [:]
        sidecar["caption"] = generatedCaption
        sidecar["prompt"] = generatedCaption
        sidecar["caption_source"] = "native"
        sidecar["annotation"] = annotation
        try writeJSONObject(sidecar, to: clipURL.deletingPathExtension().appendingPathExtension("json"))

        return updated
    }

    nonisolated private static func refreshCachedStemAnalysisIfNeeded(
        record: [String: Any],
        datasetDirectory: URL,
        force: Bool
    ) throws -> [StemBundleRole: [String: Any]] {
        guard let separation = record["separation"] as? [String: Any],
              let separationFile = separation["separation_file"] as? String,
              !separationFile.isEmpty else {
            return [:]
        }

        let separationURL = resolve(separationFile, relativeTo: datasetDirectory)
        guard var payload = readJSONObject(from: separationURL),
              var stems = payload["stems"] as? [String: Any] else {
            return [:]
        }

        var changed = false
        var analyses: [StemBundleRole: [String: Any]] = [:]
        for role in StemBundleRole.allCases {
            guard var stem = stems[role.assetKey] as? [String: Any] else { continue }
            let filePath = stringValue(stem["file"]) ?? stemPath(for: role, in: record)
            guard let filePath else { continue }

            let stemURL = resolve(filePath, relativeTo: datasetDirectory)
            guard FileManager.default.fileExists(atPath: stemURL.path(percentEncoded: false)) else { continue }

            var analysis = stem["analysis"] as? [String: Any] ?? [:]
            if shouldRefreshStemAnalysis(analysis, role: role, force: force) {
                let refreshed = analyzeAudioFile(stemURL, role: role.assetKey)
                analysis = mergeAudioAnalysis(existing: analysis, refreshed: refreshed, force: force)
                stem["analysis"] = analysis
                stems[role.assetKey] = stem
                changed = true
            }
            analyses[role] = analysis
        }

        if changed {
            payload["stems"] = stems
            try writeJSONObject(payload, to: separationURL)
        }
        return analyses
    }

    nonisolated private static func writeStemTrainingBundle(
        datasetDirectory: URL,
        records: [[String: Any]],
        role: StemBundleRole
    ) throws -> Int {
        let createdAt = ISO8601DateFormatter().string(from: Date())
        let clips = records.enumerated().compactMap { index, record -> [String: Any]? in
            guard let stemPath = stemPath(for: role, in: record),
                  FileManager.default.fileExists(atPath: resolve(stemPath, relativeTo: datasetDirectory).path) else {
                return nil
            }

            let prompt = promptText(from: record)
            let transcript = transcriptionText(from: record)
            let annotation = record["annotation"] as? [String: Any] ?? [:]
            let captionSource = stringValue(record["caption_source"])
                ?? stringValue(annotation["caption_source"])
                ?? (prompt.isEmpty ? "none" : "manual")
            let sourceLyrics = stringValue(record["lyrics_source"])
                ?? stringValue(annotation["lyrics_source"])
                ?? stringValue((record["transcription"] as? [String: Any])?["source"])
                ?? "transcript"
            let bundleLyrics = role == .vocals ? transcript : ""
            let bundleLyricsSource = role == .vocals
                ? (transcript.isEmpty ? "none" : sourceLyrics)
                : role.trainingLyricsSource
            var assets = record["assets"] as? [String: Any] ?? [:]
            assets[role.assetKey] = stemPath
            if let file = record["file"] as? String {
                assets["full_mix"] = file
            }
            var clipAnnotation = annotation
            clipAnnotation["schema_version"] = TrainingDatasetPackageWriter.annotationSchemaVersion
            clipAnnotation["caption_source"] = captionSource
            clipAnnotation["lyrics_source"] = bundleLyrics.isEmpty ? "none" : bundleLyricsSource
            clipAnnotation["caption_present"] = !prompt.isEmpty
            clipAnnotation["lyrics_present"] = !bundleLyrics.isEmpty

            var clip: [String: Any] = [
                "id": record["id"] as? String ?? String(format: "clip_%04d", index + 1),
                "file": stemPath,
                "assets": assets,
                "caption": prompt,
                "prompt": prompt,
                "caption_source": captionSource,
                "lyrics": bundleLyrics,
                "lyrics_source": bundleLyricsSource,
                "transcript": transcript,
                "transcription": record["transcription"] as? [String: Any] ?? [
                    "status": transcript.isEmpty ? "not_run" : "complete",
                    "text": transcript,
                ],
                "annotation_schema_version": TrainingDatasetPackageWriter.annotationSchemaVersion,
                "annotation": clipAnnotation,
                "analysis": stemAnalysis(for: role, in: record, datasetDirectory: datasetDirectory) ?? [:],
                "source": [
                    "kind": "local_file",
                    "file": record["file"] as? String ?? "",
                ],
                "tags": record["tags"] as? [Any] ?? [],
                "negative_tags": record["negative_tags"] as? [Any] ?? [],
            ]
            if let separation = record["separation"] as? [String: Any] {
                clip["separation"] = separation
            }
            return clip
        }

        let bundle: [String: Any] = [
            "pimp_audio_training_bundle_version": "1.0",
            "created_at": createdAt,
            "dataset_dir": ".",
            "include": [role.includeName],
            "training_lyrics": role.trainingLyrics,
            "training_lyrics_source": role.trainingLyricsSource,
            "annotation_schema_version": TrainingDatasetPackageWriter.annotationSchemaVersion,
            "preprocess_cache": TrainingDatasetPackageWriter.preprocessCacheFileName,
            "clip_count": clips.count,
            "clips": clips,
        ]
        try writeJSONObject(bundle, to: datasetDirectory.appendingPathComponent(role.bundleFileName))
        return clips.count
    }

    nonisolated private static func refreshDatasetMetadata(
        datasetDirectory: URL,
        result: LocalDatasetEnrichmentResult
    ) throws {
        let manifestURL = datasetDirectory.appendingPathComponent(TrainingDatasetPackageWriter.datasetManifestFileName)
        var manifest = readJSONObject(from: manifestURL) ?? [:]
        if stringValue(manifest["pimp_audio_dataset_version"]) == DatasetPackageV2.currentVersion {
            _ = try TrainingDatasetPackageWriter.refreshPackageV2FromCaptions(
                at: datasetDirectory
            )
            return
        }
        let now = ISO8601DateFormatter().string(from: Date())
        let captions = readJSONArray(from: datasetDirectory.appendingPathComponent(TrainingDatasetPackageWriter.captionsFileName))
        if !captions.isEmpty {
            manifest["clips"] = captions
            manifest["clip_count"] = captions.count
        }

        var separation = manifest["separation"] as? [String: Any] ?? [:]
        var availableAssets = ["full_mix"]
        if result.vocalBundleExists {
            availableAssets.append("vocals")
        }
        if result.instrumentalBundleExists {
            availableAssets.append("instrumental")
        }
        separation["enabled"] = result.separatedClipCount > 0
        separation["status"] = result.separatedClipCount > 0 ? "complete" : "not_run"
        separation["available_assets"] = availableAssets
        separation["updated_at"] = separation["updated_at"] ?? now
        separation["vocal_bundle"] = TrainingDatasetPackageWriter.vocalBundleFileName
        separation["instrumental_bundle"] = TrainingDatasetPackageWriter.instrumentalBundleFileName
        manifest["separation"] = separation

        var transcription = manifest["transcription"] as? [String: Any] ?? [:]
        transcription["enabled"] = result.transcribedClipCount > 0
        transcription["status"] = transcriptionStatus(for: result)
        transcription["clip_count"] = result.transcribedClipCount
        transcription["updated_at"] = transcription["updated_at"] ?? now
        transcription["artifacts"] = [
            "lyrics.txt",
            "lyrics.timed.json",
            "lyrics.lrc",
        ]
        let transcriptionModels = sortedUniqueStrings(captions.compactMap { record in
            (record["transcription"] as? [String: Any])?["model"] as? String
        })
        if !transcriptionModels.isEmpty {
            transcription["models"] = transcriptionModels
        }
        let transcriptionBackends = sortedUniqueStrings(captions.compactMap { record in
            (record["transcription"] as? [String: Any])?["backend"] as? String
        })
        if !transcriptionBackends.isEmpty {
            transcription["backends"] = transcriptionBackends
        }
        manifest["transcription"] = transcription

        var annotation = manifest["annotation"] as? [String: Any] ?? [:]
        annotation["schema_version"] = TrainingDatasetPackageWriter.annotationSchemaVersion
        annotation["updated_at"] = now
        annotation["status"] = annotationStatus(in: captions)
        annotation["approved_clip_count"] = approvedClipCount(in: captions)
        annotation["caption_sources"] = sortedUniqueStrings(captions.compactMap {
            stringValue($0["caption_source"]) ?? stringValue(($0["annotation"] as? [String: Any])?["caption_source"])
        })
        annotation["lyrics_sources"] = sortedUniqueStrings(captions.compactMap {
            stringValue($0["lyrics_source"]) ?? stringValue(($0["annotation"] as? [String: Any])?["lyrics_source"])
        })
        manifest["annotation"] = annotation

        var analysis = manifest["analysis"] as? [String: Any] ?? [:]
        let keyValues = keyMetadataValues(in: captions)
        let keyStatus = keyMetadataStatus(in: captions)
        let hasAnalysis = result.bpmEstimateCount > 0 || result.keyEstimateCount > 0
        analysis["enabled"] = hasAnalysis
        analysis["status"] = hasAnalysis ? "stem_analysis_available" : (analysis["status"] ?? "not_run")
        analysis["bpm_status"] = result.bpmEstimateCount > 0 ? "estimated_from_stems" : "not_run"
        analysis["key_status"] = keyStatus
        if let averageBPM = result.averageBPM {
            analysis["stem_tempo_bpm"] = [
                "average": averageBPM,
                "estimate_count": result.bpmEstimateCount,
            ]
        }
        if !keyValues.isEmpty {
            analysis["keys"] = keyValues
        }
        manifest["analysis"] = analysis

        var training = manifest["training"] as? [String: Any] ?? [:]
        training["default_bundle"] = TrainingDatasetPackageWriter.trainingBundleFileName
        training["full_mix_bundle"] = TrainingDatasetPackageWriter.fullMixBundleFileName
        training["vocal_bundle"] = TrainingDatasetPackageWriter.vocalBundleFileName
        training["instrumental_bundle"] = TrainingDatasetPackageWriter.instrumentalBundleFileName
        training["dataset_config"] = TrainingDatasetPackageWriter.datasetConfigFileName
        training["preprocess_cache"] = TrainingDatasetPackageWriter.preprocessCacheFileName
        training["preprocess_cache_status"] = "declared"
        manifest["training"] = training

        manifest["enrichment"] = [
            "updated_at": now,
            "status": result.separatedClipCount > 0 ? "partial" : "not_run",
            "separated_clip_count": result.separatedClipCount,
            "vocal_bundle": result.vocalBundleExists ? TrainingDatasetPackageWriter.vocalBundleFileName : "",
            "instrumental_bundle": result.instrumentalBundleExists ? TrainingDatasetPackageWriter.instrumentalBundleFileName : "",
            "transcription_status": transcriptionStatus(for: result),
            "transcribed_clip_count": result.transcribedClipCount,
            "bpm_status": result.bpmEstimateCount > 0 ? "estimated_from_stems" : "not_run",
            "key_status": keyStatus,
        ]

        try writeJSONObject(manifest, to: manifestURL)
        try TrainingDatasetPackageWriter.refreshPreprocessCacheManifest(at: datasetDirectory)
    }

    nonisolated private static func inspectResult(datasetDirectory: URL) -> LocalDatasetEnrichmentResult {
        let captions = readJSONArray(from: datasetDirectory.appendingPathComponent(TrainingDatasetPackageWriter.captionsFileName))
        let separatedClipCount = captions.filter {
            hasCachedInstrumentalSeparation(record: $0, datasetDirectory: datasetDirectory)
        }.count
        let transcribedClipCount = captions.filter { !transcriptionText(from: $0).isEmpty }.count
        let bpmEstimates = stemBPMEstimates(datasetDirectory: datasetDirectory, captions: captions)
        let averageBPM = bpmEstimates.isEmpty
            ? nil
            : rounded(bpmEstimates.reduce(0, +) / Double(bpmEstimates.count), places: 1)
        let keyValues = keyMetadataValues(in: captions)
        let autoCaptionedClipCount = nativeCaptionedClipCount(in: captions)

        return LocalDatasetEnrichmentResult(
            datasetDirectory: datasetDirectory,
            separatedClipCount: separatedClipCount,
            vocalBundleExists: FileManager.default.fileExists(
                atPath: datasetDirectory
                    .appendingPathComponent(TrainingDatasetPackageWriter.vocalBundleFileName)
                    .path
            ),
            instrumentalBundleExists: FileManager.default.fileExists(
                atPath: datasetDirectory
                    .appendingPathComponent(TrainingDatasetPackageWriter.instrumentalBundleFileName)
                    .path
            ),
            bpmEstimateCount: bpmEstimates.count,
            averageBPM: averageBPM,
            keyEstimateCount: keyValues.count,
            transcribedClipCount: transcribedClipCount,
            autoCaptionedClipCount: autoCaptionedClipCount
        )
    }

    nonisolated private static func loadCaptionRecords(from datasetDirectory: URL) throws -> [LocalDatasetCaptionRecord] {
        let url = datasetDirectory.appendingPathComponent(TrainingDatasetPackageWriter.captionsFileName)
        let records = readJSONArray(from: url)
        guard !records.isEmpty else {
            throw LocalDatasetEnrichmentError.invalidCaptions("captions.json is missing or empty.")
        }
        return try records.map { record in
            guard let file = record["file"] as? String, !file.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw LocalDatasetEnrichmentError.invalidCaptions("a caption record is missing its file path.")
            }
            return LocalDatasetCaptionRecord(object: record, file: file)
        }
    }

    nonisolated private static func hasCachedInstrumentalSeparation(
        record: [String: Any],
        datasetDirectory: URL
    ) -> Bool {
        guard let vocals = stemPath(for: .vocals, in: record),
              let instrumental = stemPath(for: .instrumental, in: record) else {
            return false
        }
        return FileManager.default.fileExists(atPath: resolve(vocals, relativeTo: datasetDirectory).path) &&
            FileManager.default.fileExists(atPath: resolve(instrumental, relativeTo: datasetDirectory).path)
    }

    nonisolated private static func stemPath(for role: StemBundleRole, in record: [String: Any]) -> String? {
        if let assets = record["assets"] as? [String: Any],
           let value = assets[role.assetKey] as? String,
           !value.isEmpty {
            return value
        }
        guard let separation = record["separation"] as? [String: Any],
              let stems = separation["stems"] as? [String: Any],
              let value = stems[role.assetKey] as? String,
              !value.isEmpty else {
            return nil
        }
        return value
    }

    nonisolated private static func stemAnalysis(
        for role: StemBundleRole,
        in record: [String: Any],
        datasetDirectory: URL
    ) -> [String: Any]? {
        guard let separation = record["separation"] as? [String: Any],
              let separationFile = separation["separation_file"] as? String,
              !separationFile.isEmpty,
              let payload = readJSONObject(from: resolve(separationFile, relativeTo: datasetDirectory)),
              let stems = payload["stems"] as? [String: Any],
              let stem = stems[role.assetKey] as? [String: Any],
              let analysis = stem["analysis"] as? [String: Any] else {
            return nil
        }
        return analysis
    }

    nonisolated private static func stemBPMEstimates(
        datasetDirectory: URL,
        captions: [[String: Any]]
    ) -> [Double] {
        captions.flatMap { record -> [Double] in
            StemBundleRole.allCases.compactMap { role in
                stemAnalysis(for: role, in: record, datasetDirectory: datasetDirectory)
                    .flatMap { doubleValue($0["tempo_bpm_estimate"]) }
            }
        }
    }

    nonisolated private static func keyMetadataValues(in captions: [[String: Any]]) -> [String] {
        sortedUniqueStrings(captions.compactMap { record in
            if let key = stringValue(record["key"]) ?? stringValue(record["keyscale"]) {
                return key
            }
            if let analysis = record["analysis"] as? [String: Any] {
                return stringValue(analysis["key"]) ?? stringValue(analysis["keyscale"])
            }
            return nil
        })
    }

    nonisolated private static func keyMetadataStatus(in captions: [[String: Any]]) -> String {
        guard !keyMetadataValues(in: captions).isEmpty else {
            return "not_run"
        }

        let hasNativeEstimate = captions.contains { record in
            guard let analysis = record["analysis"] as? [String: Any] else { return false }
            if stringValue(analysis["key_status"]) == "estimated_from_audio" {
                return true
            }
            return stringValue(analysis["key_source"])?.contains("native-dsp") == true
        }
        return hasNativeEstimate ? "estimated_from_audio" : "available_from_metadata"
    }

    nonisolated static func analyzeAudioFile(_ url: URL, role: String? = nil) -> [String: Any] {
        var object: [String: Any] = [
            "format": url.pathExtension.lowercased().isEmpty ? "unknown" : url.pathExtension.lowercased(),
            "metadata_status": "not_collected",
        ]
        if let attributes = try? FileManager.default.attributesOfItem(atPath: url.path),
           let byteSize = attributes[.size] as? NSNumber {
            object["byte_size"] = byteSize.int64Value
        }

        do {
            let audioFile = try AVAudioFile(forReading: url)
            let sampleRate = audioFile.fileFormat.sampleRate
            object["metadata_status"] = "ok"
            object["sample_rate"] = Int(sampleRate.rounded())
            object["channels"] = Int(audioFile.fileFormat.channelCount)
            object["frame_count"] = Int64(audioFile.length)
            if sampleRate > 0 {
                object["duration_seconds"] = rounded(Double(audioFile.length) / sampleRate, places: 3)
            }
            if let sampleBuffer = try? monoSamples(from: url) {
                if let bpm = estimateTempoBPM(from: sampleBuffer) {
                    object["tempo_bpm_estimate"] = bpm
                }
                if role?.lowercased() != "vocals",
                   let keyEstimate = estimateKeyScale(from: sampleBuffer) {
                    object["key"] = keyEstimate.keyscale
                    object["keyscale"] = keyEstimate.keyscale
                    object["key_status"] = "estimated_from_audio"
                    object["key_source"] = "native-dsp-\(role ?? "audio")"
                    object["key_confidence"] = keyEstimate.confidence
                    object["key_tonic"] = keyEstimate.tonic
                    object["key_mode"] = keyEstimate.mode
                    if let runnerUp = keyEstimate.runnerUp {
                        object["key_runner_up"] = runnerUp
                    }
                }
            }
        } catch {
            object["metadata_status"] = "unreadable_audio_header"
            object["metadata_error"] = error.localizedDescription
        }
        return object
    }

    nonisolated private static func shouldRefreshStemAnalysis(
        _ analysis: [String: Any],
        role: StemBundleRole,
        force: Bool
    ) -> Bool {
        if force { return true }
        if doubleValue(analysis["tempo_bpm_estimate"]) == nil {
            return true
        }
        if role == .instrumental,
           stringValue(analysis["key"]) == nil,
           stringValue(analysis["keyscale"]) == nil {
            return true
        }
        return false
    }

    nonisolated private static func mergeAudioAnalysis(
        existing: [String: Any],
        refreshed: [String: Any],
        force: Bool
    ) -> [String: Any] {
        var merged = existing
        for (key, value) in refreshed where force || merged[key] == nil {
            merged[key] = value
        }
        return merged
    }

    nonisolated private static func missingBPM(
        in record: [String: Any],
        analysis: [String: Any]
    ) -> Bool {
        doubleValue(record["bpm"]) == nil &&
            doubleValue(analysis["bpm"]) == nil &&
            doubleValue(analysis["tempo_bpm_estimate"]) == nil
    }

    nonisolated private static func missingKey(
        in record: [String: Any],
        analysis: [String: Any]
    ) -> Bool {
        stringValue(record["key"]) == nil &&
            stringValue(record["keyscale"]) == nil &&
            stringValue(analysis["key"]) == nil &&
            stringValue(analysis["keyscale"]) == nil
    }

    nonisolated private static func preferredBPM(
        instrumental: [String: Any]?,
        vocal: [String: Any]?,
        existingAnalysis: [String: Any]
    ) -> Double? {
        [
            instrumental?["tempo_bpm_estimate"],
            instrumental?["bpm"],
            existingAnalysis["tempo_bpm_estimate"],
            existingAnalysis["bpm"],
            vocal?["tempo_bpm_estimate"],
            vocal?["bpm"],
        ].compactMap(doubleValue).first
    }

    nonisolated private static func bpmSource(
        instrumental: [String: Any]?,
        vocal: [String: Any]?
    ) -> String? {
        if doubleValue(instrumental?["tempo_bpm_estimate"]) != nil || doubleValue(instrumental?["bpm"]) != nil {
            return stringValue(instrumental?["bpm_source"]) ?? "instrumental_stem"
        }
        if doubleValue(vocal?["tempo_bpm_estimate"]) != nil || doubleValue(vocal?["bpm"]) != nil {
            return stringValue(vocal?["bpm_source"]) ?? "vocal_stem"
        }
        return nil
    }

    nonisolated private static func preferredKey(
        instrumental: [String: Any]?,
        existingAnalysis: [String: Any]
    ) -> String? {
        stringValue(instrumental?["keyscale"])
            ?? stringValue(instrumental?["key"])
            ?? stringValue(existingAnalysis["keyscale"])
            ?? stringValue(existingAnalysis["key"])
    }

    nonisolated private static func estimateTempoBPM(
        from sampleBuffer: (samples: [Float], sampleRate: Double)
    ) -> Double? {
        guard sampleBuffer.samples.count > 4_096 else {
            return nil
        }
        let sampleRate = sampleBuffer.sampleRate
        let values = sampleBuffer.samples
        let frameSize = 1_024
        let hopSize = 512
        guard values.count > frameSize * 4 else { return nil }

        var envelope: [Double] = []
        var previousEnergy = 0.0
        var index = 0
        while index + frameSize < values.count {
            var energy = 0.0
            for frame in 0..<frameSize {
                let sample = Double(values[index + frame])
                energy += sample * sample
            }
            let onset = max(0, energy - previousEnergy)
            envelope.append(onset)
            previousEnergy = energy
            index += hopSize
        }

        guard envelope.count > 8 else { return nil }
        let mean = envelope.reduce(0, +) / Double(envelope.count)
        let centered = envelope.map { max(0, $0 - mean) }
        guard centered.contains(where: { $0 > 0 }) else { return nil }

        let envelopeRate = sampleRate / Double(hopSize)
        let minLag = max(1, Int((60.0 / 200.0) * envelopeRate))
        let maxLag = min(centered.count - 1, Int((60.0 / 50.0) * envelopeRate))
        guard minLag < maxLag else { return nil }

        func autocorrelationScore(for lag: Int) -> Double {
            var score = 0.0
            for i in lag..<centered.count {
                score += centered[i] * centered[i - lag]
            }
            return score
        }

        var bestLag = minLag
        var bestScore = 0.0
        for lag in minLag...maxLag {
            let score = autocorrelationScore(for: lag)
            if score > bestScore {
                bestScore = score
                bestLag = lag
            }
        }
        guard bestScore > 0 else { return nil }

        var selectedLag = bestLag
        let selectedBPM = 60.0 * envelopeRate / Double(selectedLag)
        if selectedBPM < 70 {
            let doubleTimeLag = max(minLag, bestLag / 2)
            if doubleTimeLag < bestLag {
                let doubleTimeScore = autocorrelationScore(for: doubleTimeLag)
                if doubleTimeScore >= bestScore * 0.45 {
                    selectedLag = doubleTimeLag
                }
            }
        } else if selectedBPM > 160 {
            let halfTimeLag = min(maxLag, bestLag * 2)
            if halfTimeLag > bestLag {
                let halfTimeScore = autocorrelationScore(for: halfTimeLag)
                if halfTimeScore >= bestScore * 0.45 {
                    selectedLag = halfTimeLag
                }
            }
        }
        return rounded(60.0 * envelopeRate / Double(selectedLag), places: 1)
    }

    nonisolated private static func estimateKeyScale(
        from sampleBuffer: (samples: [Float], sampleRate: Double)
    ) -> AudioKeyScaleEstimate? {
        let samples = sampleBuffer.samples
        guard samples.count >= 4_096 else { return nil }

        let frameSize = 4_096
        let hopSize = 2_048
        guard let dft = try? vDSP.DiscreteFourierTransform<Float>(
            count: frameSize,
            direction: .forward,
            transformType: .complexComplex,
            ofType: Float.self
        ) else {
            return nil
        }

        let sampleRate = sampleBuffer.sampleRate
        let noteNames = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        let majorProfile = normalizedProfile([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        let minorProfile = normalizedProfile([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        var window = [Float](repeating: 0, count: frameSize)
        vDSP_hann_window(&window, vDSP_Length(frameSize), Int32(vDSP_HANN_NORM))

        var chroma = [Double](repeating: 0, count: 12)
        var inputReal = [Float](repeating: 0, count: frameSize)
        let inputImaginary = [Float](repeating: 0, count: frameSize)
        var outputReal = [Float](repeating: 0, count: frameSize)
        var outputImaginary = [Float](repeating: 0, count: frameSize)
        var start = 0
        while start + frameSize <= samples.count {
            for index in 0..<frameSize {
                inputReal[index] = samples[start + index] * window[index]
            }
            dft.transform(
                inputReal: inputReal,
                inputImaginary: inputImaginary,
                outputReal: &outputReal,
                outputImaginary: &outputImaginary
            )

            for bin in 1..<(frameSize / 2) {
                let frequency = Double(bin) * sampleRate / Double(frameSize)
                guard frequency >= 41.0, frequency <= 4_200.0 else { continue }
                let midi = Int((69.0 + 12.0 * log2(frequency / 440.0)).rounded())
                guard midi >= 28, midi <= 108 else { continue }
                let real = Double(outputReal[bin])
                let imaginary = Double(outputImaginary[bin])
                let magnitude = sqrt((real * real) + (imaginary * imaginary))
                guard magnitude.isFinite, magnitude > 0 else { continue }
                let pitchClass = ((midi % 12) + 12) % 12
                chroma[pitchClass] += log1p(magnitude)
            }
            start += hopSize
        }

        let chromaTotal = chroma.reduce(0, +)
        guard chromaTotal > 0.001 else { return nil }
        let normalizedChroma = normalizedProfile(chroma)

        var candidates: [(tonic: Int, mode: String, score: Double)] = []
        for tonic in 0..<12 {
            candidates.append((
                tonic: tonic,
                mode: "major",
                score: keyProfileScore(chroma: normalizedChroma, profile: majorProfile, tonic: tonic)
            ))
            candidates.append((
                tonic: tonic,
                mode: "minor",
                score: keyProfileScore(chroma: normalizedChroma, profile: minorProfile, tonic: tonic)
            ))
        }
        candidates.sort { $0.score > $1.score }
        guard let best = candidates.first else { return nil }
        let runnerUp = candidates.dropFirst().first
        let confidence = max(0, min(1, (best.score - (runnerUp?.score ?? 0)) / max(abs(best.score), 0.001)))
        guard confidence >= 0.015 else { return nil }

        let tonic = noteNames[best.tonic]
        let runnerUpLabel = runnerUp.map { "\(noteNames[$0.tonic]) \($0.mode)" }
        return AudioKeyScaleEstimate(
            keyscale: "\(tonic) \(best.mode)",
            confidence: rounded(confidence, places: 3),
            tonic: tonic,
            mode: best.mode,
            runnerUp: runnerUpLabel
        )
    }

    nonisolated private static func normalizedProfile(_ values: [Double]) -> [Double] {
        guard !values.isEmpty else { return [] }
        let mean = values.reduce(0, +) / Double(values.count)
        let centered = values.map { $0 - mean }
        let norm = sqrt(centered.reduce(0) { $0 + ($1 * $1) })
        guard norm > 0 else { return centered }
        return centered.map { $0 / norm }
    }

    nonisolated private static func keyProfileScore(
        chroma: [Double],
        profile: [Double],
        tonic: Int
    ) -> Double {
        guard chroma.count == 12, profile.count == 12 else { return 0 }
        var score = 0.0
        for pitchClass in 0..<12 {
            let interval = (pitchClass - tonic + 12) % 12
            score += chroma[pitchClass] * profile[interval]
        }
        return score
    }

    nonisolated private static func monoSamples(from url: URL) throws -> (samples: [Float], sampleRate: Double) {
        let audioFile = try AVAudioFile(forReading: url)
        let format = audioFile.processingFormat
        let frameCount = AVAudioFrameCount(min(audioFile.length, 4_000_000))
        guard frameCount > 0,
              let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
            return ([], format.sampleRate)
        }
        try audioFile.read(into: buffer, frameCount: frameCount)
        guard let channelData = buffer.floatChannelData else {
            return ([], format.sampleRate)
        }

        let channels = max(1, Int(format.channelCount))
        let frames = Int(buffer.frameLength)
        var samples = [Float](repeating: 0, count: frames)
        for frame in 0..<frames {
            var sum: Float = 0
            for channel in 0..<channels {
                sum += channelData[channel][frame]
            }
            samples[frame] = sum / Float(channels)
        }
        return (samples, format.sampleRate)
    }

    nonisolated private static func promptText(from record: [String: Any]) -> String {
        for key in ["prompt", "caption"] {
            if let value = record[key] as? String,
               !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                return value.trimmingCharacters(in: .whitespacesAndNewlines)
            }
        }
        return record["file"] as? String ?? ""
    }

    nonisolated private static func transcriptionText(from record: [String: Any]) -> String {
        if let transcription = record["transcription"] as? [String: Any],
           let text = transcription["text"] as? String,
           !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return text.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        if let transcript = record["transcript"] as? String,
           !transcript.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return transcript.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return ""
    }

    nonisolated private static func transcriptionStatus(for result: LocalDatasetEnrichmentResult) -> String {
        guard result.transcribedClipCount > 0 else { return "not_run" }
        if result.separatedClipCount > 0, result.transcribedClipCount < result.separatedClipCount {
            return "partial"
        }
        return "complete"
    }

    nonisolated private static func hasCompletedTranscription(_ record: [String: Any]) -> Bool {
        !transcriptionText(from: record).isEmpty
    }

    nonisolated private static func shouldGenerateNativeCaption(
        for record: [String: Any],
        force: Bool
    ) -> Bool {
        let source = captionSource(in: record)
        switch source {
        case "none", "manual", "filename", "native":
            return true
        default:
            return false
        }
    }

    nonisolated private static func captionSource(in record: [String: Any]) -> String {
        let annotation = record["annotation"] as? [String: Any] ?? [:]
        return stringValue(record["caption_source"])
            ?? stringValue(annotation["caption_source"])
            ?? (explicitCaptionText(from: record).isEmpty ? "none" : "manual")
    }

    nonisolated private static func shouldUseCaptionAsNativeSeed(
        record: [String: Any],
        captionSource: String,
        datasetDirectory: URL
    ) -> Bool {
        guard captionSource != "filename" else { return false }
        if captionSource == "manual",
           datasetCaptionMode(datasetDirectory: datasetDirectory) == "filename",
           looksLikeFilenameCaption(record: record) {
            return false
        }
        return true
    }

    nonisolated private static func datasetCaptionMode(datasetDirectory: URL) -> String? {
        let manifestURL = datasetDirectory.appendingPathComponent(TrainingDatasetPackageWriter.datasetManifestFileName)
        return readJSONObject(from: manifestURL).flatMap { stringValue($0["caption_mode"]) }
    }

    nonisolated private static func looksLikeFilenameCaption(record: [String: Any]) -> Bool {
        let caption = explicitCaptionText(from: record)
        guard !caption.isEmpty else { return false }

        let candidates = filenameCaptionCandidates(record: record)
        let normalizedCaption = normalizedCaptionComparisonText(caption)
        return candidates.contains { candidate in
            let normalizedCandidate = normalizedCaptionComparisonText(candidate)
            guard !normalizedCandidate.isEmpty else { return false }
            return normalizedCaption == normalizedCandidate ||
                normalizedCaption.hasPrefix("\(normalizedCandidate) clip ") ||
                normalizedCaption.hasPrefix("\(normalizedCandidate) take ")
        }
    }

    nonisolated private static func filenameCaptionCandidates(record: [String: Any]) -> [String] {
        var candidates: [String] = []
        if let file = stringValue(record["file"]) {
            candidates.append(URL(fileURLWithPath: file).deletingPathExtension().lastPathComponent)
        }
        if let source = record["source"] as? [String: Any],
           let fileName = stringValue(source["file_name"]) {
            let stem = URL(fileURLWithPath: fileName).deletingPathExtension().lastPathComponent
            candidates.append(stem)
        }
        if let assets = record["assets"] as? [String: Any],
           let fullMix = stringValue(assets["full_mix"]) {
            candidates.append(URL(fileURLWithPath: fullMix).deletingPathExtension().lastPathComponent)
        }
        return sortedUniqueStrings(candidates)
    }

    nonisolated private static func normalizedCaptionComparisonText(_ value: String) -> String {
        value
            .lowercased()
            .replacingOccurrences(of: "(", with: " ")
            .replacingOccurrences(of: ")", with: " ")
            .replacingOccurrences(of: "_", with: " ")
            .replacingOccurrences(of: "-", with: " ")
            .components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { !$0.isEmpty }
            .joined(separator: " ")
    }

    nonisolated private static func explicitCaptionText(from record: [String: Any]) -> String {
        for key in ["caption", "prompt"] {
            if let value = record[key] as? String {
                let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmed.isEmpty {
                    return trimmed
                }
            }
        }
        return ""
    }

    nonisolated private static func generatedMetadataCaption(
        for record: [String: Any],
        seedCaption: String
    ) -> String? {
        var components: [String] = []
        appendCaptionComponent(seedCaption, to: &components)

        if let bpm = captionBPM(in: record) {
            appendCaptionComponent("\(tempoDescriptor(for: bpm)) \(formatBPM(bpm)) BPM", to: &components)
        }
        if let key = captionKey(in: record) {
            appendCaptionComponent(key, to: &components)
        }
        appendCaptionComponent(contentDescriptor(for: record), to: &components)

        guard !components.isEmpty else { return nil }
        return components.joined(separator: ", ")
    }

    nonisolated private static func appendCaptionComponent(_ value: String, to components: inout [String]) {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        let normalized = trimmed.lowercased()
        guard !components.contains(where: { existing in
            let existingNormalized = existing.lowercased()
            return existingNormalized == normalized ||
                existingNormalized.contains(normalized) ||
                normalized.contains(existingNormalized)
        }) else {
            return
        }
        components.append(trimmed)
    }

    nonisolated private static func captionBPM(in record: [String: Any]) -> Double? {
        if let bpm = doubleValue(record["bpm"]) {
            return bpm
        }
        if let analysis = record["analysis"] as? [String: Any] {
            return doubleValue(analysis["bpm"]) ?? doubleValue(analysis["tempo_bpm_estimate"])
        }
        return nil
    }

    nonisolated private static func captionKey(in record: [String: Any]) -> String? {
        if let key = stringValue(record["keyscale"]) ?? stringValue(record["key"]) {
            return key
        }
        if let analysis = record["analysis"] as? [String: Any] {
            return stringValue(analysis["keyscale"]) ?? stringValue(analysis["key"])
        }
        return nil
    }

    nonisolated private static func tempoDescriptor(for bpm: Double) -> String {
        switch bpm {
        case ..<75:
            return "very slow"
        case ..<95:
            return "slow"
        case ..<125:
            return "mid-tempo"
        case ..<155:
            return "upbeat"
        default:
            return "fast"
        }
    }

    nonisolated private static func formatBPM(_ bpm: Double) -> String {
        bpm.rounded() == bpm
            ? String(Int(bpm))
            : bpm.formatted(.number.precision(.fractionLength(1)))
    }

    nonisolated private static func contentDescriptor(for record: [String: Any]) -> String {
        let role = stringValue(record["role"]) ?? "full_mix"
        let hasVocals = boolValue(record["vocal_overlap"]) == true ||
            !transcriptionText(from: record).isEmpty ||
            stemPath(for: .vocals, in: record) != nil
        let hasInstrumental = stemPath(for: .instrumental, in: record) != nil

        switch role {
        case "vocal", "vocals":
            return "isolated vocal stem"
        case "instrumental":
            return "instrumental backing stem"
        default:
            if hasVocals && hasInstrumental {
                return "full mix with vocals and instrumental backing"
            }
            if hasVocals {
                return "full mix with vocals"
            }
            if hasInstrumental {
                return "instrumental full mix"
            }
            return "full mix"
        }
    }

    nonisolated private static func nativeCaptionInputs(in record: [String: Any], usedSeed: Bool) -> [String] {
        var inputs: [String] = []
        if usedSeed {
            inputs.append("caption_seed")
        }
        if captionBPM(in: record) != nil {
            inputs.append("bpm")
        }
        if captionKey(in: record) != nil {
            inputs.append("key")
        }
        if !transcriptionText(from: record).isEmpty {
            inputs.append("transcript")
        }
        if record["separation"] != nil {
            inputs.append("stems")
        }
        return inputs
    }

    nonisolated private static func nativeCaptionedClipCount(in captions: [[String: Any]]) -> Int {
        captions.filter { record in
            if stringValue(record["caption_source"]) == "native" {
                return true
            }
            if let annotation = record["annotation"] as? [String: Any],
               stringValue(annotation["caption_source"]) == "native" {
                return true
            }
            return false
        }.count
    }

    nonisolated private static func readJSONObject(from url: URL) -> [String: Any]? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
    }

    nonisolated private static func readJSONArray(from url: URL) -> [[String: Any]] {
        guard let data = try? Data(contentsOf: url) else { return [] }
        return (try? JSONSerialization.jsonObject(with: data)) as? [[String: Any]] ?? []
    }

    nonisolated private static func writeJSONObject(_ object: [String: Any], to url: URL) throws {
        let data = try JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: url, options: .atomic)
    }

    nonisolated private static func writeJSONObjectArray(_ object: [[String: Any]], to url: URL) throws {
        let data = try JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: url, options: .atomic)
    }

    nonisolated private static func relativePath(for url: URL, from baseDirectory: URL) -> String {
        let base = normalizedDirectoryPath(baseDirectory)
        let path = url.standardizedFileURL.path
        guard path.hasPrefix(base) else {
            return url.lastPathComponent
        }
        return String(path.dropFirst(base.count))
    }

    nonisolated private static func normalizedDirectoryPath(_ url: URL) -> String {
        var path = url.standardizedFileURL.path
        if !path.hasSuffix("/") {
            path.append("/")
        }
        return path
    }

    nonisolated private static func resolve(_ path: String, relativeTo baseDirectory: URL) -> URL {
        let expanded = (path as NSString).expandingTildeInPath
        if expanded.hasPrefix("/") {
            return URL(fileURLWithPath: expanded).standardizedFileURL
        }
        return baseDirectory.appendingPathComponent(expanded).standardizedFileURL
    }

    nonisolated private static func doubleValue(_ value: Any?) -> Double? {
        if let number = value as? NSNumber {
            return number.doubleValue
        }
        if let string = value as? String {
            return Double(string)
        }
        return nil
    }

    nonisolated private static func stringValue(_ value: Any?) -> String? {
        switch value {
        case let string as String:
            let trimmed = string.trimmingCharacters(in: .whitespacesAndNewlines)
            return trimmed.isEmpty ? nil : trimmed
        case let number as NSNumber:
            return number.stringValue
        case let double as Double:
            return String(double)
        case let int as Int:
            return String(int)
        default:
            return nil
        }
    }

    nonisolated private static func boolValue(_ value: Any?) -> Bool? {
        switch value {
        case let bool as Bool:
            return bool
        case let number as NSNumber:
            return number.boolValue
        case let string as String:
            return ["true", "yes", "1"].contains(string.lowercased())
        default:
            return nil
        }
    }

    nonisolated private static func annotationStatus(in captions: [[String: Any]]) -> String {
        guard !captions.isEmpty else { return "empty" }
        let approved = approvedClipCount(in: captions)
        if approved == captions.count {
            return "reviewed"
        }
        if approved > 0 {
            return "partial_review"
        }
        return "manual_review_required"
    }

    nonisolated private static func approvedClipCount(in captions: [[String: Any]]) -> Int {
        captions.filter { record in
            if let trainingReview = record["training_review"] as? [String: Any],
               boolValue(trainingReview["approved"]) == true {
                return true
            }
            if let annotation = record["annotation"] as? [String: Any],
               stringValue(annotation["review_status"]) == "approved" {
                return true
            }
            return false
        }.count
    }

    nonisolated private static func rounded(_ value: Double, places: Int) -> Double {
        let scale = pow(10, Double(places))
        return (value * scale).rounded() / scale
    }

    nonisolated private static func sortedUniqueStrings(_ values: [String]) -> [String] {
        Array(Set(values.filter { !$0.isEmpty })).sorted()
    }
}

struct DatasetLyricsTranscriptionArtifact: Equatable, Sendable {
    let text: String
    let source: String
    let backend: String
    let model: String
    let alignerModel: String?
    let lyricsURL: URL
    let timedLyricsURL: URL
    let lrcURL: URL
    let wordCount: Int
    let elapsedSeconds: Double
}

protocol DatasetLyricsTranscribing: Sendable {
    func transcribe(
        vocalsURL: URL,
        outputDirectory: URL,
        force: Bool
    ) async throws -> DatasetLyricsTranscriptionArtifact
}

protocol DatasetLyricsTranscriptionEvicting: Sendable {
    func evictModels() async
}

protocol DatasetLyricsAligning: Sendable {
    func align(
        lyrics: String,
        audioURL: URL,
        outputDirectory: URL,
        force: Bool
    ) async throws -> DatasetLyricsTranscriptionArtifact
}

actor LyricsTranscriptionService:
    DatasetLyricsTranscribing,
    DatasetLyricsTranscriptionEvicting,
    DatasetLyricsAligning
{
    private let transcriber: QwenLyricsTranscriber
    private let options: LyricsTranscriptionOptions

    init(
        transcriber: QwenLyricsTranscriber = QwenLyricsTranscriber(),
        options: LyricsTranscriptionOptions = .qualityDataset
    ) {
        self.transcriber = transcriber
        self.options = options
    }

    func evictModels() async {
        await transcriber.evictModels()
    }

    func transcribe(
        vocalsURL: URL,
        outputDirectory: URL,
        force: Bool
    ) async throws -> DatasetLyricsTranscriptionArtifact {
        try FileManager.default.createDirectory(at: outputDirectory, withIntermediateDirectories: true)

        let lyricsURL = outputDirectory.appendingPathComponent("lyrics.txt")
        let timedLyricsURL = outputDirectory.appendingPathComponent("lyrics.timed.json")
        let lrcURL = outputDirectory.appendingPathComponent("lyrics.lrc")
        if !force,
           FileManager.default.fileExists(atPath: lyricsURL.path(percentEncoded: false)),
           let cachedText = try? String(contentsOf: lyricsURL, encoding: .utf8),
           !cachedText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            let text = cachedText.trimmingCharacters(in: .whitespacesAndNewlines)
            if !FileManager.default.fileExists(atPath: timedLyricsURL.path(percentEncoded: false)) {
                try Self.writeJSONObject(
                    cachedTimedLyricsPayload(text: text, vocalsURL: vocalsURL),
                    to: timedLyricsURL
                )
            }
            if !FileManager.default.fileExists(atPath: lrcURL.path(percentEncoded: false)) {
                try Self.lrcText(from: text).write(to: lrcURL, atomically: true, encoding: .utf8)
            }
            return DatasetLyricsTranscriptionArtifact(
                text: text,
                source: "cached",
                backend: QwenLyricsTranscriber.backendName,
                model: options.modelID,
                alignerModel: options.alignerModelID,
                lyricsURL: lyricsURL,
                timedLyricsURL: timedLyricsURL,
                lrcURL: lrcURL,
                wordCount: text.split(whereSeparator: \.isWhitespace).count,
                elapsedSeconds: 0
            )
        }

        let result = try await transcriber.transcribe(audioURL: vocalsURL, options: options)
        try result.text.write(to: lyricsURL, atomically: true, encoding: .utf8)
        try Self.writeJSONObject(
            timedLyricsPayload(result: result, vocalsURL: vocalsURL),
            to: timedLyricsURL
        )
        try Self.lrcText(from: result).write(to: lrcURL, atomically: true, encoding: .utf8)

        return DatasetLyricsTranscriptionArtifact(
            text: result.text,
            source: "qwen3-asr",
            backend: result.backend,
            model: result.modelID,
            alignerModel: result.alignerModelID,
            lyricsURL: lyricsURL,
            timedLyricsURL: timedLyricsURL,
            lrcURL: lrcURL,
            wordCount: result.words.count,
            elapsedSeconds: result.elapsedSeconds
        )
    }

    func align(
        lyrics: String,
        audioURL: URL,
        outputDirectory: URL,
        force: Bool
    ) async throws -> DatasetLyricsTranscriptionArtifact {
        try FileManager.default.createDirectory(
            at: outputDirectory,
            withIntermediateDirectories: true
        )
        let lyricsURL = outputDirectory.appendingPathComponent("lyrics.txt")
        let timedLyricsURL = outputDirectory.appendingPathComponent(
            "lyrics.timed.json"
        )
        let lrcURL = outputDirectory.appendingPathComponent("lyrics.lrc")
        if !force,
           FileManager.default.fileExists(
               atPath: timedLyricsURL.path(percentEncoded: false)
           ),
           FileManager.default.fileExists(
               atPath: lrcURL.path(percentEncoded: false)
           ) {
            if !FileManager.default.fileExists(
                atPath: lyricsURL.path(percentEncoded: false)
            ) {
                try lyrics.write(
                    to: lyricsURL,
                    atomically: true,
                    encoding: .utf8
                )
            }
            return DatasetLyricsTranscriptionArtifact(
                text: lyrics,
                source: "cached-alignment",
                backend: QwenLyricsTranscriber.alignerBackendName,
                model: options.alignerModelID ?? "",
                alignerModel: options.alignerModelID,
                lyricsURL: lyricsURL,
                timedLyricsURL: timedLyricsURL,
                lrcURL: lrcURL,
                wordCount: lyrics.split(whereSeparator: \.isWhitespace).count,
                elapsedSeconds: 0
            )
        }

        let result = try await transcriber.align(
            audioURL: audioURL,
            text: lyrics,
            options: options
        )
        let compatibleResult = LyricsTranscriptionResult(
            backend: result.backend,
            modelID: result.modelID,
            alignerModelID: result.modelID,
            text: result.text,
            words: result.words,
            sampleRate: result.sampleRate,
            trim: result.trim,
            elapsedSeconds: result.elapsedSeconds
        )
        try result.text.write(
            to: lyricsURL,
            atomically: true,
            encoding: .utf8
        )
        try Self.writeJSONObject(
            timedLyricsPayload(
                result: compatibleResult,
                vocalsURL: audioURL
            ),
            to: timedLyricsURL
        )
        try Self.lrcText(from: compatibleResult).write(
            to: lrcURL,
            atomically: true,
            encoding: .utf8
        )
        return DatasetLyricsTranscriptionArtifact(
            text: result.text,
            source: "qwen3-forced-alignment",
            backend: result.backend,
            model: result.modelID,
            alignerModel: result.modelID,
            lyricsURL: lyricsURL,
            timedLyricsURL: timedLyricsURL,
            lrcURL: lrcURL,
            wordCount: result.words.count,
            elapsedSeconds: result.elapsedSeconds
        )
    }

    private func timedLyricsPayload(result: LyricsTranscriptionResult, vocalsURL: URL) -> [String: Any] {
        var payload: [String: Any] = [
            "pimp_audio_lyrics_transcription_version": "1.0",
            "generated_at": ISO8601DateFormatter().string(from: Date()),
            "source_vocals": vocalsURL.lastPathComponent,
            "backend": result.backend,
            "model": result.modelID,
            "text": result.text,
            "sample_rate": result.sampleRate,
            "elapsed_seconds": Self.rounded(result.elapsedSeconds, places: 3),
            "trim": [
                "leading_seconds": Self.rounded(result.trim.leadingSeconds, places: 3),
                "trailing_seconds": Self.rounded(result.trim.trailingSeconds, places: 3),
            ],
            "words": result.words.map { word in
                [
                    "text": word.text,
                    "start": Self.rounded(word.startTime, places: 3),
                    "end": Self.rounded(word.endTime, places: 3),
                ]
            },
        ]
        if let alignerModelID = result.alignerModelID {
            payload["aligner_model"] = alignerModelID
        }
        return payload
    }

    private func cachedTimedLyricsPayload(text: String, vocalsURL: URL) -> [String: Any] {
        [
            "pimp_audio_lyrics_transcription_version": "1.0",
            "generated_at": ISO8601DateFormatter().string(from: Date()),
            "source_vocals": vocalsURL.lastPathComponent,
            "backend": QwenLyricsTranscriber.backendName,
            "model": options.modelID,
            "source": "cached",
            "text": text,
            "sample_rate": options.sampleRate,
            "elapsed_seconds": 0,
            "trim": [
                "leading_seconds": 0,
                "trailing_seconds": 0,
            ],
            "words": [],
        ]
    }

    private static func lrcText(from result: LyricsTranscriptionResult) -> String {
        guard !result.words.isEmpty else {
            let text = result.text.trimmingCharacters(in: .whitespacesAndNewlines)
            return lrcText(from: text)
        }

        var lines: [String] = []
        var currentWords: [LyricsTranscriptionWord] = []
        for word in result.words {
            currentWords.append(word)
            let shouldBreak = currentWords.count >= 8 || word.text.range(of: #"[.!?]$"#, options: .regularExpression) != nil
            if shouldBreak {
                lines.append(lrcLine(from: currentWords))
                currentWords.removeAll(keepingCapacity: true)
            }
        }
        if !currentWords.isEmpty {
            lines.append(lrcLine(from: currentWords))
        }
        return lines.joined(separator: "\n") + "\n"
    }

    private static func lrcText(from text: String) -> String {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? "" : "[00:00.00]\(trimmed)\n"
    }

    private static func lrcLine(from words: [LyricsTranscriptionWord]) -> String {
        let timestamp = lrcTimestamp(words.first?.startTime ?? 0)
        let text = words.map(\.text).joined(separator: " ")
        return "[\(timestamp)]\(text)"
    }

    private static func lrcTimestamp(_ seconds: Double) -> String {
        let clamped = max(0, seconds)
        let minutes = Int(clamped / 60)
        let remaining = clamped - Double(minutes * 60)
        return String(format: "%02d:%05.2f", minutes, remaining)
    }

    private static func writeJSONObject(_ object: [String: Any], to url: URL) throws {
        let data = try JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: url, options: .atomic)
    }

    private static func rounded(_ value: Double, places: Int) -> Double {
        let scale = pow(10, Double(places))
        return (value * scale).rounded() / scale
    }
}

private struct LocalDatasetCaptionRecord {
    var object: [String: Any]
    let file: String
}

private struct AudioKeyScaleEstimate {
    let keyscale: String
    let confidence: Double
    let tonic: String
    let mode: String
    let runnerUp: String?
}

private enum StemBundleRole: CaseIterable {
    case vocals
    case instrumental

    var assetKey: String {
        switch self {
        case .vocals: return "vocals"
        case .instrumental: return "instrumental"
        }
    }

    var includeName: String {
        assetKey
    }

    var bundleFileName: String {
        switch self {
        case .vocals:
            return TrainingDatasetPackageWriter.vocalBundleFileName
        case .instrumental:
            return TrainingDatasetPackageWriter.instrumentalBundleFileName
        }
    }

    var trainingLyrics: String {
        switch self {
        case .vocals:
            return "vocal stem"
        case .instrumental:
            return "[Instrumental]"
        }
    }

    var trainingLyricsSource: String {
        switch self {
        case .vocals:
            return "transcript"
        case .instrumental:
            return "constant"
        }
    }
}
