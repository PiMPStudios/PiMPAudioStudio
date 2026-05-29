import AVFoundation
import Foundation
import AnvilCore

enum LocalDatasetEnrichmentStep: String, Equatable {
    case separatingStems
    case transcribingVocals
    case exportingVocalsBundle
    case exportingInstrumentalBundle
    case refreshingMetadata

    var title: String {
        switch self {
        case .separatingStems:
            return "Separating stems"
        case .transcribingVocals:
            return "Transcribing vocals"
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
    let transcribedClipCount: Int

    var summary: String {
        var parts: [String] = []
        parts.append("Separated \(separatedClipCount) clip\(separatedClipCount == 1 ? "" : "s")")
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
        parts.append("key analysis pending")
        return parts.joined(separator: "; ") + "."
    }
}

enum LocalDatasetEnrichmentError: LocalizedError {
    case invalidCaptions(String)
    case missingClip(String)

    var errorDescription: String? {
        switch self {
        case .invalidCaptions(let message):
            return "Dataset captions are not valid: \(message)"
        case .missingClip(let path):
            return "Dataset clip is missing: \(path)"
        }
    }
}

@MainActor
final class LocalDatasetEnrichmentService: ObservableObject {
    @Published var phase: LocalDatasetEnrichmentPhase = .idle

    private var task: Task<Void, Never>?

    func enrich(datasetDirectory: URL, force: Bool = false) {
        cancel()
        phase = .running(step: .separatingStems)

        task = Task { [weak self] in
            guard let self else { return }
            do {
                let result = try await Self.run(
                    datasetDirectory: datasetDirectory,
                    force: force,
                    onStep: { [weak self] step in
                        await MainActor.run {
                            self?.phase = .running(step: step)
                        }
                    }
                )
                phase = .complete(result)
            } catch is CancellationError {
                phase = .idle
            } catch {
                phase = .failed(error.localizedDescription)
            }
        }
    }

    func cancel() {
        task?.cancel()
        task = nil
    }

    func reset() {
        cancel()
        phase = .idle
    }

    nonisolated static func run(
        datasetDirectory: URL,
        force: Bool = false,
        lyricsTranscriber: any DatasetLyricsTranscribing = LyricsTranscriptionService(),
        onStep: @escaping (LocalDatasetEnrichmentStep) async -> Void
    ) async throws -> LocalDatasetEnrichmentResult {
        let didStartAccessing = datasetDirectory.startAccessingSecurityScopedResource()
        defer {
            if didStartAccessing {
                datasetDirectory.stopAccessingSecurityScopedResource()
            }
        }

        var records = try loadCaptionRecords(from: datasetDirectory)
        let sourceSeparator = SourceSeparationService()

        await onStep(.separatingStems)
        for index in records.indices {
            try Task.checkCancellation()
            if !force, hasCachedInstrumentalSeparation(record: records[index].object, datasetDirectory: datasetDirectory) {
                continue
            }
            records[index].object = try await separateClip(
                record: records[index],
                datasetDirectory: datasetDirectory,
                sourceSeparator: sourceSeparator,
                force: force
            )
        }

        await onStep(.transcribingVocals)
        for index in records.indices {
            try Task.checkCancellation()
            records[index].object = try await transcribeClipVocals(
                record: records[index],
                datasetDirectory: datasetDirectory,
                lyricsTranscriber: lyricsTranscriber,
                force: force
            )
        }

        try writeJSONObjectArray(
            records.map(\.object),
            to: datasetDirectory.appendingPathComponent(TrainingDatasetPackageWriter.captionsFileName)
        )

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

        try Task.checkCancellation()
        await onStep(.refreshingMetadata)
        let result = inspectResult(datasetDirectory: datasetDirectory)
        try refreshDatasetMetadata(datasetDirectory: datasetDirectory, result: result)
        try TrainingDatasetPackageWriter.refreshValidationReport(at: datasetDirectory)
        return inspectResult(datasetDirectory: datasetDirectory)
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
        return (vocalCount, instrumentalCount)
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
        force: Bool
    ) async throws -> [String: Any] {
        guard let vocalPath = stemPath(for: .vocals, in: record.object) else {
            return record.object
        }
        if !force, hasCompletedTranscription(record.object) {
            return record.object
        }

        let vocalURL = resolve(vocalPath, relativeTo: datasetDirectory)
        guard FileManager.default.fileExists(atPath: vocalURL.path(percentEncoded: false)) else {
            throw LocalDatasetEnrichmentError.missingClip(vocalPath)
        }

        let outputDirectory = vocalURL.deletingLastPathComponent()
        let transcription = try await lyricsTranscriber.transcribe(
            vocalsURL: vocalURL,
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

        let clipURL = resolve(record.file, relativeTo: datasetDirectory)
        var sidecar = readJSONObject(from: clipURL.deletingPathExtension().appendingPathExtension("json")) ?? [:]
        sidecar["transcription"] = payload
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
            "anvil_dataset_version": "1.0",
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
                    "analysis": analyzeAudioFile(item.value),
                ]
            },
        ]
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
            var assets = record["assets"] as? [String: Any] ?? [:]
            assets[role.assetKey] = stemPath
            if let file = record["file"] as? String {
                assets["full_mix"] = file
            }

            var clip: [String: Any] = [
                "id": record["id"] as? String ?? String(format: "clip_%04d", index + 1),
                "file": stemPath,
                "assets": assets,
                "caption": prompt,
                "prompt": prompt,
                "lyrics": role == .vocals ? transcript : "",
                "lyrics_source": role == .vocals && !transcript.isEmpty ? "transcript" : role.trainingLyricsSource,
                "transcript": transcript,
                "transcription": record["transcription"] as? [String: Any] ?? [
                    "status": transcript.isEmpty ? "not_run" : "complete",
                    "text": transcript,
                ],
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
            "anvil_training_bundle_version": "1.0",
            "created_at": createdAt,
            "dataset_dir": ".",
            "include": [role.includeName],
            "training_lyrics": role.trainingLyrics,
            "training_lyrics_source": role.trainingLyricsSource,
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
        let now = ISO8601DateFormatter().string(from: Date())
        let captions = readJSONArray(from: datasetDirectory.appendingPathComponent(TrainingDatasetPackageWriter.captionsFileName))

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

        var analysis = manifest["analysis"] as? [String: Any] ?? [:]
        analysis["enabled"] = result.bpmEstimateCount > 0
        analysis["status"] = result.bpmEstimateCount > 0 ? "stem_analysis_available" : (analysis["status"] ?? "not_run")
        analysis["bpm_status"] = result.bpmEstimateCount > 0 ? "estimated_from_stems" : "not_run"
        analysis["key_status"] = "not_run"
        if let averageBPM = result.averageBPM {
            analysis["stem_tempo_bpm"] = [
                "average": averageBPM,
                "estimate_count": result.bpmEstimateCount,
            ]
        }
        manifest["analysis"] = analysis

        var training = manifest["training"] as? [String: Any] ?? [:]
        training["default_bundle"] = TrainingDatasetPackageWriter.trainingBundleFileName
        training["full_mix_bundle"] = TrainingDatasetPackageWriter.fullMixBundleFileName
        training["vocal_bundle"] = TrainingDatasetPackageWriter.vocalBundleFileName
        training["instrumental_bundle"] = TrainingDatasetPackageWriter.instrumentalBundleFileName
        training["dataset_config"] = TrainingDatasetPackageWriter.datasetConfigFileName
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
            "key_status": "not_run",
        ]

        try writeJSONObject(manifest, to: manifestURL)
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
            transcribedClipCount: transcribedClipCount
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

    nonisolated private static func analyzeAudioFile(_ url: URL) -> [String: Any] {
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
            if let bpm = estimateTempoBPM(from: url) {
                object["tempo_bpm_estimate"] = bpm
            }
        } catch {
            object["metadata_status"] = "unreadable_audio_header"
            object["metadata_error"] = error.localizedDescription
        }
        return object
    }

    nonisolated private static func estimateTempoBPM(from url: URL) -> Double? {
        guard let samples = try? monoSamples(from: url), samples.samples.count > 4_096 else {
            return nil
        }
        let sampleRate = samples.sampleRate
        let values = samples.samples
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

        var bestLag = minLag
        var bestScore = 0.0
        for lag in minLag...maxLag {
            var score = 0.0
            for i in lag..<centered.count {
                score += centered[i] * centered[i - lag]
            }
            if score > bestScore {
                bestScore = score
                bestLag = lag
            }
        }
        guard bestScore > 0 else { return nil }
        return rounded(60.0 * envelopeRate / Double(bestLag), places: 1)
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

actor LyricsTranscriptionService: DatasetLyricsTranscribing {
    private let transcriber: QwenLyricsTranscriber
    private let options: LyricsTranscriptionOptions

    init(
        transcriber: QwenLyricsTranscriber = QwenLyricsTranscriber(),
        options: LyricsTranscriptionOptions = .qualityDataset
    ) {
        self.transcriber = transcriber
        self.options = options
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

    private func timedLyricsPayload(result: LyricsTranscriptionResult, vocalsURL: URL) -> [String: Any] {
        var payload: [String: Any] = [
            "anvil_lyrics_transcription_version": "1.0",
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
            "anvil_lyrics_transcription_version": "1.0",
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
