import AVFoundation
import Darwin
import Foundation

enum LocalDatasetEnrichmentStep: String, Equatable {
    case separatingStems
    case exportingVocalsBundle
    case exportingInstrumentalBundle
    case refreshingMetadata

    var title: String {
        switch self {
        case .separatingStems:
            return "Separating stems"
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

struct LocalDatasetEnrichmentCommand: Equatable {
    let step: LocalDatasetEnrichmentStep
    let executableURL: URL
    let arguments: [String]

    var displayCommand: String {
        ([executableURL.path] + arguments).map(Self.shellQuoted).joined(separator: " ")
    }

    private static func shellQuoted(_ value: String) -> String {
        guard value.rangeOfCharacter(from: .whitespacesAndNewlines) != nil else {
            return value
        }
        return "'\(value.replacingOccurrences(of: "'", with: "'\\''"))'"
    }
}

enum LocalDatasetEnrichmentError: LocalizedError {
    case missingAudioSeparatorCLI
    case invalidCaptions(String)
    case missingClip(String)
    case noStemOutput(String)
    case commandFailed(step: LocalDatasetEnrichmentStep, status: Int32, command: String, output: String)

    var errorDescription: String? {
        switch self {
        case .missingAudioSeparatorCLI:
            return """
            audio-separator was not found. Install the audio-separator CLI or set ANVIL_AUDIO_SEPARATOR_BIN to its executable path.
            """
        case .invalidCaptions(let message):
            return "Dataset captions are not valid: \(message)"
        case .missingClip(let path):
            return "Dataset clip is missing: \(path)"
        case .noStemOutput(let message):
            return "Stem separation did not produce the expected files: \(message)"
        case .commandFailed(let step, let status, let command, let output):
            let detail = output.trimmingCharacters(in: .whitespacesAndNewlines)
            let clipped = String(detail.prefix(1_200))
            return """
            \(step.title) failed with exit code \(status).

            \(command)

            \(clipped)
            """
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
        onStep: @escaping (LocalDatasetEnrichmentStep) async -> Void
    ) async throws -> LocalDatasetEnrichmentResult {
        let didStartAccessing = datasetDirectory.startAccessingSecurityScopedResource()
        defer {
            if didStartAccessing {
                datasetDirectory.stopAccessingSecurityScopedResource()
            }
        }

        var records = try loadCaptionRecords(from: datasetDirectory)
        let needsSeparator = force || records.contains {
            !hasCachedInstrumentalSeparation(record: $0.object, datasetDirectory: datasetDirectory)
        }
        let separatorURL = needsSeparator ? try resolveAudioSeparatorExecutableURL() : nil
        let separatorVersion = separatorURL.flatMap { audioSeparatorVersion(executableURL: $0) } ?? "cached"

        await onStep(.separatingStems)
        for index in records.indices {
            try Task.checkCancellation()
            if !force, hasCachedInstrumentalSeparation(record: records[index].object, datasetDirectory: datasetDirectory) {
                continue
            }
            guard let separatorURL else {
                throw LocalDatasetEnrichmentError.missingAudioSeparatorCLI
            }
            records[index].object = try await separateClip(
                record: records[index],
                datasetDirectory: datasetDirectory,
                separatorURL: separatorURL,
                separatorVersion: separatorVersion,
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

    nonisolated static func separatorCommand(
        executableURL: URL,
        inputURL: URL,
        outputDirectory: URL
    ) -> LocalDatasetEnrichmentCommand {
        LocalDatasetEnrichmentCommand(
            step: .separatingStems,
            executableURL: executableURL,
            arguments: [
                inputURL.path,
                "--output_dir",
                outputDirectory.path,
                "--output_format",
                "WAV",
            ]
        )
    }

    nonisolated static func resolveAudioSeparatorExecutableURL(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> URL {
        for url in candidateAudioSeparatorExecutableURLs(environment: environment)
            where FileManager.default.isExecutableFile(atPath: url.path) {
            return url
        }
        throw LocalDatasetEnrichmentError.missingAudioSeparatorCLI
    }

    nonisolated static func candidateAudioSeparatorExecutableURLs(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> [URL] {
        var candidates: [URL] = []
        if let configured = environment["ANVIL_AUDIO_SEPARATOR_BIN"]?.trimmingCharacters(in: .whitespacesAndNewlines),
           !configured.isEmpty {
            candidates.append(URL(fileURLWithPath: (configured as NSString).expandingTildeInPath))
        }

        let searchPath = separatorSearchPath(environment: environment)
        candidates.append(contentsOf: searchPath.map {
            URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath)
                .appendingPathComponent("audio-separator")
        })
        return candidates.deduplicatedByPath()
    }

    nonisolated private static func separateClip(
        record: LocalDatasetCaptionRecord,
        datasetDirectory: URL,
        separatorURL: URL,
        separatorVersion: String,
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
        let tempOutputDirectory = outputDirectory.appendingPathComponent(".audio_separator_tmp", isDirectory: true)
        if FileManager.default.fileExists(atPath: tempOutputDirectory.path) {
            try FileManager.default.removeItem(at: tempOutputDirectory)
        }
        try FileManager.default.createDirectory(at: tempOutputDirectory, withIntermediateDirectories: true)

        let startedAt = Date()
        let command = separatorCommand(
            executableURL: separatorURL,
            inputURL: input.url,
            outputDirectory: tempOutputDirectory
        )
        let runner = LocalDatasetCapturedProcess(
            command: command,
            environment: separatorEnvironment(separatorURL: separatorURL)
        )
        _ = try await runner.run()

        let copiedStems = try canonicalizeStemOutputs(
            from: tempOutputDirectory,
            to: outputDirectory
        )
        try? FileManager.default.removeItem(at: tempOutputDirectory)

        let elapsed = Date().timeIntervalSince(startedAt)
        let warnings = input.warning.map { [$0] } ?? []
        let payload = separationPayload(
            datasetDirectory: datasetDirectory,
            sourceClip: record.file,
            backendVersion: separatorVersion,
            elapsedSeconds: elapsed,
            stems: copiedStems,
            warnings: warnings
        )
        let separationURL = outputDirectory.appendingPathComponent("separation.json")
        try writeJSONObject(payload, to: separationURL)

        var sidecar = readJSONObject(from: clipURL.deletingPathExtension().appendingPathExtension("json")) ?? [:]
        sidecar["separation"] = payload
        try writeJSONObject(sidecar, to: clipURL.deletingPathExtension().appendingPathExtension("json"))

        var updated = record.object
        let stemPaths = copiedStems.mapValues { relativePath(for: $0, from: datasetDirectory) }
        updated["separation"] = [
            "mode": "instrumental",
            "backend": "audio-separator",
            "model": "auto",
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

    nonisolated private static func canonicalizeStemOutputs(
        from tempOutputDirectory: URL,
        to outputDirectory: URL
    ) throws -> [String: URL] {
        let files = (try? FileManager.default.contentsOfDirectory(
            at: tempOutputDirectory,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        )) ?? []

        guard let vocals = findStemFile(named: "vocals", in: files),
              let instrumental = findStemFile(named: "instrumental", in: files) else {
            let names = files.map(\.lastPathComponent).joined(separator: ", ")
            throw LocalDatasetEnrichmentError.noStemOutput(names.isEmpty ? "no files were written" : names)
        }

        let targets = [
            "vocals": outputDirectory.appendingPathComponent("vocals.wav"),
            "instrumental": outputDirectory.appendingPathComponent("instrumental.wav"),
        ]
        try copyStem(vocals, to: targets["vocals"]!)
        try copyStem(instrumental, to: targets["instrumental"]!)
        return targets
    }

    nonisolated private static func findStemFile(named stem: String, in files: [URL]) -> URL? {
        let aliases: [String]
        switch stem {
        case "vocals":
            aliases = ["vocals", "vocal", "voice"]
        case "instrumental":
            aliases = ["instrumental", "inst", "no_vocals", "no-vocals"]
        default:
            aliases = [stem]
        }

        return files.first { file in
            let base = file.deletingPathExtension().lastPathComponent.lowercased()
            return aliases.contains { base.contains($0) }
        }
    }

    nonisolated private static func copyStem(_ source: URL, to target: URL) throws {
        if FileManager.default.fileExists(atPath: target.path) {
            try FileManager.default.removeItem(at: target)
        }
        try FileManager.default.copyItem(at: source, to: target)
    }

    nonisolated private static func separationPayload(
        datasetDirectory: URL,
        sourceClip: String,
        backendVersion: String,
        elapsedSeconds: TimeInterval,
        stems: [String: URL],
        warnings: [String]
    ) -> [String: Any] {
        [
            "anvil_dataset_version": "1.0",
            "generated_at": ISO8601DateFormatter().string(from: Date()),
            "backend": "audio-separator",
            "backend_version": backendVersion,
            "model": "auto",
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

    nonisolated private static func audioSeparatorVersion(executableURL: URL) -> String? {
        let process = Process()
        let pipe = Pipe()
        process.executableURL = executableURL
        process.arguments = ["--version"]
        process.standardOutput = pipe
        process.standardError = Pipe()
        process.environment = separatorEnvironment(separatorURL: executableURL)
        do {
            try process.run()
            process.waitUntilExit()
        } catch {
            return nil
        }
        guard process.terminationStatus == 0 else { return nil }
        return String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8)?
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    nonisolated private static func separatorEnvironment(
        separatorURL: URL,
        base: [String: String] = ProcessInfo.processInfo.environment
    ) -> [String: String] {
        var environment = base
        environment["PATH"] = ([separatorURL.deletingLastPathComponent().path] + separatorSearchPath(environment: base))
            .deduplicated()
            .joined(separator: ":")
        environment["PYTHONUNBUFFERED"] = "1"
        environment["NO_COLOR"] = "1"
        environment["CLICOLOR"] = "0"
        environment["TERM"] = "dumb"
        return environment
    }

    nonisolated private static func separatorSearchPath(environment: [String: String]) -> [String] {
        let pathParts = (environment["PATH"] ?? "")
            .split(separator: ":")
            .map(String.init)
        return [
            "/opt/homebrew/bin",
            "/usr/local/bin",
            "/opt/local/bin",
            "/usr/bin",
            "/bin",
            "/usr/sbin",
            "/sbin",
        ] + pathParts
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

private final class LocalDatasetCapturedProcess: @unchecked Sendable {
    private let process = Process()
    private let stdout = Pipe()
    private let stderr = Pipe()
    private let lock = NSLock()
    private let command: LocalDatasetEnrichmentCommand
    private var stdoutData = Data()
    private var stderrData = Data()
    private var continuation: CheckedContinuation<String, Error>?
    private var didResume = false
    private var didCancel = false

    init(command: LocalDatasetEnrichmentCommand, environment: [String: String]) {
        self.command = command
        process.executableURL = command.executableURL
        process.arguments = command.arguments
        process.environment = environment
        process.standardOutput = stdout
        process.standardError = stderr
    }

    func run() async throws -> String {
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<String, Error>) in
                start(continuation: continuation)
            }
        } onCancel: {
            cancel()
        }
    }

    private func start(continuation: CheckedContinuation<String, Error>) {
        lock.lock()
        if didCancel {
            lock.unlock()
            continuation.resume(throwing: CancellationError())
            return
        }
        self.continuation = continuation
        lock.unlock()

        stdout.fileHandleForReading.readabilityHandler = { [weak self] handle in
            self?.appendStdout(handle.availableData)
        }
        stderr.fileHandleForReading.readabilityHandler = { [weak self] handle in
            self?.appendStderr(handle.availableData)
        }
        process.terminationHandler = { [weak self] process in
            self?.finish(process: process)
        }

        do {
            try process.run()
        } catch {
            finish(error: error)
        }
    }

    private func appendStdout(_ data: Data) {
        guard !data.isEmpty else { return }
        lock.lock()
        stdoutData.append(data)
        lock.unlock()
    }

    private func appendStderr(_ data: Data) {
        guard !data.isEmpty else { return }
        lock.lock()
        stderrData.append(data)
        lock.unlock()
    }

    private func finish(process: Process) {
        stdout.fileHandleForReading.readabilityHandler = nil
        stderr.fileHandleForReading.readabilityHandler = nil
        let remainingStdout = stdout.fileHandleForReading.readDataToEndOfFile()
        let remainingStderr = stderr.fileHandleForReading.readDataToEndOfFile()

        lock.lock()
        stdoutData.append(remainingStdout)
        stderrData.append(remainingStderr)
        guard !didResume, let continuation else {
            lock.unlock()
            return
        }
        didResume = true
        self.continuation = nil
        let outData = stdoutData
        let errData = stderrData
        lock.unlock()

        let output = [
            String(data: outData, encoding: .utf8) ?? "",
            String(data: errData, encoding: .utf8) ?? "",
        ]
        .filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
        .joined(separator: "\n")

        if process.terminationStatus == 0 {
            continuation.resume(returning: output)
        } else {
            continuation.resume(
                throwing: LocalDatasetEnrichmentError.commandFailed(
                    step: command.step,
                    status: process.terminationStatus,
                    command: command.displayCommand,
                    output: output
                )
            )
        }
    }

    private func finish(error: Error) {
        stdout.fileHandleForReading.readabilityHandler = nil
        stderr.fileHandleForReading.readabilityHandler = nil

        lock.lock()
        guard !didResume, let continuation else {
            lock.unlock()
            return
        }
        didResume = true
        self.continuation = nil
        lock.unlock()

        continuation.resume(throwing: error)
    }

    private func cancel() {
        stdout.fileHandleForReading.readabilityHandler = nil
        stderr.fileHandleForReading.readabilityHandler = nil

        lock.lock()
        didCancel = true
        let continuation = didResume ? nil : self.continuation
        if continuation != nil {
            didResume = true
            self.continuation = nil
        }
        lock.unlock()

        if process.isRunning {
            let pid = process.processIdentifier
            process.terminate()
            DispatchQueue.global(qos: .utility).asyncAfter(deadline: .now() + 2.0) { [process] in
                if process.isRunning {
                    kill(pid, SIGKILL)
                }
            }
        }

        continuation?.resume(throwing: CancellationError())
    }
}

private extension Array where Element == URL {
    func deduplicatedByPath() -> [URL] {
        var seen = Set<String>()
        return filter { url in
            seen.insert(url.standardizedFileURL.path).inserted
        }
    }
}

private extension Array where Element == String {
    func deduplicated() -> [String] {
        var seen = Set<String>()
        return filter { seen.insert($0).inserted }
    }
}
