import AVFoundation
import Foundation

struct TrainingDatasetPackageValidationResult: Equatable {
    let packageDirectory: URL
    let checkedAt: String
    let clipCount: Int
    let importableClipCount: Int
    let missingPackageFiles: [String]
    let missingClipFiles: [String]
    let errors: [String]

    var isValid: Bool {
        missingPackageFiles.isEmpty && missingClipFiles.isEmpty && errors.isEmpty && clipCount > 0
    }

    var summary: String {
        if isValid {
            return "\(clipCount) clips validated for package export."
        }
        let issueCount = missingPackageFiles.count + missingClipFiles.count + errors.count
        return "\(issueCount) package issue\(issueCount == 1 ? "" : "s") found."
    }
}

enum TrainingDatasetPackageError: LocalizedError {
    case invalidPackage(String)
    case exportFailed(String)

    var errorDescription: String? {
        switch self {
        case .invalidPackage(let message):
            return "Dataset package is not ready to export: \(message)"
        case .exportFailed(let message):
            return "Dataset package export failed: \(message)"
        }
    }
}

enum TrainingDatasetPackageWriter {
    static let captionsFileName = "captions.json"
    static let datasetConfigFileName = "dataset_config.json"
    static let datasetManifestFileName = "dataset_manifest.json"
    static let trainingBundleFileName = "training_bundle.json"
    static let fullMixBundleFileName = "training_bundle_full_mix.json"
    static let vocalBundleFileName = "training_bundle_vocals.json"
    static let instrumentalBundleFileName = "training_bundle_instrumental.json"
    static let remotePackageFileName = "remote_training_package.json"
    static let validationReportFileName = "package_validation.json"

    private struct AudioFileMetadata {
        let format: String
        let byteSize: Int64?
        let durationSeconds: Double?
        let sampleRate: Int?
        let channels: Int?
        let frameCount: Int64?
        let status: String
        let errorDescription: String?

        var jsonObject: [String: Any] {
            var object: [String: Any] = [
                "format": format,
                "metadata_status": status,
            ]
            if let byteSize {
                object["byte_size"] = byteSize
            }
            if let durationSeconds {
                object["duration_seconds"] = seconds(durationSeconds)
            }
            if let sampleRate {
                object["sample_rate"] = sampleRate
            }
            if let channels {
                object["channels"] = channels
            }
            if let frameCount {
                object["frame_count"] = frameCount
            }
            if let errorDescription, !errorDescription.isEmpty {
                object["metadata_error"] = errorDescription
            }
            return object
        }
    }

    static func writeLocalPackage(
        outputDirectory: URL,
        clips: [TrainingImportedFile],
        sourceFiles: [URL],
        config: LocalFilesDatasetConfig,
        styleTemplate: String,
        createdAt: Date = Date()
    ) throws {
        try FileManager.default.createDirectory(at: outputDirectory, withIntermediateDirectories: true)

        let existingCaptionRecords = captionRecordsByFile(from: outputDirectory)
        try writeCaptionsFile(
            clips: clips.map { (url: $0.url, caption: normalizedCaption(for: $0)) },
            to: outputDirectory,
            existingRecordsByFile: existingCaptionRecords
        )

        let clipAudioMetadata = audioMetadataByPath(for: clips.map(\.url))
        let sourceAudioMetadata = audioMetadataByPath(for: sourceFiles)
        let clipEntries = packageClips(
            from: clips,
            sourceFiles: sourceFiles,
            outputDirectory: outputDirectory,
            config: config,
            audioMetadata: clipAudioMetadata,
            existingCaptionRecordsByFile: existingCaptionRecords
        )
        let createdAtString = ISO8601DateFormatter().string(from: createdAt)
        let trimmedStyle = styleTemplate.trimmingCharacters(in: .whitespacesAndNewlines)

        try writeJSONObject(
            datasetConfig(),
            to: outputDirectory.appendingPathComponent(datasetConfigFileName)
        )
        try writeJSONObject(
            datasetManifest(
                outputDirectory: outputDirectory,
                clips: clipEntries,
                sourceFiles: sourceFiles,
                clipAudioMetadata: clips.map { clipAudioMetadata[normalizedFilePath($0.url)] ?? fallbackAudioMetadata(for: $0.url) },
                sourceAudioMetadata: sourceAudioMetadata,
                config: config,
                styleTemplate: trimmedStyle,
                createdAt: createdAtString
            ),
            to: outputDirectory.appendingPathComponent(datasetManifestFileName)
        )

        let bundle = trainingBundle(clips: clipEntries, createdAt: createdAtString)
        try writeJSONObject(bundle, to: outputDirectory.appendingPathComponent(trainingBundleFileName))
        try writeJSONObject(bundle, to: outputDirectory.appendingPathComponent(fullMixBundleFileName))
        try writeJSONObject(
            remotePackageDescriptor(
                outputDirectory: outputDirectory,
                clipCount: clips.count,
                createdAt: createdAtString
            ),
            to: outputDirectory.appendingPathComponent(remotePackageFileName)
        )

        let validation = validatePackage(at: outputDirectory, checkedAt: createdAt)
        try writeValidationReport(validation, to: outputDirectory)
    }

    static func validatePackage(
        at outputDirectory: URL,
        checkedAt: Date = Date()
    ) -> TrainingDatasetPackageValidationResult {
        let checkedAtString = ISO8601DateFormatter().string(from: checkedAt)
        let requiredFiles = [
            captionsFileName,
            datasetConfigFileName,
            datasetManifestFileName,
            trainingBundleFileName,
            fullMixBundleFileName,
            remotePackageFileName,
        ]
        let missingPackageFiles = requiredFiles.filter {
            !FileManager.default.fileExists(atPath: outputDirectory.appendingPathComponent($0).path)
        }

        let bundleURL = outputDirectory.appendingPathComponent(fullMixBundleFileName)
        let bundle = readJSONObject(from: bundleURL)
        let clips = bundle?["clips"] as? [[String: Any]] ?? []
        let datasetBase = datasetBaseDirectory(from: bundle, fallback: outputDirectory)
        let missingClipFiles = clips.compactMap { clip -> String? in
            guard let path = selectedAudioPath(from: clip) else {
                return "missing audio path"
            }
            let audioURL = resolve(path, relativeTo: datasetBase)
            return FileManager.default.fileExists(atPath: audioURL.path(percentEncoded: false)) ? nil : path
        }

        var errors: [String] = []
        if bundle == nil {
            errors.append("\(fullMixBundleFileName) is missing or invalid JSON.")
        }
        if clips.isEmpty {
            errors.append("\(fullMixBundleFileName) does not contain any clips.")
        }

        let importableClipCount = (try? TrainingDatasetImporter.importFiles(from: bundleURL).count) ?? 0
        if importableClipCount != clips.count {
            errors.append("\(importableClipCount) of \(clips.count) clips are importable by the training importer.")
        }

        return TrainingDatasetPackageValidationResult(
            packageDirectory: outputDirectory,
            checkedAt: checkedAtString,
            clipCount: clips.count,
            importableClipCount: importableClipCount,
            missingPackageFiles: missingPackageFiles,
            missingClipFiles: missingClipFiles,
            errors: errors
        )
    }

    static func refreshValidationReport(
        at outputDirectory: URL,
        checkedAt: Date = Date()
    ) throws {
        try refreshDerivedManifestMetadata(at: outputDirectory, updatedAt: checkedAt)
        let validation = validatePackage(at: outputDirectory, checkedAt: checkedAt)
        try writeValidationReport(validation, to: outputDirectory)
    }

    @discardableResult
    static func exportPackageArchive(outputDirectory: URL, to destinationURL: URL) throws -> URL {
        let validation = validatePackage(at: outputDirectory)
        guard validation.isValid else {
            throw TrainingDatasetPackageError.invalidPackage(validation.summary)
        }

        if FileManager.default.fileExists(atPath: destinationURL.path) {
            try FileManager.default.removeItem(at: destinationURL)
        }

        let process = Process()
        let stderr = Pipe()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/ditto")
        process.arguments = [
            "-c",
            "-k",
            "--keepParent",
            outputDirectory.path,
            destinationURL.path,
        ]
        process.standardError = stderr
        try process.run()
        process.waitUntilExit()

        guard process.terminationStatus == 0 else {
            let data = stderr.fileHandleForReading.readDataToEndOfFile()
            let message = String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines)
            let detail = message?.isEmpty == false ? message ?? "" : "ditto exited with status \(process.terminationStatus)"
            throw TrainingDatasetPackageError.exportFailed(detail)
        }

        return destinationURL
    }

    private static func datasetConfig() -> [String: Any] {
        [
            "anvil_dataset_config_version": "1.0",
            "dataset_type": "audio_dir",
            "datasets": [
                [
                    "path": "clips",
                    "caption_file": captionsFileName,
                    "audio_role": "full_mix",
                ],
            ],
            "random_crop": true,
        ]
    }

    private static func datasetManifest(
        outputDirectory: URL,
        clips: [[String: Any]],
        sourceFiles: [URL],
        clipAudioMetadata: [AudioFileMetadata],
        sourceAudioMetadata: [String: AudioFileMetadata],
        config: LocalFilesDatasetConfig,
        styleTemplate: String,
        createdAt: String
    ) -> [String: Any] {
        let audioSummary = audioSummary(
            from: clipAudioMetadata,
            clipCount: clips.count,
            fallbackClipLength: config.clipLength
        )
        return [
            "anvil_dataset_version": "1.0",
            "name": outputDirectory.lastPathComponent,
            "created_at": createdAt,
            "source": [
                "kind": "local_files",
                "files": sourceFiles.map {
                    sourceFileMetadata(
                        for: $0,
                        metadata: sourceAudioMetadata[normalizedFilePath($0)]
                    )
                },
            ],
            "caption_mode": styleTemplate.isEmpty ? "filename" : "style_template",
            "style_hint": styleTemplate,
            "clip_count": clips.count,
            "clip_length_seconds": seconds(config.clipLength),
            "estimated_total_seconds": seconds(Double(clips.count) * config.clipLength),
            "audio": audioSummary,
            "separation": [
                "enabled": false,
                "status": "not_run",
                "available_assets": ["full_mix"],
            ],
            "transcription": [
                "enabled": false,
                "status": "not_run",
            ],
            "analysis": [
                "enabled": false,
                "status": audioSummary["metadata_status"] as? String ?? "audio_properties_unavailable",
                "audio_properties": audioSummary["metadata_status"] as? String ?? "audio_properties_unavailable",
                "bpm_status": "not_run",
                "key_status": "not_run",
            ],
            "training": [
                "default_bundle": trainingBundleFileName,
                "full_mix_bundle": fullMixBundleFileName,
                "dataset_config": datasetConfigFileName,
            ],
            "remote_packaging": [
                "status": "ready_for_upload",
                "descriptor": remotePackageFileName,
                "provider_targets": ["runpod", "digitalocean_gpu"],
            ],
            "clips": clips,
        ]
    }

    private static func trainingBundle(clips: [[String: Any]], createdAt: String) -> [String: Any] {
        [
            "anvil_training_bundle_version": "1.0",
            "created_at": createdAt,
            "dataset_dir": ".",
            "include": ["full_mix"],
            "training_lyrics": "[Instrumental]",
            "training_lyrics_source": "constant",
            "clip_count": clips.count,
            "clips": clips,
        ]
    }

    private static func remotePackageDescriptor(
        outputDirectory: URL,
        clipCount: Int,
        createdAt: String
    ) -> [String: Any] {
        [
            "anvil_remote_package_version": "1.0",
            "created_at": createdAt,
            "package_name": outputDirectory.lastPathComponent,
            "package_root": ".",
            "clip_count": clipCount,
            "default_bundle": fullMixBundleFileName,
            "bundles": [
                [
                    "name": "full_mix",
                    "path": fullMixBundleFileName,
                    "include": ["full_mix"],
                    "training_lyrics": "[Instrumental]",
                    "training_lyrics_source": "constant",
                ],
                [
                    "name": "vocals",
                    "path": vocalBundleFileName,
                    "include": ["vocals"],
                    "training_lyrics": "vocal stem",
                    "training_lyrics_source": "transcript",
                    "optional": true,
                ],
                [
                    "name": "instrumental",
                    "path": instrumentalBundleFileName,
                    "include": ["instrumental"],
                    "training_lyrics": "[Instrumental]",
                    "training_lyrics_source": "constant",
                    "optional": true,
                ],
            ],
            "provider_targets": ["runpod", "digitalocean_gpu"],
            "copy_rules": [
                "include": [
                    "clips/**",
                    "stems/**",
                    captionsFileName,
                    datasetConfigFileName,
                    datasetManifestFileName,
                    trainingBundleFileName,
                    fullMixBundleFileName,
                    vocalBundleFileName,
                    instrumentalBundleFileName,
                    remotePackageFileName,
                    validationReportFileName,
                ],
                "exclude": [],
            ],
            "worker_contract": [
                "resolve_relative_paths_from": "package_root",
                "entry_bundle": fullMixBundleFileName,
                "secrets_required": [],
                "expected_gpu_class": "h100_or_h200",
            ],
            "notes": [
                "Local-file packages start with full-mix clips. Stem bundles are optional and are written after local enrichment runs.",
                "Remote workers should copy the package root intact and resolve clip assets relative to the selected bundle file.",
            ],
        ]
    }

    private static func packageClips(
        from clips: [TrainingImportedFile],
        sourceFiles: [URL],
        outputDirectory: URL,
        config: LocalFilesDatasetConfig,
        audioMetadata: [String: AudioFileMetadata],
        existingCaptionRecordsByFile: [String: [String: Any]]
    ) -> [[String: Any]] {
        clips.enumerated().map { index, clip in
            let relativePath = relativeClipPath(for: clip.url, from: outputDirectory)
            let caption = normalizedCaption(for: clip)
            let metadata = audioMetadata[normalizedFilePath(clip.url)] ?? fallbackAudioMetadata(for: clip.url)
            var object: [String: Any] = [
                "id": String(format: "clip_%04d", index + 1),
                "file": relativePath,
                "assets": [
                    "full_mix": relativePath,
                ],
                "caption": caption,
                "prompt": caption,
                "lyrics": "",
                "transcript": "",
                "transcription": [
                    "status": "not_run",
                    "text": "",
                ],
                "analysis": [
                    "status": "not_run",
                ],
                "audio": metadata.jsonObject,
                "clip_length_seconds": metadata.durationSeconds.map(seconds) ?? seconds(config.clipLength),
                "source": sourceMetadata(for: clip.url, sourceFiles: sourceFiles),
                "tags": [],
                "negative_tags": [],
            ]
            preserveEnrichmentMetadata(
                from: existingCaptionRecordsByFile[relativePath],
                into: &object
            )
            object["caption"] = caption
            object["prompt"] = caption
            object["file"] = relativePath
            object["assets"] = [
                "full_mix": relativePath,
            ]
            return object
        }
    }

    private static func audioMetadataByPath(for urls: [URL]) -> [String: AudioFileMetadata] {
        var metadata: [String: AudioFileMetadata] = [:]
        for url in urls {
            metadata[normalizedFilePath(url)] = audioMetadata(for: url)
        }
        return metadata
    }

    private static func audioMetadata(for url: URL) -> AudioFileMetadata {
        let format = audioFormat(for: url)
        let byteSize = fileByteSize(for: url)
        guard FileManager.default.fileExists(atPath: url.path(percentEncoded: false)) else {
            return AudioFileMetadata(
                format: format,
                byteSize: byteSize,
                durationSeconds: nil,
                sampleRate: nil,
                channels: nil,
                frameCount: nil,
                status: "missing_file",
                errorDescription: nil
            )
        }

        do {
            let audioFile = try AVAudioFile(forReading: url)
            let sampleRate = audioFile.fileFormat.sampleRate
            let frameCount = Int64(audioFile.length)
            return AudioFileMetadata(
                format: format,
                byteSize: byteSize,
                durationSeconds: sampleRate > 0 ? Double(audioFile.length) / sampleRate : nil,
                sampleRate: sampleRate > 0 ? Int(sampleRate.rounded()) : nil,
                channels: Int(audioFile.fileFormat.channelCount),
                frameCount: frameCount,
                status: "ok",
                errorDescription: nil
            )
        } catch {
            return AudioFileMetadata(
                format: format,
                byteSize: byteSize,
                durationSeconds: nil,
                sampleRate: nil,
                channels: nil,
                frameCount: nil,
                status: "unreadable_audio_header",
                errorDescription: error.localizedDescription
            )
        }
    }

    private static func fallbackAudioMetadata(for url: URL) -> AudioFileMetadata {
        AudioFileMetadata(
            format: audioFormat(for: url),
            byteSize: fileByteSize(for: url),
            durationSeconds: nil,
            sampleRate: nil,
            channels: nil,
            frameCount: nil,
            status: "not_collected",
            errorDescription: nil
        )
    }

    private static func audioSummary(
        from metadata: [AudioFileMetadata],
        clipCount: Int,
        fallbackClipLength: TimeInterval
    ) -> [String: Any] {
        let formats = sortedUnique(metadata.map(\.format).filter { !$0.isEmpty })
        let sampleRates = sortedUnique(metadata.compactMap(\.sampleRate))
        let channelCounts = sortedUnique(metadata.compactMap(\.channels))
        let durations = metadata.compactMap(\.durationSeconds)
        let byteSizes = metadata.compactMap(\.byteSize)
        let collectedCount = metadata.filter { $0.status == "ok" }.count
        let status: String
        if metadata.isEmpty || collectedCount == 0 {
            status = "audio_properties_unavailable"
        } else if collectedCount == metadata.count {
            status = "audio_properties_collected"
        } else {
            status = "partial_audio_properties"
        }

        var durationSummary: [String: Any] = [
            "estimated_total_seconds": seconds(Double(clipCount) * fallbackClipLength),
            "known_count": durations.count,
        ]
        if !durations.isEmpty {
            let total = durations.reduce(0, +)
            durationSummary["known_total_seconds"] = seconds(total)
            durationSummary["min_seconds"] = seconds(durations.min() ?? 0)
            durationSummary["max_seconds"] = seconds(durations.max() ?? 0)
            durationSummary["average_seconds"] = seconds(total / Double(durations.count))
        }

        var summary: [String: Any] = [
            "format": formats.count == 1 ? formats[0] : formats.first ?? "m4a",
            "role": "full_mix",
            "sample_rate": sampleRates.count == 1 ? sampleRates[0] : 0,
            "channels": channelCounts.count == 1 ? channelCounts[0] : 0,
            "formats": formats,
            "sample_rates": sampleRates,
            "channel_counts": channelCounts,
            "duration_seconds": durationSummary,
            "metadata_status": status,
        ]
        if !byteSizes.isEmpty {
            summary["total_byte_size"] = byteSizes.reduce(Int64(0), +)
        }
        return summary
    }

    private static func normalizedCaption(for clip: TrainingImportedFile) -> String {
        let trimmed = clip.description.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmed.isEmpty {
            return trimmed
        }
        return clip.url.deletingPathExtension().lastPathComponent
    }

    private static func sourceMetadata(for clipURL: URL, sourceFiles: [URL]) -> [String: Any] {
        var metadata: [String: Any] = [
            "kind": "local_file",
        ]
        if let source = sourceFileName(for: clipURL, sourceFiles: sourceFiles) {
            metadata["file_name"] = source
        }
        return metadata
    }

    private static func sourceFileMetadata(for url: URL, metadata: AudioFileMetadata?) -> [String: Any] {
        var object: [String: Any] = [
            "file_name": url.lastPathComponent,
        ]
        let audio = metadata ?? fallbackAudioMetadata(for: url)
        for (key, value) in audio.jsonObject {
            object[key] = value
        }
        return object
    }

    private static func sourceFileName(for clipURL: URL, sourceFiles: [URL]) -> String? {
        guard !sourceFiles.isEmpty else { return nil }
        if let index = sourceIndex(from: clipURL.lastPathComponent),
           sourceFiles.indices.contains(index - 1) {
            return sourceFiles[index - 1].lastPathComponent
        }
        return sourceFiles.count == 1 ? sourceFiles[0].lastPathComponent : nil
    }

    private static func sourceIndex(from fileName: String) -> Int? {
        guard fileName.first == "f" else { return nil }
        let digits = fileName.dropFirst().prefix { $0.isNumber }
        guard !digits.isEmpty else { return nil }
        return Int(digits)
    }

    private static func relativeClipPath(for url: URL, from baseDirectory: URL) -> String {
        let basePath = normalizedDirectoryPath(baseDirectory)
        let filePath = url.standardizedFileURL.path
        if filePath.hasPrefix(basePath) {
            return String(filePath.dropFirst(basePath.count))
        }
        return "clips/\(url.lastPathComponent)"
    }

    private static func normalizedDirectoryPath(_ url: URL) -> String {
        var path = url.standardizedFileURL.path
        if !path.hasSuffix("/") {
            path.append("/")
        }
        return path
    }

    private static func normalizedFilePath(_ url: URL) -> String {
        url.standardizedFileURL.path
    }

    private static func audioFormat(for url: URL) -> String {
        let format = url.pathExtension.lowercased()
        return format.isEmpty ? "unknown" : format
    }

    private static func fileByteSize(for url: URL) -> Int64? {
        guard let attributes = try? FileManager.default.attributesOfItem(atPath: url.path(percentEncoded: false)),
              let size = attributes[.size] as? NSNumber else {
            return nil
        }
        return size.int64Value
    }

    private static func sortedUnique<T: Comparable & Hashable>(_ values: [T]) -> [T] {
        Array(Set(values)).sorted()
    }

    private static func refreshDerivedManifestMetadata(
        at outputDirectory: URL,
        updatedAt: Date
    ) throws {
        let manifestURL = outputDirectory.appendingPathComponent(datasetManifestFileName)
        guard var manifest = readJSONObject(from: manifestURL) else { return }

        let captions = readJSONArray(from: outputDirectory.appendingPathComponent(captionsFileName))
        guard !captions.isEmpty else { return }

        let separatedClipCount = captions.filter {
            hasExistingStem("vocals", in: $0, outputDirectory: outputDirectory) &&
                hasExistingStem("instrumental", in: $0, outputDirectory: outputDirectory)
        }.count
        let transcribedClipCount = captions.filter { !transcriptionText(from: $0).isEmpty }.count
        let vocalBundleExists = FileManager.default.fileExists(
            atPath: outputDirectory.appendingPathComponent(vocalBundleFileName).path(percentEncoded: false)
        )
        let instrumentalBundleExists = FileManager.default.fileExists(
            atPath: outputDirectory.appendingPathComponent(instrumentalBundleFileName).path(percentEncoded: false)
        )
        let bpmEstimates = stemBPMEstimates(in: captions, outputDirectory: outputDirectory)
        let hasDerivedMetadata = separatedClipCount > 0 || transcribedClipCount > 0 ||
            vocalBundleExists || instrumentalBundleExists || !bpmEstimates.isEmpty
        guard hasDerivedMetadata else { return }

        let updatedAtString = ISO8601DateFormatter().string(from: updatedAt)

        var availableAssets = ["full_mix"]
        if vocalBundleExists || captions.contains(where: { stemPath("vocals", in: $0) != nil }) {
            availableAssets.append("vocals")
        }
        if instrumentalBundleExists || captions.contains(where: { stemPath("instrumental", in: $0) != nil }) {
            availableAssets.append("instrumental")
        }

        var separation = manifest["separation"] as? [String: Any] ?? [:]
        separation["enabled"] = separatedClipCount > 0
        separation["status"] = derivedStatus(completed: separatedClipCount, total: captions.count)
        separation["available_assets"] = availableAssets
        separation["updated_at"] = updatedAtString
        if vocalBundleExists {
            separation["vocal_bundle"] = vocalBundleFileName
        }
        if instrumentalBundleExists {
            separation["instrumental_bundle"] = instrumentalBundleFileName
        }
        manifest["separation"] = separation

        var transcription = manifest["transcription"] as? [String: Any] ?? [:]
        transcription["enabled"] = transcribedClipCount > 0
        transcription["status"] = derivedStatus(
            completed: transcribedClipCount,
            total: separatedClipCount > 0 ? separatedClipCount : captions.count
        )
        transcription["clip_count"] = transcribedClipCount
        transcription["updated_at"] = updatedAtString
        if transcribedClipCount > 0 {
            transcription["artifacts"] = [
                "lyrics.txt",
                "lyrics.timed.json",
                "lyrics.lrc",
            ]
        }
        let transcriptionModels = sortedUnique(captions.compactMap {
            ($0["transcription"] as? [String: Any])?["model"] as? String
        })
        if !transcriptionModels.isEmpty {
            transcription["models"] = transcriptionModels
        }
        let transcriptionBackends = sortedUnique(captions.compactMap {
            ($0["transcription"] as? [String: Any])?["backend"] as? String
        })
        if !transcriptionBackends.isEmpty {
            transcription["backends"] = transcriptionBackends
        }
        manifest["transcription"] = transcription

        var analysis = manifest["analysis"] as? [String: Any] ?? [:]
        analysis["enabled"] = !bpmEstimates.isEmpty
        analysis["status"] = !bpmEstimates.isEmpty ? "stem_analysis_available" : (analysis["status"] ?? "not_run")
        analysis["bpm_status"] = !bpmEstimates.isEmpty ? "estimated_from_stems" : "not_run"
        analysis["key_status"] = "not_run"
        if !bpmEstimates.isEmpty {
            analysis["stem_tempo_bpm"] = [
                "average": rounded(bpmEstimates.reduce(0, +) / Double(bpmEstimates.count), places: 1),
                "estimate_count": bpmEstimates.count,
            ]
        }
        manifest["analysis"] = analysis

        var training = manifest["training"] as? [String: Any] ?? [:]
        training["default_bundle"] = trainingBundleFileName
        training["full_mix_bundle"] = fullMixBundleFileName
        training["dataset_config"] = datasetConfigFileName
        if vocalBundleExists {
            training["vocal_bundle"] = vocalBundleFileName
        }
        if instrumentalBundleExists {
            training["instrumental_bundle"] = instrumentalBundleFileName
        }
        manifest["training"] = training

        manifest["enrichment"] = [
            "updated_at": updatedAtString,
            "status": derivedStatus(completed: separatedClipCount, total: captions.count),
            "separated_clip_count": separatedClipCount,
            "vocal_bundle": vocalBundleExists ? vocalBundleFileName : "",
            "instrumental_bundle": instrumentalBundleExists ? instrumentalBundleFileName : "",
            "transcription_status": derivedStatus(
                completed: transcribedClipCount,
                total: separatedClipCount > 0 ? separatedClipCount : captions.count
            ),
            "transcribed_clip_count": transcribedClipCount,
            "bpm_status": !bpmEstimates.isEmpty ? "estimated_from_stems" : "not_run",
            "key_status": "not_run",
        ]

        try writeJSONObject(manifest, to: manifestURL)
    }

    private static func derivedStatus(completed: Int, total: Int) -> String {
        guard completed > 0 else { return "not_run" }
        return completed >= total ? "complete" : "partial"
    }

    private static func hasExistingStem(
        _ role: String,
        in record: [String: Any],
        outputDirectory: URL
    ) -> Bool {
        guard let path = stemPath(role, in: record) else { return false }
        return FileManager.default.fileExists(
            atPath: resolve(path, relativeTo: outputDirectory).path(percentEncoded: false)
        )
    }

    private static func stemPath(_ role: String, in record: [String: Any]) -> String? {
        if let assets = record["assets"] as? [String: Any],
           let value = assets[role] as? String,
           !value.isEmpty {
            return value
        }
        guard let separation = record["separation"] as? [String: Any],
              let stems = separation["stems"] as? [String: Any],
              let value = stems[role] as? String,
              !value.isEmpty else {
            return nil
        }
        return value
    }

    private static func transcriptionText(from record: [String: Any]) -> String {
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

    private static func stemBPMEstimates(
        in captions: [[String: Any]],
        outputDirectory: URL
    ) -> [Double] {
        captions.flatMap { record -> [Double] in
            guard let separation = record["separation"] as? [String: Any],
                  let separationFile = separation["separation_file"] as? String,
                  !separationFile.isEmpty,
                  let payload = readJSONObject(from: resolve(separationFile, relativeTo: outputDirectory)),
                  let stems = payload["stems"] as? [String: Any] else {
                return []
            }

            return ["vocals", "instrumental"].compactMap { role in
                guard let stem = stems[role] as? [String: Any],
                      let analysis = stem["analysis"] as? [String: Any] else {
                    return nil
                }
                return doubleValue(analysis["tempo_bpm_estimate"])
            }
        }
    }

    private static func doubleValue(_ value: Any?) -> Double? {
        switch value {
        case let number as NSNumber:
            return number.doubleValue
        case let double as Double:
            return double
        case let float as Float:
            return Double(float)
        case let int as Int:
            return Double(int)
        case let string as String:
            return Double(string)
        default:
            return nil
        }
    }

    private static func rounded(_ value: Double, places: Int) -> Double {
        let scale = pow(10.0, Double(places))
        return (value * scale).rounded() / scale
    }

    private static func seconds(_ value: TimeInterval) -> Double {
        (value * 1000).rounded() / 1000
    }

    private static func writeValidationReport(
        _ validation: TrainingDatasetPackageValidationResult,
        to outputDirectory: URL
    ) throws {
        let object: [String: Any] = [
            "anvil_package_validation_version": "1.0",
            "checked_at": validation.checkedAt,
            "is_valid": validation.isValid,
            "summary": validation.summary,
            "clip_count": validation.clipCount,
            "importable_clip_count": validation.importableClipCount,
            "missing_package_files": validation.missingPackageFiles,
            "missing_clip_files": validation.missingClipFiles,
            "errors": validation.errors,
        ]
        try writeJSONObject(object, to: outputDirectory.appendingPathComponent(validationReportFileName))
    }

    private static func writeCaptionsFile(
        clips: [(url: URL, caption: String)],
        to outputDir: URL,
        existingRecordsByFile: [String: [String: Any]]
    ) throws {
        let entries = clips.map {
            let file = "clips/\($0.url.lastPathComponent)"
            var entry = existingRecordsByFile[file] ?? [:]
            entry["file"] = file
            entry["caption"] = $0.caption
            return entry
        }
        try writeJSONObjectArray(entries, to: outputDir.appendingPathComponent(captionsFileName))
    }

    private static func captionRecordsByFile(from outputDir: URL) -> [String: [String: Any]] {
        let url = outputDir.appendingPathComponent(captionsFileName)
        return readJSONArray(from: url).reduce(into: [:]) { records, record in
            guard let file = record["file"] as? String, !file.isEmpty else { return }
            records[file] = record
        }
    }

    private static func preserveEnrichmentMetadata(
        from existing: [String: Any]?,
        into object: inout [String: Any]
    ) {
        guard let existing else { return }
        for key in [
            "analysis",
            "lyrics",
            "negative_tags",
            "separation",
            "tags",
            "transcript",
            "transcription",
        ] where existing[key] != nil {
            object[key] = existing[key]
        }
    }

    private static func writeJSONObject(_ object: [String: Any], to url: URL) throws {
        let data = try JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: url, options: .atomic)
    }

    private static func writeJSONObjectArray(_ object: [[String: Any]], to url: URL) throws {
        let data = try JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: url, options: .atomic)
    }

    private static func readJSONObject(from url: URL) -> [String: Any]? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
    }

    private static func readJSONArray(from url: URL) -> [[String: Any]] {
        guard let data = try? Data(contentsOf: url) else { return [] }
        return (try? JSONSerialization.jsonObject(with: data)) as? [[String: Any]] ?? []
    }

    private static func datasetBaseDirectory(from dict: [String: Any]?, fallback: URL) -> URL {
        guard let datasetDir = dict?["dataset_dir"] as? String, !datasetDir.isEmpty else {
            return fallback
        }
        return resolve(datasetDir, relativeTo: fallback)
    }

    private static func selectedAudioPath(from clip: [String: Any]) -> String? {
        if let assets = clip["assets"] as? [String: Any],
           let fullMix = assets["full_mix"] as? String {
            return fullMix
        }
        for key in ["file", "path", "audio"] {
            if let value = clip[key] as? String, !value.isEmpty {
                return value
            }
        }
        return nil
    }

    private static func resolve(_ path: String, relativeTo baseDirectory: URL) -> URL {
        let expanded = (path as NSString).expandingTildeInPath
        if expanded.hasPrefix("/") {
            return URL(fileURLWithPath: expanded).standardizedFileURL
        }
        return baseDirectory.appendingPathComponent(expanded).standardizedFileURL
    }
}
