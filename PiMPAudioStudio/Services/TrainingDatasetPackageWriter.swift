import AVFoundation
import Foundation
import PiMPAudioCore

protocol DatasetAnnotationBackend: Sendable {
    var annotationDescriptor: DatasetAnnotationBackendDescriptor { get }
}

struct DatasetAnnotationBackendDescriptor: Equatable, Sendable {
    var id: String
    var version: String
    var modelID: String
    var modelRevision: String

    init(
        id: String,
        version: String = "",
        modelID: String = "",
        modelRevision: String = ""
    ) {
        self.id = id
        self.version = version
        self.modelID = modelID
        self.modelRevision = modelRevision
    }
}

struct DatasetAnnotationMetadata: Equatable, Sendable {
    var schemaVersion: String = TrainingDatasetPackageWriter.annotationSchemaVersion
    var updatedAt: Date
    var backend: DatasetAnnotationBackendDescriptor
    var captionSource: String
    var lyricsSource: String
    var language: String
    var caption: String
    var lyrics: String
    var reviewStatus: String
    var rawOutputFiles: [String: String]

    init(
        updatedAt: Date,
        backend: DatasetAnnotationBackendDescriptor,
        captionSource: String,
        lyricsSource: String,
        language: String,
        caption: String,
        lyrics: String,
        reviewStatus: String,
        rawOutputFiles: [String: String] = [:],
        rawCaptionOutputFile: String = "",
        rawLyricsOutputFile: String = ""
    ) {
        self.updatedAt = updatedAt
        self.backend = backend
        self.captionSource = captionSource
        self.lyricsSource = lyricsSource
        self.language = language
        self.caption = caption
        self.lyrics = lyrics
        self.reviewStatus = reviewStatus

        var rawOutputFiles = rawOutputFiles
        if !rawCaptionOutputFile.isEmpty {
            rawOutputFiles["caption"] = rawCaptionOutputFile
        }
        if !rawLyricsOutputFile.isEmpty {
            rawOutputFiles["lyrics"] = rawLyricsOutputFile
        }
        self.rawOutputFiles = rawOutputFiles
    }

    var persistedObject: [String: Any] {
        [
            "schema_version": schemaVersion,
            "updated_at": ISO8601DateFormatter().string(from: updatedAt),
            "annotation_backend": backend.id,
            "annotation_backend_version": backend.version,
            "caption_source": captionSource,
            "lyrics_source": lyricsSource,
            "model_id": backend.modelID,
            "model_revision": backend.modelRevision,
            "language": language,
            "caption_present": !caption.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
            "lyrics_present": !lyrics.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
            "review_status": reviewStatus,
            "raw_output_files": rawOutputFiles,
        ]
    }
}

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
            return "\(clipCount) training asset\(clipCount == 1 ? "" : "s") validated for package export."
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
    static let aceStepTrainingMaximumDuration: TimeInterval = 240
    static let annotationSchemaVersion = "1.0"
    static let preprocessCacheVersion = "1.0"
    static let captionsFileName = "captions.json"
    static let datasetConfigFileName = "dataset_config.json"
    static let datasetManifestFileName = "dataset_manifest.json"
    static let trainingBundleFileName = "training_bundle.json"
    static let fullMixBundleFileName = "training_bundle_full_mix.json"
    static let vocalBundleFileName = "training_bundle_vocals.json"
    static let instrumentalBundleFileName = "training_bundle_instrumental.json"
    static let remotePackageFileName = "remote_training_package.json"
    static let validationReportFileName = "package_validation.json"
    static let preprocessCacheFileName = "preprocess_cache.json"
    static let reviewBackupsDirectoryName = "review_backups"

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

    @discardableResult
    static func writeLocalPackage(
        outputDirectory: URL,
        clips: [TrainingImportedFile],
        sourceFiles: [URL],
        config: LocalFilesDatasetConfig,
        styleTemplate: String,
        createdAt: Date = Date()
    ) throws -> [TrainingImportedFile] {
        if config.buildMode == .wholeTracks {
            return try writeWholeTrackPackage(
                outputDirectory: outputDirectory,
                trainingItems: clips,
                sourceFiles: sourceFiles,
                styleTemplate: styleTemplate,
                createdAt: createdAt
            )
        }

        return try writeClipExtractionPackage(
            outputDirectory: outputDirectory,
            clips: clips,
            sourceFiles: sourceFiles,
            config: config,
            styleTemplate: styleTemplate,
            createdAt: createdAt
        )
    }

    static func stageWholeTrackSource(
        _ sourceURL: URL,
        index: Int,
        outputDirectory: URL
    ) throws -> URL {
        let sourcesDirectory = outputDirectory.appendingPathComponent("sources", isDirectory: true)
        try FileManager.default.createDirectory(at: sourcesDirectory, withIntermediateDirectories: true)

        let baseName = slug(sourceURL.deletingPathExtension().lastPathComponent)
        let fileExtension = sourceURL.pathExtension.lowercased()
        let extensionSuffix = fileExtension.isEmpty ? "" : ".\(fileExtension)"
        let fileName = "\(String(format: "f%03d", index + 1))_\(baseName)\(extensionSuffix)"
        let destinationURL = sourcesDirectory.appendingPathComponent(fileName)
        guard sourceURL.standardizedFileURL != destinationURL.standardizedFileURL else {
            return destinationURL
        }
        guard !FileManager.default.fileExists(atPath: destinationURL.path(percentEncoded: false)) else {
            return destinationURL
        }
        try FileManager.default.copyItem(at: sourceURL, to: destinationURL)
        return destinationURL
    }

    private static func writeWholeTrackPackage(
        outputDirectory: URL,
        trainingItems: [TrainingImportedFile],
        sourceFiles: [URL],
        styleTemplate: String,
        createdAt: Date
    ) throws -> [TrainingImportedFile] {
        try FileManager.default.createDirectory(at: outputDirectory, withIntermediateDirectories: true)
        let manifestURL = outputDirectory.appendingPathComponent(datasetManifestFileName)
        let existingPackage = readPackageV2(from: manifestURL)
        let existingRecords = captionRecordsByFile(from: outputDirectory)
        let existingAssetsByID = Dictionary(
            uniqueKeysWithValues: (existingPackage?.assets ?? []).map { ($0.id, $0) }
        )
        let trimmedStyle = styleTemplate.trimmingCharacters(in: .whitespacesAndNewlines)
        let createdAtString = existingPackage?.createdAt ?? ISO8601DateFormatter().string(from: createdAt)

        var tracks: [DatasetSourceTrackV2] = []
        var assets: [DatasetAssetV2] = []
        var selections: [DatasetTrainingSelectionV2] = []
        var packagedItems: [TrainingImportedFile] = []

        for (index, sourceURL) in sourceFiles.enumerated() {
            let packagedURL = try stageWholeTrackSource(
                sourceURL,
                index: index,
                outputDirectory: outputDirectory
            )
            let metadata = audioMetadata(for: packagedURL)
            let relativePath = relativeClipPath(for: packagedURL, from: outputDirectory)
            let trackID = "track-\(String(format: "%04d", index + 1))-\(slug(sourceURL.deletingPathExtension().lastPathComponent))"
            let assetID = "asset-\(trackID)-full-mix"
            let matchingItem = wholeTrackTrainingItem(
                sourceURL: sourceURL,
                packagedURL: packagedURL,
                index: index,
                trainingItems: trainingItems
            )
            let caption = normalizedWholeTrackCaption(
                item: matchingItem,
                sourceURL: sourceURL,
                styleTemplate: trimmedStyle
            )
            let captionSource = trimmedStyle.isEmpty ? "filename" : "manual"
            let existingRecord = existingRecords[relativePath]
                ?? existingRecords.values.first {
                    stringValue($0["source_file_name"]) == sourceURL.lastPathComponent
                }
            let annotation = packageV2Annotation(
                from: existingRecord,
                fallbackCaption: caption,
                fallbackCaptionSource: captionSource
            )
            let file = DatasetAudioFileV2(
                path: relativePath,
                format: metadata.format == "unknown" ? nil : metadata.format,
                byteSize: metadata.byteSize,
                durationSeconds: metadata.durationSeconds,
                sampleRate: metadata.sampleRate,
                channels: metadata.channels
            )
            let sourceRange = DatasetTimeRangeV2(
                startSeconds: 0,
                endSeconds: metadata.durationSeconds
            )

            tracks.append(
                DatasetSourceTrackV2(
                    id: trackID,
                    originalFileName: sourceURL.lastPathComponent,
                    durationSeconds: metadata.durationSeconds,
                    sourceFile: file
                )
            )
            assets.append(
                DatasetAssetV2(
                    id: assetID,
                    sourceTrackID: trackID,
                    role: .fullMix,
                    file: file,
                    sourceRange: sourceRange,
                    derivation: DatasetAssetDerivationV2(kind: .imported),
                    annotation: annotation
                )
            )

            let effectiveEnd = metadata.durationSeconds.map {
                min($0, aceStepTrainingMaximumDuration)
            }
            let isBackendLimited = (metadata.durationSeconds ?? 0) > aceStepTrainingMaximumDuration
            selections.append(
                DatasetTrainingSelectionV2(
                    assetID: assetID,
                    includeInTraining: true,
                    trainingRange: DatasetTimeRangeV2(startSeconds: 0, endSeconds: effectiveEnd),
                    rangePolicy: isBackendLimited ? .backendLimit : .fullSource
                )
            )
            packagedItems.append(
                TrainingImportedFile(
                    url: packagedURL,
                    description: annotation.caption?.text ?? caption,
                    lyrics: annotation.lyrics?.text ?? "",
                    sourceClip: TrainingSourceClipMetadata(
                        fileName: sourceURL.lastPathComponent,
                        startSeconds: 0,
                        frameCount: metadata.frameCount,
                        sampleRate: metadata.sampleRate.map(Double.init)
                    )
                )
            )

            let preservedDerivedAssets = existingAssetsByID.values
                .filter { $0.sourceTrackID == trackID && $0.role != .fullMix }
                .sorted { $0.id < $1.id }
            assets.append(contentsOf: preservedDerivedAssets)
            selections.append(contentsOf: preservedDerivedAssets.map {
                DatasetTrainingSelectionV2(
                    assetID: $0.id,
                    includeInTraining: false,
                    trainingRange: limitedTrainingRange(for: $0.sourceRange),
                    rangePolicy: trainingRangePolicy(for: $0.sourceRange)
                )
            })
        }

        let package = DatasetPackageV2(
            id: slug(outputDirectory.lastPathComponent),
            name: outputDirectory.lastPathComponent,
            createdAt: createdAtString,
            sourceTracks: tracks,
            assets: assets,
            trainingPlans: [
                DatasetTrainingPlanV2(
                    id: "ace-step-full-mix",
                    backendID: "ace-step-training-v2",
                    name: "ACE-Step full mixes",
                    isDefault: true,
                    selections: selections
                ),
            ]
        )
        try writePackageV2Artifacts(
            package,
            outputDirectory: outputDirectory,
            existingRecords: existingRecords,
            buildMode: .wholeTracks,
            writtenAt: createdAt
        )
        return packagedItems
    }

    private static func writeClipExtractionPackage(
        outputDirectory: URL,
        clips: [TrainingImportedFile],
        sourceFiles: [URL],
        config: LocalFilesDatasetConfig,
        styleTemplate: String,
        createdAt: Date
    ) throws -> [TrainingImportedFile] {
        try FileManager.default.createDirectory(at: outputDirectory, withIntermediateDirectories: true)
        let manifestURL = outputDirectory.appendingPathComponent(datasetManifestFileName)
        let existingPackage = readPackageV2(from: manifestURL)
        let existingRecords = captionRecordsByFile(from: outputDirectory)
        var packageRecords = existingRecords
        let existingAssets = existingPackage?.assets ?? []
        let trimmedStyle = styleTemplate.trimmingCharacters(in: .whitespacesAndNewlines)
        let createdAtString = existingPackage?.createdAt ?? ISO8601DateFormatter().string(from: createdAt)

        var tracks: [DatasetSourceTrackV2] = []
        var assets: [DatasetAssetV2] = []
        var selections: [DatasetTrainingSelectionV2] = []
        var fullMixAssets: [DatasetAssetV2] = []

        for (index, sourceURL) in sourceFiles.enumerated() {
            let packagedURL = try stageWholeTrackSource(
                sourceURL,
                index: index,
                outputDirectory: outputDirectory
            )
            let metadata = audioMetadata(for: packagedURL)
            let file = DatasetAudioFileV2(
                path: relativeClipPath(for: packagedURL, from: outputDirectory),
                format: metadata.format == "unknown" ? nil : metadata.format,
                byteSize: metadata.byteSize,
                durationSeconds: metadata.durationSeconds,
                sampleRate: metadata.sampleRate,
                channels: metadata.channels
            )
            let trackID = "track-\(String(format: "%04d", index + 1))-\(slug(sourceURL.deletingPathExtension().lastPathComponent))"
            let fullMixID = "asset-\(trackID)-full-mix"
            let sourceRange = DatasetTimeRangeV2(startSeconds: 0, endSeconds: metadata.durationSeconds)
            let fallbackCaption = trimmedStyle.isEmpty
                ? sourceURL.deletingPathExtension().lastPathComponent
                : trimmedStyle
            let track = DatasetSourceTrackV2(
                id: trackID,
                originalFileName: sourceURL.lastPathComponent,
                durationSeconds: metadata.durationSeconds,
                sourceFile: file
            )
            let fullMix = DatasetAssetV2(
                id: fullMixID,
                sourceTrackID: trackID,
                role: .fullMix,
                file: file,
                sourceRange: sourceRange,
                derivation: DatasetAssetDerivationV2(kind: .imported),
                annotation: packageV2Annotation(
                    from: existingRecords[file.path],
                    fallbackCaption: fallbackCaption,
                    fallbackCaptionSource: trimmedStyle.isEmpty ? "filename" : "manual"
                )
            )
            tracks.append(track)
            assets.append(fullMix)
            fullMixAssets.append(fullMix)
            selections.append(
                DatasetTrainingSelectionV2(
                    assetID: fullMixID,
                    includeInTraining: false,
                    trainingRange: limitedTrainingRange(for: sourceRange),
                    rangePolicy: trainingRangePolicy(for: sourceRange)
                )
            )
        }

        let clipsDirectory = outputDirectory.appendingPathComponent("clips", isDirectory: true)
        try FileManager.default.createDirectory(at: clipsDirectory, withIntermediateDirectories: true)
        var packagedClips: [TrainingImportedFile] = []

        for (index, clip) in clips.enumerated() {
            let clipURL: URL
            if normalizedFilePath(clip.url).hasPrefix(normalizedDirectoryPath(outputDirectory)) {
                clipURL = clip.url
            } else {
                let destination = clipsDirectory.appendingPathComponent(clip.url.lastPathComponent)
                if !FileManager.default.fileExists(atPath: destination.path(percentEncoded: false)) {
                    try FileManager.default.copyItem(at: clip.url, to: destination)
                }
                clipURL = destination
            }

            let indexedSource = sourceIndex(from: clipURL.lastPathComponent).map { $0 - 1 }
            let metadataSource = clip.sourceClip?.fileName.flatMap { fileName in
                sourceFiles.firstIndex { $0.lastPathComponent == fileName }
            }
            let sourceFileIndex = indexedSource
                ?? metadataSource
                ?? (sourceFiles.count == 1 ? 0 : nil)
            guard let sourceFileIndex,
                  tracks.indices.contains(sourceFileIndex),
                  fullMixAssets.indices.contains(sourceFileIndex) else {
                throw TrainingDatasetPackageError.invalidPackage(
                    "Could not associate \(clipURL.lastPathComponent) with a preserved source track."
                )
            }
            let track = tracks[sourceFileIndex]
            let fullMix = fullMixAssets[sourceFileIndex]

            let metadata = audioMetadata(for: clipURL)
            let relativePath = relativeClipPath(for: clipURL, from: outputDirectory)
            let startSeconds = clip.sourceClip?.startSeconds ?? 0
            let durationSeconds = metadata.durationSeconds ?? config.clipLength
            let sourceRange = DatasetTimeRangeV2(
                startSeconds: startSeconds,
                endSeconds: startSeconds + durationSeconds
            )
            let clipID = "asset-\(track.id)-clip-\(String(format: "%04d", index + 1))"
            let fallbackCaption = normalizedCaption(for: clip)
            let existingRecord = existingRecords[relativePath]
            var annotationRecord = existingRecord ?? [:]
            annotationRecord["caption"] = fallbackCaption
            annotationRecord["prompt"] = fallbackCaption
            annotationRecord["caption_source"] = trimmedStyle.isEmpty ? "filename" : "manual"
            let annotation = packageV2Annotation(
                from: annotationRecord,
                fallbackCaption: fallbackCaption,
                fallbackCaptionSource: trimmedStyle.isEmpty ? "filename" : "manual"
            )
            let file = DatasetAudioFileV2(
                path: relativePath,
                format: metadata.format == "unknown" ? nil : metadata.format,
                byteSize: metadata.byteSize,
                durationSeconds: metadata.durationSeconds,
                sampleRate: metadata.sampleRate,
                channels: metadata.channels
            )
            let clipAsset = DatasetAssetV2(
                id: clipID,
                sourceTrackID: track.id,
                role: .clip,
                file: file,
                derivedFrom: fullMix.id,
                sourceRange: sourceRange,
                derivation: DatasetAssetDerivationV2(
                    kind: .clipExtraction,
                    backendID: "pimp-audio-native",
                    createdAt: createdAtString,
                    details: [
                        "sampling_mode": config.samplingMode.rawValue,
                        "sampling_seed": String(config.samplingSeed),
                    ]
                ),
                annotation: annotation
            )
            assets.append(clipAsset)
            var compatibilityRecord = existingRecord ?? [:]
            compatibilityRecord["id"] = clipID
            compatibilityRecord["file"] = relativePath
            compatibilityRecord["caption"] = annotation.caption?.text ?? fallbackCaption
            compatibilityRecord["lyrics"] = annotation.lyrics?.text ?? clip.lyrics
            compatibilityRecord["audio"] = metadata.jsonObject
            applySourceAuditFields(
                to: &compatibilityRecord,
                source: sourceMetadata(for: clip, sourceFiles: sourceFiles)
            )
            packageRecords[relativePath] = compatibilityRecord
            selections.append(
                DatasetTrainingSelectionV2(
                    assetID: clipID,
                    includeInTraining: true,
                    trainingRange: sourceRange,
                    rangePolicy: .fullSource
                )
            )
            packagedClips.append(
                TrainingImportedFile(
                    url: clipURL,
                    description: annotation.caption?.text ?? fallbackCaption,
                    lyrics: annotation.lyrics?.text ?? clip.lyrics,
                    sourceClip: clip.sourceClip
                )
            )

            let preservedDerivedAssets = existingAssets
                .filter { $0.derivedFrom == clipID && $0.role != .clip }
                .sorted { $0.id < $1.id }
            assets.append(contentsOf: preservedDerivedAssets)
            selections.append(contentsOf: preservedDerivedAssets.map {
                DatasetTrainingSelectionV2(
                    assetID: $0.id,
                    includeInTraining: false,
                    trainingRange: limitedTrainingRange(for: $0.sourceRange),
                    rangePolicy: trainingRangePolicy(for: $0.sourceRange)
                )
            })
        }

        let package = DatasetPackageV2(
            id: slug(outputDirectory.lastPathComponent),
            name: outputDirectory.lastPathComponent,
            createdAt: createdAtString,
            sourceTracks: tracks,
            assets: assets,
            trainingPlans: [
                DatasetTrainingPlanV2(
                    id: "ace-step-clips",
                    backendID: "ace-step-training-v2",
                    name: "ACE-Step derived clips",
                    isDefault: true,
                    selections: selections
                ),
            ]
        )
        try writePackageV2Artifacts(
            package,
            outputDirectory: outputDirectory,
            existingRecords: packageRecords,
            buildMode: .clipExtraction,
            writtenAt: createdAt
        )
        return packagedClips
    }

    private static func wholeTrackTrainingItem(
        sourceURL: URL,
        packagedURL: URL,
        index: Int,
        trainingItems: [TrainingImportedFile]
    ) -> TrainingImportedFile? {
        trainingItems.first {
            $0.sourceClip?.fileName == sourceURL.lastPathComponent
                || $0.url.standardizedFileURL == packagedURL.standardizedFileURL
                || $0.url.standardizedFileURL == sourceURL.standardizedFileURL
        } ?? (trainingItems.indices.contains(index) ? trainingItems[index] : nil)
    }

    private static func normalizedWholeTrackCaption(
        item: TrainingImportedFile?,
        sourceURL: URL,
        styleTemplate: String
    ) -> String {
        if let item {
            let caption = item.description.trimmingCharacters(in: .whitespacesAndNewlines)
            if !caption.isEmpty {
                return caption
            }
        }
        if !styleTemplate.isEmpty {
            return styleTemplate
        }
        return sourceURL.deletingPathExtension().lastPathComponent
    }

    private static func limitedTrainingRange(
        for sourceRange: DatasetTimeRangeV2
    ) -> DatasetTimeRangeV2 {
        DatasetTimeRangeV2(
            startSeconds: sourceRange.startSeconds,
            endSeconds: sourceRange.endSeconds.map {
                min($0, sourceRange.startSeconds + aceStepTrainingMaximumDuration)
            }
        )
    }

    private static func trainingRangePolicy(
        for sourceRange: DatasetTimeRangeV2
    ) -> DatasetTrainingRangePolicyV2 {
        guard let duration = sourceRange.durationSeconds else { return .fullSource }
        return duration > aceStepTrainingMaximumDuration ? .backendLimit : .fullSource
    }

    private static func packageV2Annotation(
        from record: [String: Any]?,
        fallbackCaption: String,
        fallbackCaptionSource: String
    ) -> DatasetAssetAnnotationV2 {
        let record = record ?? [:]
        let legacyAnnotation = record["annotation"] as? [String: Any] ?? [:]
        let transcription = record["transcription"] as? [String: Any] ?? [:]
        let review = record["training_review"] as? [String: Any] ?? [:]
        let caption = stringValue(record["caption"])
            ?? stringValue(record["prompt"])
            ?? fallbackCaption
        let captionSource = stringValue(record["caption_source"])
            ?? stringValue(legacyAnnotation["caption_source"])
            ?? fallbackCaptionSource
        let lyrics = stringValue(record["lyrics"])
            ?? stringValue(record["transcript"])
            ?? stringValue(transcription["text"])
            ?? ""
        let lyricsSource = stringValue(record["lyrics_source"])
            ?? stringValue(legacyAnnotation["lyrics_source"])
            ?? stringValue(transcription["source"])
            ?? (lyrics.isEmpty ? "none" : "manual")
        let language = stringValue(record["language"])
            ?? stringValue(legacyAnnotation["language"])
        let rawOutputs = (legacyAnnotation["raw_output_files"] as? [String: Any] ?? [:])
            .compactMapValues { stringValue($0) }

        let captionBackend = slug(
            stringValue(legacyAnnotation["annotation_backend"])
                ?? (captionSource == "filename" ? "filename" : "manual")
        )
        let lyricsBackend = slug(
            stringValue(transcription["backend"])
                ?? stringValue(legacyAnnotation["annotation_backend"])
                ?? "manual"
        )
        let captionEntry = DatasetTextAnnotationV2(
            text: caption,
            source: captionSource,
            language: language,
            provenance: DatasetAnnotationProvenanceV2(
                backendID: captionBackend,
                backendVersion: stringValue(legacyAnnotation["annotation_backend_version"]),
                modelID: stringValue(legacyAnnotation["model_id"]),
                modelRevision: stringValue(legacyAnnotation["model_revision"]),
                generatedAt: stringValue(legacyAnnotation["updated_at"]),
                rawOutputFiles: rawOutputs.filter { $0.key != "lyrics" }
            )
        )
        let lyricsEntry = lyrics.isEmpty ? nil : DatasetTextAnnotationV2(
            text: lyrics,
            source: lyricsSource,
            language: language,
            provenance: DatasetAnnotationProvenanceV2(
                backendID: lyricsBackend,
                backendVersion: stringValue(transcription["backend_version"])
                    ?? stringValue(legacyAnnotation["annotation_backend_version"]),
                modelID: stringValue(transcription["model"])
                    ?? stringValue(legacyAnnotation["model_id"]),
                modelRevision: stringValue(transcription["model_revision"])
                    ?? stringValue(legacyAnnotation["model_revision"]),
                generatedAt: stringValue(transcription["updated_at"])
                    ?? stringValue(legacyAnnotation["updated_at"]),
                rawOutputFiles: rawOutputs.filter { $0.key != "caption" }
            )
        )
        let approved = boolValue(review["approved"])
            ?? (stringValue(legacyAnnotation["review_status"]) == "approved")
        return DatasetAssetAnnotationV2(
            caption: captionEntry,
            lyrics: lyricsEntry,
            reviewStatus: stringValue(review["status"])
                ?? stringValue(legacyAnnotation["review_status"])
                ?? (approved ? "approved" : "needs_review"),
            approved: approved,
            reviewedAt: stringValue(review["updated_at"]),
            notes: stringValue(review["notes"])
        )
    }

    private static func writePackageV2Artifacts(
        _ package: DatasetPackageV2,
        outputDirectory: URL,
        existingRecords: [String: [String: Any]],
        buildMode: LocalDatasetBuildMode,
        writtenAt: Date
    ) throws {
        let manifestData = try DatasetPackageV2Codec.encodeValidated(package)
        try manifestData.write(
            to: outputDirectory.appendingPathComponent(datasetManifestFileName),
            options: .atomic
        )

        let bundle = try compatibilityBundleObject(
            from: package,
            existingRecords: existingRecords
        )
        try writeJSONObject(bundle, to: outputDirectory.appendingPathComponent(trainingBundleFileName))
        try writeJSONObject(bundle, to: outputDirectory.appendingPathComponent(fullMixBundleFileName))
        try writeJSONObjectArray(
            bundle["clips"] as? [[String: Any]] ?? [],
            to: outputDirectory.appendingPathComponent(captionsFileName)
        )
        try writeJSONObject(
            datasetConfig(for: buildMode),
            to: outputDirectory.appendingPathComponent(datasetConfigFileName)
        )
        try refreshPreprocessCacheManifest(at: outputDirectory, createdAt: writtenAt)
        try writeJSONObject(
            remotePackageDescriptor(
                outputDirectory: outputDirectory,
                clipCount: package.trainingPlans.first(where: \.isDefault)?
                    .selections.filter(\.includeInTraining).count ?? 0,
                createdAt: ISO8601DateFormatter().string(from: writtenAt),
                buildMode: buildMode
            ),
            to: outputDirectory.appendingPathComponent(remotePackageFileName)
        )
        let validation = validatePackage(at: outputDirectory, checkedAt: writtenAt)
        try writeValidationReport(validation, to: outputDirectory)
    }

    private static func compatibilityBundleObject(
        from package: DatasetPackageV2,
        existingRecords: [String: [String: Any]]
    ) throws -> [String: Any] {
        let data = try DatasetTrainingBundleV1Projection.encodeBundle(from: package)
        guard var bundle = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw TrainingDatasetPackageError.invalidPackage(
                "Package v2 could not produce a compatibility training bundle."
            )
        }
        bundle["preprocess_cache"] = preprocessCacheFileName
        bundle["training_lyrics_source"] = "per_asset_annotation"

        let assetsByID = Dictionary(uniqueKeysWithValues: package.assets.map { ($0.id, $0) })
        let tracksByID = Dictionary(uniqueKeysWithValues: package.sourceTracks.map { ($0.id, $0) })
        let existingRecordsByID = existingRecords.values.reduce(into: [String: [String: Any]]()) {
            result, record in
            guard let id = stringValue(record["id"]), result[id] == nil else { return }
            result[id] = record
        }
        let projectedClips = bundle["clips"] as? [[String: Any]] ?? []
        bundle["clips"] = projectedClips.compactMap { projected -> [String: Any]? in
            guard let assetID = stringValue(projected["asset_id"]),
                  let asset = assetsByID[assetID],
                  let track = tracksByID[asset.sourceTrackID] else {
                return nil
            }

            var record = existingRecordsByID[asset.id]
                ?? existingRecords[asset.file.path]
                ?? [:]
            for (key, value) in projected {
                record[key] = value
            }
            var assetPaths = record["assets"] as? [String: Any] ?? [:]
            assetPaths[asset.role.rawValue] = asset.file.path
            record["id"] = asset.id
            record["file"] = asset.file.path
            record["assets"] = assetPaths
            record["asset_id"] = asset.id
            record["asset_role"] = asset.role.rawValue
            record["source_track_id"] = track.id

            let caption = asset.annotation?.caption?.text
                ?? URL(fileURLWithPath: asset.file.path).deletingPathExtension().lastPathComponent
            let lyrics = asset.annotation?.lyrics?.text ?? ""
            let captionSource = asset.annotation?.caption?.source ?? "filename"
            let lyricsSource = lyrics.isEmpty ? "none" : (asset.annotation?.lyrics?.source ?? "manual")
            record["caption"] = caption
            record["prompt"] = caption
            record["caption_source"] = captionSource
            record["lyrics"] = lyrics
            record["transcript"] = lyrics
            record["lyrics_source"] = lyricsSource
            record["annotation_schema_version"] = annotationSchemaVersion
            record["annotation"] = legacyAnnotationObject(from: asset.annotation)
            record["training_review"] = [
                "approved": asset.annotation?.approved ?? false,
                "status": asset.annotation?.reviewStatus ?? "needs_review",
            ]
            record["transcription"] = [
                "status": lyrics.isEmpty ? "not_run" : "complete",
                "text": lyrics,
                "source": lyricsSource,
            ]
            record["audio"] = audioFileJSONObject(asset.file)
            if let duration = asset.file.durationSeconds {
                record["clip_length_seconds"] = seconds(duration)
            }
            var source = record["source"] as? [String: Any] ?? [:]
            source["kind"] = "local_file"
            source["file_name"] = track.originalFileName
            source["start_seconds"] = seconds(asset.sourceRange.startSeconds)
            source["source_start_seconds"] = seconds(asset.sourceRange.startSeconds)
            applySourceAuditFields(to: &record, source: source)
            return record
        }
        bundle["clip_count"] = (bundle["clips"] as? [[String: Any]])?.count ?? 0
        return bundle
    }

    private static func legacyAnnotationObject(
        from annotation: DatasetAssetAnnotationV2?
    ) -> [String: Any] {
        let caption = annotation?.caption
        let lyrics = annotation?.lyrics
        let backend = caption?.provenance ?? lyrics?.provenance
        var rawOutputFiles = caption?.provenance?.rawOutputFiles ?? [:]
        for (key, value) in lyrics?.provenance?.rawOutputFiles ?? [:] {
            rawOutputFiles[key] = value
        }
        return DatasetAnnotationMetadata(
            updatedAt: Date(),
            backend: DatasetAnnotationBackendDescriptor(
                id: backend?.backendID ?? "manual",
                version: backend?.backendVersion ?? "",
                modelID: backend?.modelID ?? "",
                modelRevision: backend?.modelRevision ?? ""
            ),
            captionSource: caption?.source ?? "none",
            lyricsSource: lyrics?.source ?? "none",
            language: caption?.language ?? lyrics?.language ?? "",
            caption: caption?.text ?? "",
            lyrics: lyrics?.text ?? "",
            reviewStatus: annotation?.reviewStatus ?? "needs_review",
            rawOutputFiles: rawOutputFiles
        ).persistedObject
    }

    private static func audioFileJSONObject(_ file: DatasetAudioFileV2) -> [String: Any] {
        var object: [String: Any] = [:]
        if let format = file.format { object["format"] = format }
        if let byteSize = file.byteSize { object["byte_size"] = byteSize }
        if let duration = file.durationSeconds { object["duration_seconds"] = seconds(duration) }
        if let sampleRate = file.sampleRate { object["sample_rate"] = sampleRate }
        if let channels = file.channels { object["channels"] = channels }
        if let sha256 = file.sha256 { object["sha256"] = sha256 }
        return object
    }

    private static func datasetConfig(for buildMode: LocalDatasetBuildMode) -> [String: Any] {
        let isWholeTrackPackage = buildMode == .wholeTracks
        return [
            "pimp_audio_dataset_config_version": "2.0",
            "dataset_type": "audio_dir",
            "datasets": [
                [
                    "path": isWholeTrackPackage ? "sources" : "clips",
                    "caption_file": captionsFileName,
                    "audio_role": isWholeTrackPackage ? "full_mix" : "clip",
                ],
            ],
            "random_crop": false,
            "source_policy": isWholeTrackPackage
                ? "whole_tracks"
                : "whole_tracks_with_derived_clips",
            "training_max_duration_seconds": seconds(aceStepTrainingMaximumDuration),
        ]
    }

    private static func readPackageV2(from manifestURL: URL) -> DatasetPackageV2? {
        guard let data = try? Data(contentsOf: manifestURL) else { return nil }
        return try? DatasetPackageV2Codec.decodeValidated(data)
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
        var missingClipFiles = clips.compactMap { clip -> String? in
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

        let manifestURL = outputDirectory.appendingPathComponent(datasetManifestFileName)
        if let manifest = readJSONObject(from: manifestURL),
           stringValue(manifest["pimp_audio_dataset_version"]) == DatasetPackageV2.currentVersion {
            do {
                let package = try DatasetPackageV2Codec.decodeValidated(Data(contentsOf: manifestURL))
                for asset in package.assets {
                    let assetURL = resolve(asset.file.path, relativeTo: outputDirectory)
                    if !FileManager.default.fileExists(atPath: assetURL.path(percentEncoded: false)),
                       !missingClipFiles.contains(asset.file.path) {
                        missingClipFiles.append(asset.file.path)
                    }
                }
            } catch {
                errors.append("dataset_manifest.json is not a valid package-v2 manifest: \(error.localizedDescription)")
            }
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

    static func refreshTrainingBundles(
        at outputDirectory: URL,
        updatedAt: Date = Date()
    ) throws {
        if try refreshPackageV2FromCaptions(at: outputDirectory, updatedAt: updatedAt) {
            return
        }

        let captions = readJSONArray(from: outputDirectory.appendingPathComponent(captionsFileName))
        guard !captions.isEmpty else {
            throw TrainingDatasetPackageError.invalidPackage("captions.json does not contain any clips.")
        }

        let updatedAtString = ISO8601DateFormatter().string(from: updatedAt)
        let bundle = trainingBundle(clips: captions, createdAt: updatedAtString)
        try writeJSONObject(bundle, to: outputDirectory.appendingPathComponent(trainingBundleFileName))
        try writeJSONObject(bundle, to: outputDirectory.appendingPathComponent(fullMixBundleFileName))
        try refreshPreprocessCacheManifest(at: outputDirectory, createdAt: updatedAt)
        try refreshValidationReport(at: outputDirectory, checkedAt: updatedAt)
    }

    static func writeReviewedPackageV2(
        _ package: DatasetPackageV2,
        records: [[String: Any]],
        outputDirectory: URL,
        updatedAt: Date = Date()
    ) throws {
        let recordsByFile = records.reduce(into: [String: [String: Any]]()) { result, record in
            guard let file = stringValue(record["file"]), result[file] == nil else { return }
            result[file] = record
        }
        try writePackageV2Artifacts(
            package,
            outputDirectory: outputDirectory,
            existingRecords: recordsByFile,
            buildMode: buildMode(for: package),
            writtenAt: updatedAt
        )
    }

    @discardableResult
    static func refreshPackageV2FromCaptions(
        at outputDirectory: URL,
        updatedAt: Date = Date()
    ) throws -> Bool {
        let manifestURL = outputDirectory.appendingPathComponent(datasetManifestFileName)
        guard var package = readPackageV2(from: manifestURL) else { return false }
        let records = readJSONArray(from: outputDirectory.appendingPathComponent(captionsFileName))
        guard !records.isEmpty else {
            throw TrainingDatasetPackageError.invalidPackage("captions.json does not contain any assets.")
        }
        let recordsByFile = records.reduce(into: [String: [String: Any]]()) { result, record in
            guard let file = stringValue(record["file"]), result[file] == nil else { return }
            result[file] = record
        }
        let recordsByID = records.reduce(into: [String: [String: Any]]()) { result, record in
            guard let id = stringValue(record["id"]), result[id] == nil else { return }
            result[id] = record
        }

        for index in package.assets.indices
        where package.assets[index].role == .fullMix || package.assets[index].role == .clip {
            let asset = package.assets[index]
            guard let record = recordsByID[asset.id] ?? recordsByFile[asset.file.path] else { continue }
            package.assets[index].annotation = packageV2Annotation(
                from: record,
                fallbackCaption: asset.annotation?.caption?.text
                    ?? URL(fileURLWithPath: asset.file.path).deletingPathExtension().lastPathComponent,
                fallbackCaptionSource: asset.annotation?.caption?.source ?? "filename"
            )
        }

        let baseAssets = package.assets.filter { $0.role == .fullMix || $0.role == .clip }
        for baseAsset in baseAssets {
            guard let record = recordsByID[baseAsset.id] ?? recordsByFile[baseAsset.file.path] else {
                continue
            }
            var paths = record["assets"] as? [String: Any] ?? [:]
            let separation = record["separation"] as? [String: Any] ?? [:]
            for (key, value) in separation["stems"] as? [String: Any] ?? [:] {
                if paths[key] == nil {
                    paths[key] = value
                }
            }
            for role in [DatasetAssetRoleV2.vocals, .instrumental] {
                guard let relativePath = stringValue(paths[role.rawValue]) else { continue }
                let fileURL = resolve(relativePath, relativeTo: outputDirectory)
                let metadata = audioMetadata(for: fileURL)
                var annotation = packageV2Annotation(
                    from: record,
                    fallbackCaption: baseAsset.annotation?.caption?.text
                        ?? URL(fileURLWithPath: baseAsset.file.path).deletingPathExtension().lastPathComponent,
                    fallbackCaptionSource: baseAsset.annotation?.caption?.source ?? "filename"
                )
                if role == .instrumental {
                    annotation.lyrics = nil
                }
                let file = DatasetAudioFileV2(
                    path: relativePath,
                    format: metadata.format == "unknown" ? nil : metadata.format,
                    byteSize: metadata.byteSize,
                    durationSeconds: metadata.durationSeconds,
                    sampleRate: metadata.sampleRate,
                    channels: metadata.channels
                )
                let derivation = DatasetAssetDerivationV2(
                    kind: .stemSeparation,
                    backendID: slug(stringValue(separation["backend"]) ?? "demucs-mlx"),
                    backendVersion: stringValue(separation["backend_version"]),
                    modelID: stringValue(separation["model"]),
                    modelRevision: stringValue(separation["model_revision"]),
                    createdAt: stringValue(separation["updated_at"])
                )

                if let assetIndex = package.assets.firstIndex(where: {
                    $0.derivedFrom == baseAsset.id && $0.role == role
                }) {
                    package.assets[assetIndex].file = file
                    package.assets[assetIndex].derivedFrom = baseAsset.id
                    package.assets[assetIndex].sourceRange = baseAsset.sourceRange
                    package.assets[assetIndex].derivation = derivation
                    package.assets[assetIndex].annotation = annotation
                    ensureOptInSelection(
                        for: package.assets[assetIndex],
                        in: &package.trainingPlans
                    )
                } else {
                    let derivedAsset = DatasetAssetV2(
                        id: "asset-\(baseAsset.id)-\(role.rawValue)",
                        sourceTrackID: baseAsset.sourceTrackID,
                        role: role,
                        file: file,
                        derivedFrom: baseAsset.id,
                        sourceRange: baseAsset.sourceRange,
                        derivation: derivation,
                        annotation: annotation
                    )
                    package.assets.append(derivedAsset)
                    ensureOptInSelection(for: derivedAsset, in: &package.trainingPlans)
                }
            }
        }

        try writePackageV2Artifacts(
            package,
            outputDirectory: outputDirectory,
            existingRecords: recordsByFile,
            buildMode: buildMode(for: package),
            writtenAt: updatedAt
        )
        return true
    }

    private static func ensureOptInSelection(
        for asset: DatasetAssetV2,
        in plans: inout [DatasetTrainingPlanV2]
    ) {
        for index in plans.indices where !plans[index].selections.contains(where: { $0.assetID == asset.id }) {
            plans[index].selections.append(
                DatasetTrainingSelectionV2(
                    assetID: asset.id,
                    includeInTraining: false,
                    trainingRange: limitedTrainingRange(for: asset.sourceRange),
                    rangePolicy: trainingRangePolicy(for: asset.sourceRange)
                )
            )
        }
    }

    private static func buildMode(for package: DatasetPackageV2) -> LocalDatasetBuildMode {
        let defaultPlan = package.trainingPlans.first(where: \.isDefault)
        if defaultPlan?.id == "ace-step-clips"
            || defaultPlan?.name.localizedCaseInsensitiveContains("derived clips") == true {
            return .clipExtraction
        }
        return .wholeTracks
    }

    @discardableResult
    static func createReviewBackup(
        at outputDirectory: URL,
        reason: String,
        createdAt: Date = Date()
    ) throws -> URL? {
        let timestamp = compactTimestamp(createdAt)
        var backupDirectory = outputDirectory
            .appendingPathComponent(reviewBackupsDirectoryName, isDirectory: true)
            .appendingPathComponent("\(timestamp)_\(slug(reason))", isDirectory: true)
        if FileManager.default.fileExists(atPath: backupDirectory.path(percentEncoded: false)) {
            backupDirectory = outputDirectory
                .appendingPathComponent(reviewBackupsDirectoryName, isDirectory: true)
                .appendingPathComponent("\(timestamp)_\(slug(reason))_\(UUID().uuidString.prefix(8))", isDirectory: true)
        }

        let filesToCopy = [
            captionsFileName,
            datasetManifestFileName,
            trainingBundleFileName,
            fullMixBundleFileName,
            vocalBundleFileName,
            instrumentalBundleFileName,
            validationReportFileName,
            preprocessCacheFileName,
        ].filter {
            FileManager.default.fileExists(
                atPath: outputDirectory.appendingPathComponent($0).path(percentEncoded: false)
            )
        }

        guard !filesToCopy.isEmpty else { return nil }
        try FileManager.default.createDirectory(at: backupDirectory, withIntermediateDirectories: true)
        for file in filesToCopy {
            try FileManager.default.copyItem(
                at: outputDirectory.appendingPathComponent(file),
                to: backupDirectory.appendingPathComponent(file)
            )
        }
        try writeJSONObject(
            [
                "pimp_audio_review_backup_version": "1.0",
                "created_at": ISO8601DateFormatter().string(from: createdAt),
                "reason": reason,
                "files": filesToCopy,
            ],
            to: backupDirectory.appendingPathComponent("backup_manifest.json")
        )
        return backupDirectory
    }

    static func refreshPreprocessCacheManifest(
        at outputDirectory: URL,
        createdAt: Date = Date()
    ) throws {
        let createdAtString = ISO8601DateFormatter().string(from: createdAt)
        let manifests = [
            preprocessBundleEntry(
                name: "full_mix",
                bundleFileName: fullMixBundleFileName,
                cacheDirectory: "preprocessed/full_mix",
                outputDirectory: outputDirectory
            ),
            preprocessBundleEntry(
                name: "vocals",
                bundleFileName: vocalBundleFileName,
                cacheDirectory: "preprocessed/vocals",
                outputDirectory: outputDirectory
            ),
            preprocessBundleEntry(
                name: "instrumental",
                bundleFileName: instrumentalBundleFileName,
                cacheDirectory: "preprocessed/instrumental",
                outputDirectory: outputDirectory
            ),
        ]

        let statuses = manifests.compactMap { $0["status"] as? String }
        let nonWaitingStatuses = statuses.filter { $0 != "waiting_for_bundle" }
        let manifestStatus: String
        if statuses.contains("ready_for_preprocess") {
            manifestStatus = "ready_for_preprocess"
        } else if statuses.contains("partial") ||
            (!nonWaitingStatuses.isEmpty && nonWaitingStatuses.count < statuses.count) {
            manifestStatus = "partial"
        } else if !nonWaitingStatuses.isEmpty && nonWaitingStatuses.allSatisfy({ $0 == "complete" }) {
            manifestStatus = "complete"
        } else {
            manifestStatus = "waiting_for_bundles"
        }

        let manifest: [String: Any] = [
            "pimp_audio_preprocess_cache_version": preprocessCacheVersion,
            "created_at": createdAtString,
            "updated_at": createdAtString,
            "dataset_dir": ".",
            "dataset_manifest": datasetManifestFileName,
            "cache_root": "preprocessed",
            "adapter_modes": ["lora", "lokr"],
            "status": manifestStatus,
            "bundles": manifests,
            "notes": [
                "This manifest is adapter-agnostic: LoRA and LoKR should reuse the same encoded audio and conditioning tensors when native tensor persistence is wired.",
                "Cache entries are declared before tensors exist so remote workers and future local preprocess steps share one contract.",
            ],
        ]
        try writeJSONObject(manifest, to: outputDirectory.appendingPathComponent(preprocessCacheFileName))
    }

    /// Stages package contents and zips them with `/usr/bin/ditto`.
    ///
    /// Always runs off the calling actor so multi-GB remote/job exports cannot
    /// freeze the UI. The ditto process is terminated if the task is cancelled.
    @discardableResult
    static func exportPackageArchive(
        outputDirectory: URL,
        to destinationURL: URL,
        extraFiles: [String: Data] = [:]
    ) async throws -> URL {
        try await Task.detached(priority: .userInitiated) {
            try await Self.exportPackageArchiveOffMain(
                outputDirectory: outputDirectory,
                to: destinationURL,
                extraFiles: extraFiles
            )
        }.value
    }

    private static func exportPackageArchiveOffMain(
        outputDirectory: URL,
        to destinationURL: URL,
        extraFiles: [String: Data]
    ) async throws -> URL {
        try Task.checkCancellation()

        let validation = validatePackage(at: outputDirectory)
        guard validation.isValid else {
            throw TrainingDatasetPackageError.invalidPackage(validation.summary)
        }

        if FileManager.default.fileExists(atPath: destinationURL.path) {
            try FileManager.default.removeItem(at: destinationURL)
        }

        let stagingRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("PiMPAudioPackageExport-\(UUID().uuidString)", isDirectory: true)
        let stagedPackageDirectory = stagingRoot.appendingPathComponent(outputDirectory.lastPathComponent, isDirectory: true)
        try FileManager.default.createDirectory(at: stagedPackageDirectory, withIntermediateDirectories: true)
        defer {
            try? FileManager.default.removeItem(at: stagingRoot)
        }

        for relativePath in exportPackageRelativePaths(outputDirectory: outputDirectory) {
            try Task.checkCancellation()
            let sourceURL = outputDirectory.appendingPathComponent(relativePath)
            guard FileManager.default.fileExists(atPath: sourceURL.path(percentEncoded: false)) else { continue }
            let destination = stagedPackageDirectory.appendingPathComponent(relativePath)
            try copyArchiveItem(from: sourceURL, to: destination)
        }
        try writeExtraArchiveFiles(extraFiles, to: stagedPackageDirectory)
        try Task.checkCancellation()

        try await runDittoZip(
            stagedPackageDirectory: stagedPackageDirectory,
            destinationURL: destinationURL
        )

        return destinationURL
    }

    private static func runDittoZip(
        stagedPackageDirectory: URL,
        destinationURL: URL
    ) async throws {
        let runner = DittoZipProcess(
            sourceDirectory: stagedPackageDirectory,
            destinationURL: destinationURL
        )
        try await runner.run()
    }

    private static func writeExtraArchiveFiles(_ files: [String: Data], to stagedPackageDirectory: URL) throws {
        for (relativePath, data) in files {
            guard let cleanPath = cleanArchiveRelativePath(relativePath) else {
                continue
            }
            let destination = stagedPackageDirectory.appendingPathComponent(cleanPath)
            try FileManager.default.createDirectory(
                at: destination.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            try data.write(to: destination, options: .atomic)
            if destination.lastPathComponent.hasSuffix(".sh") {
                try FileManager.default.setAttributes(
                    [.posixPermissions: 0o755],
                    ofItemAtPath: destination.path(percentEncoded: false)
                )
            }
        }
    }

    private static func cleanArchiveRelativePath(_ path: String) -> String? {
        let trimmed = path.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty,
              !trimmed.hasPrefix("/"),
              !trimmed.contains("..") else {
            return nil
        }
        return trimmed
    }

    private static func exportPackageRelativePaths(outputDirectory: URL) -> [String] {
        let remoteURL = outputDirectory.appendingPathComponent(remotePackageFileName)
        let remotePackage = readJSONObject(from: remoteURL)
        let copyRules = remotePackage?["copy_rules"] as? [String: Any]
        let include = copyRules?["include"] as? [String]
        let paths = include?.compactMap(exportRootPath(from:)) ?? defaultArchiveIncludePaths()
        return sortedUnique(paths)
    }

    private static func exportRootPath(from includeRule: String) -> String? {
        let trimmed = includeRule.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        if trimmed.hasSuffix("/**") {
            return String(trimmed.dropLast(3))
        }
        return trimmed
    }

    private static func defaultArchiveIncludePaths() -> [String] {
        [
            "clips",
            "stems",
            "annotations",
            "preprocessed",
            captionsFileName,
            datasetConfigFileName,
            datasetManifestFileName,
            trainingBundleFileName,
            fullMixBundleFileName,
            vocalBundleFileName,
            instrumentalBundleFileName,
            remotePackageFileName,
            validationReportFileName,
            preprocessCacheFileName,
        ]
    }

    private static func copyArchiveItem(from sourceURL: URL, to destinationURL: URL) throws {
        let name = sourceURL.lastPathComponent
        guard shouldIncludeArchiveItem(named: name) else { return }

        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: sourceURL.path(percentEncoded: false), isDirectory: &isDirectory) else {
            return
        }

        if isDirectory.boolValue {
            try FileManager.default.createDirectory(at: destinationURL, withIntermediateDirectories: true)
            let children = try FileManager.default.contentsOfDirectory(
                at: sourceURL,
                includingPropertiesForKeys: [.isDirectoryKey],
                options: []
            )
            for child in children {
                guard shouldIncludeArchiveItem(named: child.lastPathComponent) else { continue }
                try copyArchiveItem(from: child, to: destinationURL.appendingPathComponent(child.lastPathComponent))
            }
        } else {
            try FileManager.default.createDirectory(
                at: destinationURL.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            try FileManager.default.copyItem(at: sourceURL, to: destinationURL)
        }
    }

    private static func shouldIncludeArchiveItem(named name: String) -> Bool {
        guard !name.isEmpty else { return false }
        if name == ".DS_Store" { return false }
        if name.hasPrefix("._") { return false }
        if name == reviewBackupsDirectoryName { return false }
        return true
    }

    private static func datasetConfig() -> [String: Any] {
        [
            "pimp_audio_dataset_config_version": "1.0",
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
            "pimp_audio_dataset_version": "1.0",
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
            "clip_sampling": [
                "mode": config.samplingMode.rawValue,
                "seed": config.samplingSeed,
            ],
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
            "annotation": [
                "schema_version": annotationSchemaVersion,
                "status": "manual_review_required",
                "default_backend": "manual",
                "caption_sources": ["filename", "manual", "sidecar", "native", "ace-step-captioner", "remote"],
                "lyrics_sources": ["none", "manual", "sidecar", "native-qwen3-asr", "ace-step-transcriber", "remote"],
                "review_gate": "manual_approval_required",
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
                "preprocess_cache": preprocessCacheFileName,
                "preprocess_cache_status": "declared",
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
            "pimp_audio_training_bundle_version": "1.0",
            "created_at": createdAt,
            "dataset_dir": ".",
            "include": ["full_mix"],
            "training_lyrics": "[Instrumental]",
            "training_lyrics_source": "constant",
            "annotation_schema_version": annotationSchemaVersion,
            "preprocess_cache": preprocessCacheFileName,
            "clip_count": clips.count,
            "clips": clips,
        ]
    }

    private static func remotePackageDescriptor(
        outputDirectory: URL,
        clipCount: Int,
        createdAt: String,
        buildMode: LocalDatasetBuildMode = .clipExtraction
    ) -> [String: Any] {
        let isWholeTrackPackage = buildMode == .wholeTracks
        var fullMixBundle: [String: Any] = [
            "name": "full_mix",
            "path": fullMixBundleFileName,
            "include": [isWholeTrackPackage ? "full_mix" : "clip"],
            "training_lyrics_source": "per_asset_annotation",
        ]
        fullMixBundle["asset_role"] = isWholeTrackPackage ? "full_mix" : "clip"
        let mediaRules = isWholeTrackPackage
            ? ["sources/**"]
            : ["sources/**", "clips/**"]
        let notes = isWholeTrackPackage
            ? [
                "Complete source tracks are canonical and full mixes are selected by default.",
                "ACE-Step preprocessing uses at most 240 seconds per track; dataset_manifest.json records the effective range.",
                "Stem and clip assets are optional and remain linked to their source track.",
            ]
            : [
                "Complete source tracks are canonical; fixed-length clips are derived assets selected by this plan.",
                "Stem assets remain linked to their parent clip and are opt-in.",
                "Remote workers should copy sources and clips together so provenance remains intact.",
            ]
        return [
            "pimp_audio_remote_package_version": "1.0",
            "created_at": createdAt,
            "package_name": outputDirectory.lastPathComponent,
            "package_root": ".",
            "clip_count": clipCount,
            "default_bundle": fullMixBundleFileName,
            "bundles": [
                fullMixBundle,
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
            "preprocess_cache": preprocessCacheFileName,
            "copy_rules": [
                "include": mediaRules + [
                    "stems/**",
                    "annotations/**",
                    "preprocessed/**",
                    captionsFileName,
                    datasetConfigFileName,
                    datasetManifestFileName,
                    trainingBundleFileName,
                    fullMixBundleFileName,
                    vocalBundleFileName,
                    instrumentalBundleFileName,
                    remotePackageFileName,
                    validationReportFileName,
                    preprocessCacheFileName,
                ],
                "exclude": [],
            ],
            "worker_contract": [
                "resolve_relative_paths_from": "package_root",
                "entry_bundle": fullMixBundleFileName,
                "preprocess_cache": preprocessCacheFileName,
                "job_descriptor": RemoteTrainingJobCodec.jobFileName,
                "result_descriptor": RemoteTrainingJobCodec.resultFileName,
                "logs_file": RemoteTrainingJobCodec.logsFileName,
                "metrics_file": RemoteTrainingJobCodec.metricsFileName,
                "secrets_required": [],
                "expected_gpu_class": "h100_or_h200",
            ],
            "notes": notes,
        ]
    }

    private static func packageClips(
        from clips: [TrainingImportedFile],
        sourceFiles: [URL],
        outputDirectory: URL,
        config: LocalFilesDatasetConfig,
        audioMetadata: [String: AudioFileMetadata],
        captionSource: String,
        existingCaptionRecordsByFile: [String: [String: Any]]
    ) -> [[String: Any]] {
        clips.enumerated().map { index, clip in
            let relativePath = relativeClipPath(for: clip.url, from: outputDirectory)
            let caption = normalizedCaption(for: clip)
            let resolvedCaptionSource = caption.isEmpty ? "none" : captionSource
            let annotationBackend = resolvedCaptionSource == "filename" ? "filename" : "manual"
            let metadata = audioMetadata[normalizedFilePath(clip.url)] ?? fallbackAudioMetadata(for: clip.url)
            var object: [String: Any] = [
                "id": String(format: "clip_%04d", index + 1),
                "file": relativePath,
                "assets": [
                    "full_mix": relativePath,
                ],
                "caption": caption,
                "prompt": caption,
                "caption_source": resolvedCaptionSource,
                "lyrics": "",
                "lyrics_source": "none",
                "transcript": "",
                "transcription": [
                    "status": "not_run",
                    "text": "",
                ],
                "annotation": annotationMetadata(
                    captionSource: resolvedCaptionSource,
                    lyricsSource: "none",
                    backend: annotationBackend,
                    modelID: "",
                    modelRevision: "",
                    language: "",
                    caption: caption,
                    lyrics: "",
                    reviewStatus: "needs_review",
                    rawCaptionOutputFile: "",
                    rawLyricsOutputFile: "",
                    updatedAt: Date()
                ),
                "analysis": [
                    "status": "not_run",
                ],
                "audio": metadata.jsonObject,
                "clip_length_seconds": metadata.durationSeconds.map(seconds) ?? seconds(config.clipLength),
                "source": sourceMetadata(for: clip, sourceFiles: sourceFiles),
                "tags": [],
                "negative_tags": [],
            ]
            applySourceAuditFields(to: &object, source: object["source"] as? [String: Any])
            preserveEnrichmentMetadata(
                from: existingCaptionRecordsByFile[relativePath],
                into: &object
            )
            object["caption"] = caption
            object["prompt"] = caption
            object["caption_source"] = resolvedCaptionSource
            updateAnnotationCaptionSource(
                in: &object,
                captionSource: resolvedCaptionSource,
                caption: caption
            )
            object["file"] = relativePath
            object["assets"] = [
                "full_mix": relativePath,
            ]
            applySourceAuditFields(to: &object, source: sourceMetadata(for: clip, sourceFiles: sourceFiles))
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

    private static func sourceMetadata(for clip: TrainingImportedFile, sourceFiles: [URL]) -> [String: Any] {
        var metadata: [String: Any] = [
            "kind": "local_file",
        ]
        if let source = clip.sourceClip?.fileName ?? sourceFileName(for: clip.url, sourceFiles: sourceFiles) {
            metadata["file_name"] = source
        }
        if let startSeconds = clip.sourceClip?.startSeconds {
            metadata["start_seconds"] = seconds(startSeconds)
            metadata["source_start_seconds"] = seconds(startSeconds)
        }
        if let startFrame = clip.sourceClip?.startFrame {
            metadata["start_frame"] = startFrame
            metadata["source_start_frame"] = startFrame
        }
        if let frameCount = clip.sourceClip?.frameCount {
            metadata["frame_count"] = frameCount
        }
        if let sampleRate = clip.sourceClip?.sampleRate {
            metadata["sample_rate"] = sampleRate
        }
        return metadata
    }

    private static func applySourceAuditFields(to object: inout [String: Any], source: [String: Any]?) {
        guard let source else { return }
        object["source"] = source
        if let fileName = source["file_name"] {
            object["source_file_name"] = fileName
        }
        if let startSeconds = source["source_start_seconds"] ?? source["start_seconds"] {
            object["source_start_seconds"] = startSeconds
        }
        if let startFrame = source["source_start_frame"] ?? source["start_frame"] {
            object["source_start_frame"] = startFrame
        }
        if let frameCount = source["frame_count"] {
            object["source_frame_count"] = frameCount
        }
        if let sampleRate = source["sample_rate"] {
            object["source_sample_rate"] = sampleRate
        }
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
        if readPackageV2(from: manifestURL) != nil {
            return
        }
        guard var manifest = readJSONObject(from: manifestURL) else { return }

        let captions = readJSONArray(from: outputDirectory.appendingPathComponent(captionsFileName))
        guard !captions.isEmpty else { return }
        manifest["clips"] = captions
        manifest["clip_count"] = captions.count

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
        let keyValues = keyMetadataValues(in: captions)
        let keyStatus = keyMetadataStatus(in: captions)
        let hasDerivedMetadata = separatedClipCount > 0 || transcribedClipCount > 0 ||
            vocalBundleExists || instrumentalBundleExists || !bpmEstimates.isEmpty ||
            !keyValues.isEmpty || approvedClipCount(in: captions) > 0
        guard hasDerivedMetadata else {
            try refreshPreprocessCacheManifest(at: outputDirectory, createdAt: updatedAt)
            return
        }

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

        var annotation = manifest["annotation"] as? [String: Any] ?? [:]
        annotation["schema_version"] = annotationSchemaVersion
        annotation["updated_at"] = updatedAtString
        annotation["status"] = annotationStatus(in: captions)
        annotation["approved_clip_count"] = approvedClipCount(in: captions)
        annotation["caption_sources"] = sortedUnique(captions.compactMap {
            stringValue($0["caption_source"])
                ?? stringValue(($0["annotation"] as? [String: Any])?["caption_source"])
        })
        annotation["lyrics_sources"] = sortedUnique(captions.compactMap {
            stringValue($0["lyrics_source"])
                ?? stringValue(($0["annotation"] as? [String: Any])?["lyrics_source"])
        })
        manifest["annotation"] = annotation

        var analysis = manifest["analysis"] as? [String: Any] ?? [:]
        let hasAnalysis = !bpmEstimates.isEmpty || !keyValues.isEmpty
        analysis["enabled"] = hasAnalysis
        analysis["status"] = hasAnalysis ? "stem_analysis_available" : (analysis["status"] ?? "not_run")
        analysis["bpm_status"] = !bpmEstimates.isEmpty ? "estimated_from_stems" : "not_run"
        analysis["key_status"] = keyStatus
        if !bpmEstimates.isEmpty {
            analysis["stem_tempo_bpm"] = [
                "average": rounded(bpmEstimates.reduce(0, +) / Double(bpmEstimates.count), places: 1),
                "estimate_count": bpmEstimates.count,
            ]
        }
        if !keyValues.isEmpty {
            analysis["keys"] = keyValues
        }
        manifest["analysis"] = analysis

        var training = manifest["training"] as? [String: Any] ?? [:]
        training["default_bundle"] = trainingBundleFileName
        training["full_mix_bundle"] = fullMixBundleFileName
        training["dataset_config"] = datasetConfigFileName
        training["preprocess_cache"] = preprocessCacheFileName
        training["preprocess_cache_status"] = "declared"
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
            "key_status": keyStatus,
        ]

        try writeJSONObject(manifest, to: manifestURL)
        try refreshPreprocessCacheManifest(at: outputDirectory, createdAt: updatedAt)
    }

    private static func derivedStatus(completed: Int, total: Int) -> String {
        guard completed > 0 else { return "not_run" }
        return completed >= total ? "complete" : "partial"
    }

    private static func preprocessBundleEntry(
        name: String,
        bundleFileName: String,
        cacheDirectory: String,
        outputDirectory: URL
    ) -> [String: Any] {
        let bundleURL = outputDirectory.appendingPathComponent(bundleFileName)
        let bundle = readJSONObject(from: bundleURL)
        let clips = bundle?["clips"] as? [[String: Any]] ?? []
        let include = bundle?["include"] as? [String] ?? [name]
        let exists = FileManager.default.fileExists(atPath: bundleURL.path(percentEncoded: false))
        let indexFile = "\(cacheDirectory)/index.json"
        let index = readJSONObject(from: outputDirectory.appendingPathComponent(indexFile))
        let cachedEntries = index?["entries"] as? [[String: Any]] ?? []
        let cachedCount = cachedEntries.count
        let status: String
        if exists, !clips.isEmpty, cachedCount >= clips.count {
            status = "complete"
        } else if exists, !clips.isEmpty, cachedCount > 0 {
            status = "partial"
        } else if exists, !clips.isEmpty {
            status = "ready_for_preprocess"
        } else {
            status = "waiting_for_bundle"
        }
        return [
            "name": name,
            "bundle": bundleFileName,
            "include": include,
            "cache_dir": cacheDirectory,
            "clip_count": clips.count,
            "cached_clip_count": cachedCount,
            "status": status,
            "fingerprint": preprocessFingerprint(bundleFileName: bundleFileName, clips: clips),
            "tensor_files": [
                "samples": "\(cacheDirectory)/samples/*.safetensors",
                "index": indexFile,
            ],
        ]
    }

    private static func preprocessFingerprint(bundleFileName: String, clips: [[String: Any]]) -> String {
        let parts = clips.map { clip in
            [
                stringValue(clip["id"]) ?? "",
                selectedAudioPath(from: clip) ?? "",
                stringValue(clip["caption"]) ?? stringValue(clip["prompt"]) ?? "",
                stringValue(clip["lyrics"]) ?? stringValue(clip["transcript"]) ?? "",
            ].joined(separator: "|")
        }
        let input = ([bundleFileName] + parts).joined(separator: "\n")
        var hash: UInt64 = 1_469_598_103_934_665_603
        for byte in input.utf8 {
            hash ^= UInt64(byte)
            hash &*= 1_099_511_628_211
        }
        return String(format: "%016llx", hash)
    }

    private static func annotationMetadata(
        captionSource: String,
        lyricsSource: String,
        backend: String,
        modelID: String,
        modelRevision: String,
        language: String,
        caption: String,
        lyrics: String,
        reviewStatus: String,
        rawCaptionOutputFile: String,
        rawLyricsOutputFile: String,
        updatedAt: Date
    ) -> [String: Any] {
        DatasetAnnotationMetadata(
            updatedAt: updatedAt,
            backend: DatasetAnnotationBackendDescriptor(
                id: backend,
                modelID: modelID,
                modelRevision: modelRevision
            ),
            captionSource: captionSource,
            lyricsSource: lyricsSource,
            language: language,
            caption: caption,
            lyrics: lyrics,
            reviewStatus: reviewStatus,
            rawCaptionOutputFile: rawCaptionOutputFile,
            rawLyricsOutputFile: rawLyricsOutputFile
        ).persistedObject
    }

    private static func annotationStatus(in captions: [[String: Any]]) -> String {
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

    private static func approvedClipCount(in captions: [[String: Any]]) -> Int {
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

    private static func keyMetadataValues(in captions: [[String: Any]]) -> [String] {
        sortedUnique(captions.compactMap { record in
            if let key = stringValue(record["key"]) ?? stringValue(record["keyscale"]) {
                return key
            }
            if let analysis = record["analysis"] as? [String: Any] {
                return stringValue(analysis["key"]) ?? stringValue(analysis["keyscale"])
            }
            return nil
        })
    }

    private static func keyMetadataStatus(in captions: [[String: Any]]) -> String {
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

    private static func stringValue(_ value: Any?) -> String? {
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

    private static func rounded(_ value: Double, places: Int) -> Double {
        let scale = pow(10.0, Double(places))
        return (value * scale).rounded() / scale
    }

    private static func seconds(_ value: TimeInterval) -> Double {
        (value * 1000).rounded() / 1000
    }

    private static func compactTimestamp(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyyMMdd_HHmmss"
        return formatter.string(from: date)
    }

    private static func slug(_ string: String) -> String {
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "-_"))
        let scalars = string.lowercased().unicodeScalars.map { scalar -> Character in
            allowed.contains(scalar) ? Character(scalar) : "-"
        }
        let collapsed = String(scalars).split(separator: "-").joined(separator: "-")
        return collapsed.isEmpty ? "snapshot" : collapsed
    }

    private static func writeValidationReport(
        _ validation: TrainingDatasetPackageValidationResult,
        to outputDirectory: URL
    ) throws {
        let object: [String: Any] = [
            "pimp_audio_package_validation_version": "1.0",
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
        clips: [TrainingImportedFile],
        sourceFiles: [URL],
        captionSource: String,
        to outputDir: URL,
        existingRecordsByFile: [String: [String: Any]]
    ) throws {
        let entries = clips.map { clip in
            let file = relativeClipPath(for: clip.url, from: outputDir)
            var entry = existingRecordsByFile[file] ?? [:]
            entry["file"] = file
            let caption = normalizedCaption(for: clip)
            entry["caption"] = caption
            entry["caption_source"] = caption.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                ? "none"
                : captionSource
            applySourceAuditFields(to: &entry, source: sourceMetadata(for: clip, sourceFiles: sourceFiles))
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
            "annotation",
            "caption_source",
            "lyrics",
            "lyrics_source",
            "negative_tags",
            "separation",
            "tags",
            "transcript",
            "transcription",
        ] where existing[key] != nil {
            object[key] = existing[key]
        }
    }

    private static func updateAnnotationCaptionSource(
        in object: inout [String: Any],
        captionSource: String,
        caption: String
    ) {
        var annotation = object["annotation"] as? [String: Any] ?? [:]
        annotation["caption_source"] = captionSource
        annotation["caption_present"] = !caption.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        object["annotation"] = annotation
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

    private static func boolValue(_ value: Any?) -> Bool? {
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

/// Cancellable `/usr/bin/ditto` zip process for package export.
private final class DittoZipProcess: @unchecked Sendable {
    private let process = Process()
    private let stderr = Pipe()
    private let lock = NSLock()
    private var continuation: CheckedContinuation<Void, Error>?
    private var didResume = false
    private var didCancel = false

    init(sourceDirectory: URL, destinationURL: URL) {
        process.executableURL = URL(fileURLWithPath: "/usr/bin/ditto")
        process.arguments = [
            "-c",
            "-k",
            "--keepParent",
            "--norsrc",
            "--noextattr",
            "--noqtn",
            "--noacl",
            sourceDirectory.path,
            destinationURL.path,
        ]
        process.standardError = stderr
    }

    func run() async throws {
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
                start(continuation: continuation)
            }
        } onCancel: {
            cancel()
        }
    }

    private func start(continuation: CheckedContinuation<Void, Error>) {
        lock.lock()
        if didCancel {
            lock.unlock()
            continuation.resume(throwing: CancellationError())
            return
        }
        self.continuation = continuation
        process.terminationHandler = { [weak self] process in
            self?.finish(process: process)
        }
        do {
            try process.run()
            lock.unlock()
        } catch {
            self.continuation = nil
            lock.unlock()
            continuation.resume(throwing: error)
        }
    }

    private func finish(process: Process) {
        lock.lock()
        guard !didResume else {
            lock.unlock()
            return
        }
        didResume = true
        let cont = continuation
        continuation = nil
        let cancelled = didCancel
        lock.unlock()

        if cancelled {
            cont?.resume(throwing: CancellationError())
            return
        }

        guard process.terminationStatus == 0 else {
            let data = stderr.fileHandleForReading.readDataToEndOfFile()
            let message = String(data: data, encoding: .utf8)?
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let detail = message?.isEmpty == false
                ? message!
                : "ditto exited with status \(process.terminationStatus)"
            cont?.resume(throwing: TrainingDatasetPackageError.exportFailed(detail))
            return
        }
        cont?.resume(returning: ())
    }

    private func cancel() {
        lock.lock()
        didCancel = true
        let running = process.isRunning
        lock.unlock()
        if running {
            process.terminate()
        }
    }
}
