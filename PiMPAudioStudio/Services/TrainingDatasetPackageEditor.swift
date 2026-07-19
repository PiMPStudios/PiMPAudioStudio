import Foundation
import PiMPAudioCore

struct DatasetPackageSummary: Identifiable, Hashable {
    let directory: URL
    let name: String
    let createdAt: String?
    let clipCount: Int
    let isValid: Bool
    let validationSummary: String
    let enrichmentStatus: String
    let transcriptionStatus: String
    let bpmStatus: String
    let keyStatus: String

    var id: String { directory.standardizedFileURL.path(percentEncoded: false) }
}

struct DatasetPackageDocument {
    let directory: URL
    var name: String
    var createdAt: String?
    var clips: [DatasetClipReview]
    var validation: TrainingDatasetPackageValidationResult
    var packageV2: DatasetPackageV2? = nil
    var hasUnsavedChanges: Bool = false

    var approvedCount: Int {
        clips.filter { $0.includeInTraining && $0.isApproved }.count
    }

    var includedCount: Int {
        clips.filter(\.includeInTraining).count
    }

    var blockerCount: Int {
        clips.reduce(0) { count, clip in
            count + TrainingDatasetPackageEditor.issues(for: clip, datasetDirectory: directory)
                .filter { $0.severity == .blocker }
                .count
        } + validation.errors.count + validation.missingPackageFiles.count + validation.missingClipFiles.count
    }

    var warningCount: Int {
        clips.reduce(0) { count, clip in
            count + TrainingDatasetPackageEditor.issues(for: clip, datasetDirectory: directory)
                .filter { $0.severity == .warning }
                .count
        } + multipliedSourceTrackCount
    }

    var multipliedSourceTrackCount: Int {
        Dictionary(
            grouping: clips.filter {
                $0.includeInTraining && $0.sourceTrackID != nil
            },
            by: \.sourceTrackID
        )
        .values
        .filter { $0.count > 1 }
        .count
    }

    var isTrainingReady: Bool {
        blockerCount == 0 && includedCount > 0
    }
}

struct DatasetClipReview: Identifiable {
    var id: String
    var file: String
    var assets: [String: String]
    var caption: String
    var lyrics: String
    var captionSource: String
    var lyricsSource: String
    var annotationBackend: String
    var annotationModelID: String
    var annotationModelRevision: String
    var rawCaptionOutputFile: String
    var rawLyricsOutputFile: String
    var reviewStatus: String
    var bpm: String
    var key: String
    var language: String
    var timeSignature: String
    var role: String
    var notes: String
    var isApproved: Bool
    var durationSeconds: Double?
    var audioStatus: String
    var separationStatus: String
    var transcriptionStatus: String
    var assetID: String?
    var sourceTrackID: String?
    var derivedFromAssetID: String?
    var includeInTraining: Bool
    var sourceStartSeconds: Double?
    var sourceEndSeconds: Double?
    var trainingStartSeconds: Double?
    var trainingEndSeconds: Double?
    var trainingRangePolicy: String
    var rawObject: [String: Any]

    var displayName: String {
        URL(fileURLWithPath: file).lastPathComponent
    }

    var primaryAudioPath: String? {
        !file.isEmpty ? file : assets["full_mix"]
    }

    func audioURL(relativeTo datasetDirectory: URL) -> URL? {
        guard let path = !file.isEmpty ? file : assets["full_mix"] else { return nil }
        return TrainingDatasetPackageEditor.resolve(path, relativeTo: datasetDirectory)
    }

    var effectiveTrainingDurationSeconds: Double? {
        guard let trainingStartSeconds, let trainingEndSeconds else { return nil }
        return trainingEndSeconds - trainingStartSeconds
    }

    func persistedObject(updatedAt: Date = Date()) -> [String: Any] {
        var object = rawObject
        let trimmedCaption = caption.trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedLyrics = lyrics.trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedBPM = bpm.trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedKey = key.trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedLanguage = language.trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedTimeSignature = timeSignature.trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedRole = role.trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedNotes = notes.trimmingCharacters(in: .whitespacesAndNewlines)

        object["id"] = id
        object["file"] = file
        object["assets"] = assets
        object["caption"] = trimmedCaption
        object["prompt"] = trimmedCaption
        object["caption_source"] = trimmedCaption.isEmpty ? "none" : (captionSource.isEmpty ? "manual" : captionSource)
        object["lyrics"] = trimmedLyrics
        object["lyrics_source"] = trimmedLyrics.isEmpty ? "none" : (lyricsSource.isEmpty ? "manual" : lyricsSource)
        object["transcript"] = trimmedLyrics

        var transcription = object["transcription"] as? [String: Any] ?? [:]
        transcription["text"] = trimmedLyrics
        transcription["status"] = trimmedLyrics.isEmpty ? "missing" : "manual_reviewed"
        if !trimmedLyrics.isEmpty {
            transcription["source"] = lyricsSource.isEmpty ? "dataset_review" : lyricsSource
        }
        object["transcription"] = transcription

        var analysis = object["analysis"] as? [String: Any] ?? [:]
        if let bpmValue = Double(trimmedBPM) {
            analysis["bpm"] = bpmValue
            analysis["tempo_bpm_estimate"] = bpmValue
            object["bpm"] = bpmValue
        } else if trimmedBPM.isEmpty {
            analysis.removeValue(forKey: "bpm")
            analysis.removeValue(forKey: "tempo_bpm_estimate")
            object.removeValue(forKey: "bpm")
        } else {
            analysis["bpm"] = trimmedBPM
            object["bpm"] = trimmedBPM
        }

        if trimmedKey.isEmpty {
            analysis.removeValue(forKey: "key")
            analysis.removeValue(forKey: "keyscale")
            object.removeValue(forKey: "key")
            object.removeValue(forKey: "keyscale")
        } else {
            analysis["key"] = trimmedKey
            analysis["keyscale"] = trimmedKey
            object["key"] = trimmedKey
            object["keyscale"] = trimmedKey
        }

        object["language"] = trimmedLanguage
        object["timesignature"] = trimmedTimeSignature
        object["role"] = trimmedRole
        object["notes"] = trimmedNotes
        object["asset_id"] = assetID
        object["source_track_id"] = sourceTrackID
        object["derived_from"] = derivedFromAssetID
        object["asset_role"] = trimmedRole
        object["include_in_training"] = includeInTraining
        if let sourceStartSeconds {
            var sourceRange: [String: Any] = ["start_seconds": sourceStartSeconds]
            if let sourceEndSeconds {
                sourceRange["end_seconds"] = sourceEndSeconds
            }
            object["source_range"] = sourceRange
        }
        if let trainingStartSeconds {
            var trainingRange: [String: Any] = ["start_seconds": trainingStartSeconds]
            if let trainingEndSeconds {
                trainingRange["end_seconds"] = trainingEndSeconds
            }
            object["training_range"] = trainingRange
        }
        if !trainingRangePolicy.isEmpty {
            object["range_policy"] = trainingRangePolicy
        }
        object["analysis"] = analysis

        let persistedReviewStatus = isApproved ? "approved" : (reviewStatus.isEmpty ? "needs_review" : reviewStatus)
        let existingAnnotation = object["annotation"] as? [String: Any] ?? [:]
        let rawOutputs = (existingAnnotation["raw_output_files"] as? [String: Any] ?? [:])
            .compactMapValues { $0 as? String }
        let generationSettings = (existingAnnotation["generation_settings"] as? [String: Any] ?? [:])
            .compactMapValues { value -> String? in
                if let string = value as? String { return string }
                if let number = value as? NSNumber { return number.stringValue }
                return nil
            }
        object["annotation"] = DatasetAnnotationMetadata(
            updatedAt: updatedAt,
            backend: DatasetAnnotationBackendDescriptor(
                id: annotationBackend.isEmpty ? "manual" : annotationBackend,
                modelID: annotationModelID,
                modelRevision: annotationModelRevision,
                promptVersion: existingAnnotation["prompt_version"] as? String ?? "",
                quantization: existingAnnotation["quantization"] as? String ?? ""
            ),
            captionSource: trimmedCaption.isEmpty ? "none" : (captionSource.isEmpty ? "manual" : captionSource),
            lyricsSource: trimmedLyrics.isEmpty ? "none" : (lyricsSource.isEmpty ? "manual" : lyricsSource),
            language: trimmedLanguage,
            caption: trimmedCaption,
            lyrics: trimmedLyrics,
            reviewStatus: persistedReviewStatus,
            promptVersion: existingAnnotation["prompt_version"] as? String ?? "",
            quantization: existingAnnotation["quantization"] as? String ?? "",
            generationSettings: generationSettings,
            elapsedSeconds: (existingAnnotation["elapsed_seconds"] as? NSNumber)?.doubleValue,
            rawOutputFiles: rawOutputs,
            rawCaptionOutputFile: rawCaptionOutputFile,
            rawLyricsOutputFile: rawLyricsOutputFile
        ).persistedObject

        object["training_review"] = [
            "approved": isApproved,
            "status": persistedReviewStatus,
            "updated_at": ISO8601DateFormatter().string(from: updatedAt),
            "notes": trimmedNotes,
        ]
        return object
    }

    func packageV2Annotation(updatedAt: Date) -> DatasetAssetAnnotationV2 {
        let trimmedCaption = caption.trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedLyrics = lyrics.trimmingCharacters(in: .whitespacesAndNewlines)
        let language = language.trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedNotes = notes.trimmingCharacters(in: .whitespacesAndNewlines)
        let existingAnnotation = rawObject["annotation"] as? [String: Any] ?? [:]
        let existingTranscription = rawObject["transcription"] as? [String: Any] ?? [:]
        let promptVersion = existingAnnotation["prompt_version"] as? String
        let quantization = existingAnnotation["quantization"] as? String
        let generationSettings = (existingAnnotation["generation_settings"] as? [String: Any] ?? [:])
            .compactMapValues { value -> String? in
                if let string = value as? String { return string }
                if let number = value as? NSNumber { return number.stringValue }
                return nil
            }
        let elapsedSeconds = (existingAnnotation["elapsed_seconds"] as? NSNumber)?.doubleValue
        let lyricsPromptVersion = existingTranscription["prompt_version"] as? String
            ?? promptVersion
        let lyricsQuantization = existingTranscription["quantization"] as? String
            ?? quantization
        let lyricsGenerationSettings = (
            existingTranscription["generation_settings"] as? [String: Any] ?? [:]
        )
        .compactMapValues { value -> String? in
            if let string = value as? String { return string }
            if let number = value as? NSNumber { return number.stringValue }
            return nil
        }
        let lyricsElapsedSeconds = (
            existingTranscription["elapsed_seconds"] as? NSNumber
        )?.doubleValue ?? elapsedSeconds
        let captionEntry = trimmedCaption.isEmpty ? nil : DatasetTextAnnotationV2(
            text: trimmedCaption,
            source: captionSource.isEmpty ? "manual" : captionSource,
            language: language.isEmpty ? nil : language,
            provenance: DatasetAnnotationProvenanceV2(
                backendID: annotationBackend.isEmpty ? "manual" : annotationBackend,
                modelID: annotationModelID.isEmpty ? nil : annotationModelID,
                modelRevision: annotationModelRevision.isEmpty ? nil : annotationModelRevision,
                promptVersion: promptVersion,
                quantization: quantization,
                generatedAt: ISO8601DateFormatter().string(from: updatedAt),
                generationSettings: generationSettings,
                elapsedSeconds: elapsedSeconds,
                rawOutputFiles: rawCaptionOutputFile.isEmpty
                    ? [:]
                    : ["caption": rawCaptionOutputFile]
            )
        )
        let lyricsEntry = trimmedLyrics.isEmpty ? nil : DatasetTextAnnotationV2(
            text: trimmedLyrics,
            source: lyricsSource.isEmpty ? "manual" : lyricsSource,
            language: language.isEmpty ? nil : language,
            provenance: DatasetAnnotationProvenanceV2(
                backendID: existingTranscription["backend"] as? String
                    ?? (annotationBackend.isEmpty ? "manual" : annotationBackend),
                backendVersion: existingTranscription["backend_version"] as? String,
                modelID: existingTranscription["model"] as? String
                    ?? (annotationModelID.isEmpty ? nil : annotationModelID),
                modelRevision: existingTranscription["model_revision"] as? String
                    ?? (annotationModelRevision.isEmpty ? nil : annotationModelRevision),
                promptVersion: lyricsPromptVersion,
                quantization: lyricsQuantization,
                generatedAt: ISO8601DateFormatter().string(from: updatedAt),
                generationSettings: lyricsGenerationSettings.isEmpty
                    ? generationSettings
                    : lyricsGenerationSettings,
                elapsedSeconds: lyricsElapsedSeconds,
                rawOutputFiles: rawLyricsOutputFile.isEmpty
                    ? [:]
                    : ["lyrics": rawLyricsOutputFile]
            )
        )
        return DatasetAssetAnnotationV2(
            caption: captionEntry,
            lyrics: lyricsEntry,
            reviewStatus: isApproved ? "approved" : (reviewStatus.isEmpty ? "needs_review" : reviewStatus),
            approved: isApproved,
            reviewedAt: ISO8601DateFormatter().string(from: updatedAt),
            notes: trimmedNotes.isEmpty ? nil : trimmedNotes
        )
    }
}

enum DatasetPackageIssueSeverity: String, Hashable {
    case blocker
    case warning
}

struct DatasetPackageIssue: Identifiable, Hashable {
    let severity: DatasetPackageIssueSeverity
    let message: String

    var id: String { "\(severity.rawValue):\(message)" }
}

enum TrainingDatasetPackageEditorError: LocalizedError {
    case missingCaptions(URL)
    case invalidCaptions(URL)
    case missingManifest(URL)
    case invalidManifest(URL)

    var errorDescription: String? {
        switch self {
        case .missingCaptions(let url):
            return "\(url.lastPathComponent) does not contain a captions.json file."
        case .invalidCaptions(let url):
            return "\(url.lastPathComponent)'s captions.json is not a readable clip list."
        case .missingManifest(let url):
            return "\(url.lastPathComponent) does not contain a dataset_manifest.json file."
        case .invalidManifest(let url):
            return "\(url.lastPathComponent)'s dataset_manifest.json is not a supported dataset package."
        }
    }
}

private struct SidecarFields {
    var caption: String?
    var lyrics: String?
    var bpm: String?
    var key: String?
    var language: String?
    var timeSignature: String?
    var role: String?
    var notes: String?
    var captionSource: String?
    var lyricsSource: String?
    var metadataSource: String?
}

enum TrainingDatasetPackageEditor {
    static func listPackages(in outputBase: URL) -> [DatasetPackageSummary] {
        let datasetsDirectory = outputBase.appendingPathComponent("TrainingDatasets", isDirectory: true)
        guard let directories = try? FileManager.default.contentsOfDirectory(
            at: datasetsDirectory,
            includingPropertiesForKeys: [.isDirectoryKey, .contentModificationDateKey],
            options: [.skipsHiddenFiles]
        ) else {
            return []
        }

        return directories
            .filter { isDirectory($0) }
            .filter { isDatasetPackage($0) }
            .map { summary(for: $0) }
            .sorted { lhs, rhs in
                modificationDate(lhs.directory) > modificationDate(rhs.directory)
            }
    }

    static func packageSummary(at directory: URL) -> DatasetPackageSummary {
        summary(for: directory)
    }

    /// Loads either a native v2 package or a non-destructive v1 compatibility
    /// view. This is the schema migration seam; the existing clip review UI
    /// continues to use `loadPackage(at:)` until its v2 asset views are ready.
    static func loadPackageV2(at directory: URL) throws -> DatasetPackageV2 {
        let manifestURL = directory.appendingPathComponent(TrainingDatasetPackageWriter.datasetManifestFileName)
        guard FileManager.default.fileExists(atPath: manifestURL.path(percentEncoded: false)) else {
            throw TrainingDatasetPackageEditorError.missingManifest(directory)
        }
        let manifestData = try Data(contentsOf: manifestURL)
        guard let manifest = try JSONSerialization.jsonObject(with: manifestData) as? [String: Any] else {
            throw TrainingDatasetPackageEditorError.invalidManifest(directory)
        }
        let version = stringValue(manifest["pimp_audio_dataset_version"])
            ?? stringValue(manifest["anvil_dataset_version"])
            ?? "1.0"

        if version == DatasetPackageV2.currentVersion {
            return try DatasetPackageV2Codec.decodeValidated(manifestData)
        }
        guard version == "1.0" else {
            throw TrainingDatasetPackageEditorError.invalidManifest(directory)
        }

        let captionsURL = directory.appendingPathComponent(TrainingDatasetPackageWriter.captionsFileName)
        guard FileManager.default.fileExists(atPath: captionsURL.path(percentEncoded: false)) else {
            throw TrainingDatasetPackageEditorError.missingCaptions(directory)
        }
        return try DatasetPackageV1Adapter.adapt(
            manifestData: manifestData,
            captionsData: Data(contentsOf: captionsURL)
        )
    }

    static func loadPackage(at directory: URL) throws -> DatasetPackageDocument {
        let captionsURL = directory.appendingPathComponent(TrainingDatasetPackageWriter.captionsFileName)
        guard FileManager.default.fileExists(atPath: captionsURL.path(percentEncoded: false)) else {
            throw TrainingDatasetPackageEditorError.missingCaptions(directory)
        }
        guard let records = readJSONArray(from: captionsURL) else {
            throw TrainingDatasetPackageEditorError.invalidCaptions(directory)
        }

        let manifestURL = directory.appendingPathComponent(TrainingDatasetPackageWriter.datasetManifestFileName)
        let manifest = readJSONObject(from: manifestURL)
        let validation = TrainingDatasetPackageWriter.validatePackage(at: directory)
        if stringValue(manifest?["pimp_audio_dataset_version"]) == DatasetPackageV2.currentVersion {
            let package = try DatasetPackageV2Codec.decodeValidated(Data(contentsOf: manifestURL))
            let recordsByID = records.reduce(into: [String: [String: Any]]()) { result, record in
                guard let id = firstString(record, keys: ["asset_id", "id"]),
                      result[id] == nil else {
                    return
                }
                result[id] = record
            }
            let recordsByFile = records.reduce(into: [String: [String: Any]]()) { result, record in
                guard let file = firstString(record, keys: ["file", "path"]),
                      result[file] == nil else {
                    return
                }
                result[file] = record
            }
            let plan = package.trainingPlans.first(where: \.isDefault)
            let reviewAssets = package.assets.enumerated()
                .map { index, asset in
                    (
                        index: index,
                        asset: asset,
                        selection: plan?.selections.first { $0.assetID == asset.id }
                    )
                }
                .sorted { lhs, rhs in
                    let lhsIncluded = lhs.selection?.includeInTraining ?? false
                    let rhsIncluded = rhs.selection?.includeInTraining ?? false
                    if lhsIncluded != rhsIncluded {
                        return lhsIncluded && !rhsIncluded
                    }
                    return lhs.index < rhs.index
                }
            return DatasetPackageDocument(
                directory: directory,
                name: package.name,
                createdAt: package.createdAt,
                clips: reviewAssets.enumerated().map { index, item in
                    let asset = item.asset
                    let record = recordsByID[asset.id] ?? recordsByFile[asset.file.path]
                    return clip(
                        from: reviewObject(
                            for: asset,
                            selection: item.selection,
                            existingRecord: record
                        ),
                        index: index,
                        datasetDirectory: directory
                    )
                },
                validation: validation,
                packageV2: package
            )
        }
        return DatasetPackageDocument(
            directory: directory,
            name: stringValue(manifest?["name"]) ?? directory.lastPathComponent,
            createdAt: stringValue(manifest?["created_at"]),
            clips: records.enumerated().map { clip(from: $0.element, index: $0.offset, datasetDirectory: directory) },
            validation: validation
        )
    }

    static func save(_ document: DatasetPackageDocument) throws {
        let updatedAt = Date()
        _ = try TrainingDatasetPackageWriter.createReviewBackup(
            at: document.directory,
            reason: "dataset-review-save",
            createdAt: updatedAt
        )
        let records = document.clips.map { $0.persistedObject(updatedAt: updatedAt) }
        if var package = document.packageV2 {
            guard let defaultPlanIndex = package.trainingPlans.firstIndex(where: \.isDefault) else {
                throw TrainingDatasetPackageError.invalidPackage(
                    "Package v2 does not have a default training plan."
                )
            }
            for clip in document.clips {
                guard let assetID = clip.assetID,
                      let assetIndex = package.assets.firstIndex(where: { $0.id == assetID }) else {
                    continue
                }
                package.assets[assetIndex].annotation = clip.packageV2Annotation(updatedAt: updatedAt)
                if let selectionIndex = package.trainingPlans[defaultPlanIndex].selections.firstIndex(
                    where: { $0.assetID == assetID }
                ) {
                    package.trainingPlans[defaultPlanIndex]
                        .selections[selectionIndex]
                        .includeInTraining = clip.includeInTraining
                } else {
                    package.trainingPlans[defaultPlanIndex].selections.append(
                        DatasetTrainingSelectionV2(
                            assetID: assetID,
                            includeInTraining: clip.includeInTraining,
                            trainingRange: package.assets[assetIndex].sourceRange,
                            rangePolicy: .fullSource
                        )
                    )
                }
            }
            try TrainingDatasetPackageWriter.writeReviewedPackageV2(
                package,
                records: records,
                outputDirectory: document.directory,
                updatedAt: updatedAt
            )
            return
        }

        try writeJSONArray(records, to: document.directory.appendingPathComponent(TrainingDatasetPackageWriter.captionsFileName))
        try TrainingDatasetPackageWriter.refreshTrainingBundles(at: document.directory)

        if records.contains(where: { clip in
            let assets = clip["assets"] as? [String: Any] ?? [:]
            return assets["vocals"] != nil || assets["instrumental"] != nil
        }) {
            _ = try LocalDatasetEnrichmentService.exportStemTrainingBundles(datasetDirectory: document.directory)
        }

        try TrainingDatasetPackageWriter.refreshValidationReport(at: document.directory)
    }

    @discardableResult
    static func exportSidecars(from document: DatasetPackageDocument) throws -> Int {
        var exportedCount = 0
        var csvRows = [[
            "file",
            "caption",
            "lyrics",
            "bpm",
            "keyscale",
            "language",
            "timesignature",
            "role",
            "notes",
        ]]

        for clip in document.clips where clip.includeInTraining {
            guard let audioURL = clip.audioURL(relativeTo: document.directory) else { continue }
            let stemURL = audioURL.deletingPathExtension()
            let caption = clip.caption.trimmingCharacters(in: .whitespacesAndNewlines)
            let lyrics = clip.lyrics.trimmingCharacters(in: .whitespacesAndNewlines)

            if !caption.isEmpty {
                try caption.write(
                    to: stemURL.appendingPathExtension("caption.txt"),
                    atomically: true,
                    encoding: .utf8
                )
            }
            if !lyrics.isEmpty {
                try lyrics.write(
                    to: stemURL.appendingPathExtension("lyrics.txt"),
                    atomically: true,
                    encoding: .utf8
                )
            }

            var sidecar = readJSONObject(from: stemURL.appendingPathExtension("json")) ?? [:]
            sidecar["caption"] = caption
            sidecar["prompt"] = caption
            sidecar["lyrics"] = lyrics
            sidecar["transcript"] = lyrics
            sidecar["bpm"] = clip.bpm.trimmingCharacters(in: .whitespacesAndNewlines)
            sidecar["keyscale"] = clip.key.trimmingCharacters(in: .whitespacesAndNewlines)
            sidecar["language"] = clip.language.trimmingCharacters(in: .whitespacesAndNewlines)
            sidecar["timesignature"] = clip.timeSignature.trimmingCharacters(in: .whitespacesAndNewlines)
            sidecar["role"] = clip.role.trimmingCharacters(in: .whitespacesAndNewlines)
            sidecar["notes"] = clip.notes.trimmingCharacters(in: .whitespacesAndNewlines)
            sidecar["caption_source"] = clip.captionSource
            sidecar["lyrics_source"] = clip.lyricsSource
            sidecar["annotation"] = clip.persistedObject()["annotation"]
            try writeJSONObject(sidecar, to: stemURL.appendingPathExtension("json"))

            csvRows.append([
                clip.file,
                caption,
                lyrics,
                clip.bpm,
                clip.key,
                clip.language,
                clip.timeSignature,
                clip.role,
                clip.notes,
            ])
            exportedCount += 1
        }

        let csv = csvRows
            .map { row in row.map(csvEscape).joined(separator: ",") }
            .joined(separator: "\n") + "\n"
        try csv.write(
            to: document.directory.appendingPathComponent("metadata.csv"),
            atomically: true,
            encoding: .utf8
        )

        return exportedCount
    }

    @discardableResult
    static func importSidecars(into document: inout DatasetPackageDocument) -> Int {
        let csvFields = csvSidecarFields(in: document.directory)
        var changedFields = 0

        for index in document.clips.indices {
            var fields = sidecarFields(for: document.clips[index], datasetDirectory: document.directory)
            let keys = lookupKeys(for: document.clips[index], datasetDirectory: document.directory)
            for key in keys {
                if let csv = csvFields[key] {
                    fields = fields.merging(csv)
                }
            }
            changedFields += document.clips[index].apply(fields: fields)
        }

        if changedFields > 0 {
            document.hasUnsavedChanges = true
        }
        return changedFields
    }

    static func importedFiles(for document: DatasetPackageDocument) throws -> [TrainingImportedFile] {
        try TrainingDatasetImporter.importFiles(
            from: document.directory.appendingPathComponent(TrainingDatasetPackageWriter.fullMixBundleFileName)
        )
    }

    static func issues(for clip: DatasetClipReview, datasetDirectory: URL) -> [DatasetPackageIssue] {
        guard clip.includeInTraining else { return [] }
        var issues: [DatasetPackageIssue] = []

        if clip.caption.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            issues.append(DatasetPackageIssue(severity: .blocker, message: "Missing caption."))
        }
        if !clip.isApproved {
            issues.append(DatasetPackageIssue(severity: .blocker, message: "Not manually approved."))
        }
        if let audioURL = clip.audioURL(relativeTo: datasetDirectory) {
            if !FileManager.default.fileExists(atPath: audioURL.path(percentEncoded: false)) {
                issues.append(DatasetPackageIssue(severity: .blocker, message: "Audio file is missing."))
            }
        } else {
            issues.append(DatasetPackageIssue(severity: .blocker, message: "No audio path is set."))
        }
        if clip.role != DatasetAssetRoleV2.instrumental.rawValue,
           clip.lyrics.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            issues.append(DatasetPackageIssue(severity: .warning, message: "Lyrics/transcript are empty."))
        }
        if clip.bpm.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            issues.append(DatasetPackageIssue(severity: .warning, message: "BPM is missing."))
        }
        if clip.key.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            issues.append(DatasetPackageIssue(severity: .warning, message: "Key is missing."))
        }
        if (clip.role == DatasetAssetRoleV2.fullMix.rawValue
                || clip.role == DatasetAssetRoleV2.clip.rawValue),
           clip.separationStatus == "not_run" || clip.separationStatus.isEmpty {
            issues.append(DatasetPackageIssue(severity: .warning, message: "Stem separation has not been recorded."))
        }
        return issues
    }

    static func resolve(_ path: String, relativeTo baseDirectory: URL) -> URL {
        let expanded = (path as NSString).expandingTildeInPath
        if expanded.hasPrefix("/") {
            return URL(fileURLWithPath: expanded).standardizedFileURL
        }
        return baseDirectory.appendingPathComponent(expanded).standardizedFileURL
    }

    private static func isDatasetPackage(_ directory: URL) -> Bool {
        FileManager.default.fileExists(
            atPath: directory.appendingPathComponent(TrainingDatasetPackageWriter.captionsFileName)
                .path(percentEncoded: false)
        )
    }

    private static func summary(for directory: URL) -> DatasetPackageSummary {
        let manifest = readJSONObject(from: directory.appendingPathComponent(TrainingDatasetPackageWriter.datasetManifestFileName))
        let validation = TrainingDatasetPackageWriter.validatePackage(at: directory)
        let enrichment = manifest?["enrichment"] as? [String: Any]
        let transcription = manifest?["transcription"] as? [String: Any]
        let analysis = manifest?["analysis"] as? [String: Any]
        return DatasetPackageSummary(
            directory: directory,
            name: stringValue(manifest?["name"]) ?? directory.lastPathComponent,
            createdAt: stringValue(manifest?["created_at"]),
            clipCount: intValue(manifest?["clip_count"]) ?? validation.clipCount,
            isValid: validation.isValid,
            validationSummary: validation.summary,
            enrichmentStatus: stringValue(enrichment?["status"]) ?? "not_run",
            transcriptionStatus: stringValue(transcription?["status"]) ?? stringValue(enrichment?["transcription_status"]) ?? "not_run",
            bpmStatus: stringValue(analysis?["bpm_status"]) ?? stringValue(enrichment?["bpm_status"]) ?? "not_run",
            keyStatus: stringValue(analysis?["key_status"]) ?? stringValue(enrichment?["key_status"]) ?? "not_run"
        )
    }

    private static func clip(
        from object: [String: Any],
        index: Int,
        datasetDirectory: URL
    ) -> DatasetClipReview {
        let file = firstString(object, keys: ["file", "path", "audio"]) ?? ""
        let assets = stringDictionary(object["assets"]) ?? [:]
        let id = firstString(object, keys: ["id"]) ?? (!file.isEmpty ? file : "clip_\(index + 1)")
        let analysis = object["analysis"] as? [String: Any] ?? [:]
        let transcription = object["transcription"] as? [String: Any] ?? [:]
        let annotation = object["annotation"] as? [String: Any] ?? [:]
        let rawOutputFiles = annotation["raw_output_files"] as? [String: Any] ?? [:]
        let trainingReview = object["training_review"] as? [String: Any] ?? [:]
        let audio = object["audio"] as? [String: Any] ?? [:]
        let sourceRange = object["source_range"] as? [String: Any] ?? [:]
        let trainingRange = object["training_range"] as? [String: Any] ?? [:]
        let caption = firstString(object, keys: ["caption", "prompt"]) ?? ""
        let lyrics = firstString(object, keys: ["lyrics", "transcript"])
            ?? firstString(transcription, keys: ["text"])
            ?? ""
        let reviewStatus = firstString(annotation, keys: ["review_status"])
            ?? firstString(trainingReview, keys: ["status"])
            ?? "needs_review"
        let isApproved = boolValue(trainingReview["approved"]) ?? (reviewStatus == "approved")
        return DatasetClipReview(
            id: id,
            file: file,
            assets: assets,
            caption: caption,
            lyrics: lyrics,
            captionSource: firstString(object, keys: ["caption_source"])
                ?? firstString(annotation, keys: ["caption_source"])
                ?? (caption.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? "none" : "manual"),
            lyricsSource: firstString(object, keys: ["lyrics_source"])
                ?? firstString(annotation, keys: ["lyrics_source"])
                ?? firstString(transcription, keys: ["source"])
                ?? (lyrics.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? "none" : "manual"),
            annotationBackend: firstString(annotation, keys: ["annotation_backend"])
                ?? firstString(transcription, keys: ["backend"])
                ?? "manual",
            annotationModelID: firstString(annotation, keys: ["model_id"])
                ?? firstString(transcription, keys: ["model"])
                ?? "",
            annotationModelRevision: firstString(annotation, keys: ["model_revision"]) ?? "",
            rawCaptionOutputFile: firstString(rawOutputFiles, keys: ["caption"]) ?? "",
            rawLyricsOutputFile: firstString(rawOutputFiles, keys: ["lyrics"]) ?? "",
            reviewStatus: reviewStatus,
            bpm: stringValue(object["bpm"])
                ?? stringValue(analysis["bpm"])
                ?? stringValue(analysis["tempo_bpm_estimate"])
                ?? "",
            key: firstString(object, keys: ["key", "keyscale"])
                ?? firstString(analysis, keys: ["key", "keyscale"])
                ?? "",
            language: firstString(object, keys: ["language"]) ?? "",
            timeSignature: firstString(object, keys: ["timesignature", "time_signature"]) ?? "",
            role: firstString(object, keys: ["role"]) ?? inferredRole(from: assets),
            notes: firstString(object, keys: ["notes"]) ?? firstString(trainingReview, keys: ["notes"]) ?? "",
            isApproved: isApproved,
            durationSeconds: doubleValue(object["clip_length_seconds"]) ?? doubleValue(audio["duration_seconds"]),
            audioStatus: firstString(audio, keys: ["metadata_status"]) ?? "unknown",
            separationStatus: separationStatus(from: object, datasetDirectory: datasetDirectory),
            transcriptionStatus: firstString(transcription, keys: ["status"]) ?? "not_run",
            assetID: firstString(object, keys: ["asset_id"]),
            sourceTrackID: firstString(object, keys: ["source_track_id"]),
            derivedFromAssetID: firstString(object, keys: ["derived_from"]),
            includeInTraining: boolValue(object["include_in_training"]) ?? true,
            sourceStartSeconds: doubleValue(sourceRange["start_seconds"]),
            sourceEndSeconds: doubleValue(sourceRange["end_seconds"]),
            trainingStartSeconds: doubleValue(trainingRange["start_seconds"]),
            trainingEndSeconds: doubleValue(trainingRange["end_seconds"]),
            trainingRangePolicy: firstString(object, keys: ["range_policy"]) ?? "",
            rawObject: object
        )
    }

    private static func reviewObject(
        for asset: DatasetAssetV2,
        selection: DatasetTrainingSelectionV2?,
        existingRecord: [String: Any]?
    ) -> [String: Any] {
        var object = existingRecord ?? [:]
        var assets = stringDictionary(object["assets"]) ?? [:]
        assets[asset.role.rawValue] = asset.file.path
        object["id"] = asset.id
        object["asset_id"] = asset.id
        object["source_track_id"] = asset.sourceTrackID
        object["derived_from"] = asset.derivedFrom
        object["file"] = asset.file.path
        object["assets"] = assets
        object["role"] = asset.role.rawValue
        object["asset_role"] = asset.role.rawValue
        object["include_in_training"] = selection?.includeInTraining ?? false
        object["source_range"] = timeRangeObject(asset.sourceRange)
        if let selection {
            object["training_range"] = timeRangeObject(selection.trainingRange)
            object["range_policy"] = selection.rangePolicy.rawValue
        }

        var audio: [String: Any] = object["audio"] as? [String: Any] ?? [:]
        if let format = asset.file.format { audio["format"] = format }
        if let byteSize = asset.file.byteSize { audio["byte_size"] = byteSize }
        if let duration = asset.file.durationSeconds { audio["duration_seconds"] = duration }
        if let sampleRate = asset.file.sampleRate { audio["sample_rate"] = sampleRate }
        if let channels = asset.file.channels { audio["channels"] = channels }
        object["audio"] = audio
        if let duration = asset.file.durationSeconds {
            object["clip_length_seconds"] = duration
        }

        let annotation = asset.annotation
        let caption = annotation?.caption?.text ?? ""
        let lyrics = annotation?.lyrics?.text ?? ""
        object["caption"] = caption
        object["prompt"] = caption
        object["caption_source"] = annotation?.caption?.source ?? (caption.isEmpty ? "none" : "manual")
        object["lyrics"] = lyrics
        object["transcript"] = lyrics
        object["lyrics_source"] = annotation?.lyrics?.source ?? (lyrics.isEmpty ? "none" : "manual")
        object["language"] = annotation?.caption?.language ?? annotation?.lyrics?.language ?? ""
        object["notes"] = annotation?.notes ?? ""
        object["training_review"] = [
            "approved": annotation?.approved ?? false,
            "status": annotation?.reviewStatus ?? "needs_review",
            "updated_at": annotation?.reviewedAt ?? "",
            "notes": annotation?.notes ?? "",
        ]

        let captionProvenance = annotation?.caption?.provenance
        let lyricsProvenance = annotation?.lyrics?.provenance
        let provenance = captionProvenance ?? lyricsProvenance
        var rawOutputs = annotation?.caption?.provenance?.rawOutputFiles ?? [:]
        for (key, value) in annotation?.lyrics?.provenance?.rawOutputFiles ?? [:] {
            rawOutputs[key] = value
        }
        var legacyAnnotation: [String: Any] = [
            "schema_version": TrainingDatasetPackageWriter.annotationSchemaVersion,
            "annotation_backend": provenance?.backendID ?? "manual",
            "annotation_backend_version": provenance?.backendVersion ?? "",
            "model_id": provenance?.modelID ?? "",
            "model_revision": provenance?.modelRevision ?? "",
            "prompt_version": provenance?.promptVersion ?? "",
            "quantization": provenance?.quantization ?? "",
            "generation_settings": provenance?.generationSettings ?? [:],
            "caption_source": annotation?.caption?.source ?? "none",
            "lyrics_source": annotation?.lyrics?.source ?? "none",
            "review_status": annotation?.reviewStatus ?? "needs_review",
            "raw_output_files": rawOutputs,
        ]
        if let elapsedSeconds = provenance?.elapsedSeconds {
            legacyAnnotation["elapsed_seconds"] = elapsedSeconds
        }
        object["annotation"] = legacyAnnotation

        var transcription: [String: Any] = [
            "status": lyrics.isEmpty ? "not_run" : "complete",
            "text": lyrics,
            "source": annotation?.lyrics?.source ?? "none",
            "backend": lyricsProvenance?.backendID ?? "manual",
            "backend_version": lyricsProvenance?.backendVersion ?? "",
            "model": lyricsProvenance?.modelID ?? "",
            "model_revision": lyricsProvenance?.modelRevision ?? "",
            "prompt_version": lyricsProvenance?.promptVersion ?? "",
            "quantization": lyricsProvenance?.quantization ?? "",
            "generation_settings": lyricsProvenance?.generationSettings ?? [:],
        ]
        if let elapsedSeconds = lyricsProvenance?.elapsedSeconds {
            transcription["elapsed_seconds"] = elapsedSeconds
        }
        object["transcription"] = transcription
        return object
    }

    private static func timeRangeObject(_ range: DatasetTimeRangeV2) -> [String: Any] {
        var object: [String: Any] = ["start_seconds": range.startSeconds]
        if let endSeconds = range.endSeconds {
            object["end_seconds"] = endSeconds
        }
        return object
    }

    private static func separationStatus(from object: [String: Any], datasetDirectory: URL) -> String {
        guard let assets = stringDictionary(object["assets"]) else {
            return "not_run"
        }
        let hasVocals = assets["vocals"].map { FileManager.default.fileExists(atPath: resolve($0, relativeTo: datasetDirectory).path(percentEncoded: false)) } ?? false
        let hasInstrumental = assets["instrumental"].map { FileManager.default.fileExists(atPath: resolve($0, relativeTo: datasetDirectory).path(percentEncoded: false)) } ?? false
        if hasVocals && hasInstrumental {
            return "complete"
        }
        if hasVocals || hasInstrumental {
            return "partial"
        }
        return "not_run"
    }

    private static func sidecarFields(for clip: DatasetClipReview, datasetDirectory: URL) -> SidecarFields {
        guard let audioURL = clip.audioURL(relativeTo: datasetDirectory) else { return SidecarFields() }
        let stemURL = audioURL.deletingPathExtension()
        let rootStemURL = datasetDirectory.appendingPathComponent(audioURL.deletingPathExtension().lastPathComponent)
        var fields = SidecarFields()

        fields.caption = readTrimmedText(from: firstExisting([
            stemURL.appendingPathExtension("caption.txt"),
            rootStemURL.appendingPathExtension("caption.txt"),
        ]))
        fields.lyrics = cleanLyricText(readTrimmedText(from: firstExisting([
            stemURL.appendingPathExtension("lyrics.txt"),
            rootStemURL.appendingPathExtension("lyrics.txt"),
            stemURL.appendingPathExtension("lyrics.lrc"),
            rootStemURL.appendingPathExtension("lyrics.lrc"),
            stemURL.appendingPathExtension("lrc"),
            rootStemURL.appendingPathExtension("lrc"),
            stemURL.appendingPathExtension("txt"),
            rootStemURL.appendingPathExtension("txt"),
        ])))

        for jsonURL in [
            stemURL.appendingPathExtension("json"),
            rootStemURL.appendingPathExtension("json"),
        ] {
            guard let json = readJSONObject(from: jsonURL) else { continue }
            fields.caption = firstString(json, keys: ["caption", "prompt"]) ?? fields.caption
            fields.lyrics = cleanLyricText(firstString(json, keys: ["lyrics", "transcript"])) ?? fields.lyrics
            fields.bpm = stringValue(json["bpm"]) ?? fields.bpm
            fields.key = firstString(json, keys: ["key", "keyscale"]) ?? fields.key
            fields.language = firstString(json, keys: ["language"]) ?? fields.language
            fields.timeSignature = firstString(json, keys: ["timesignature", "time_signature"]) ?? fields.timeSignature
            fields.role = firstString(json, keys: ["role"]) ?? fields.role
            fields.notes = firstString(json, keys: ["notes"]) ?? fields.notes
        }
        if fields.caption != nil {
            fields.captionSource = "sidecar"
        }
        if fields.lyrics != nil {
            fields.lyricsSource = "sidecar"
        }
        if fields.bpm != nil || fields.key != nil || fields.language != nil ||
            fields.timeSignature != nil || fields.role != nil || fields.notes != nil {
            fields.metadataSource = "sidecar"
        }
        return fields
    }

    private static func csvSidecarFields(in datasetDirectory: URL) -> [String: SidecarFields] {
        guard let files = try? FileManager.default.contentsOfDirectory(
            at: datasetDirectory,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        ) else {
            return [:]
        }

        var lookup: [String: SidecarFields] = [:]
        for csvURL in files where csvURL.pathExtension.lowercased() == "csv" {
            for row in readCSVRows(from: csvURL) {
                let path = firstCSV(row, keys: ["file", "path", "audio", "filename", "track", "name"])
                guard let path, !path.isEmpty else { continue }
                let lyrics = cleanLyricText(firstCSV(row, keys: ["lyrics", "transcript"]))
                let fields = SidecarFields(
                    caption: firstCSV(row, keys: ["caption", "prompt"]),
                    lyrics: lyrics,
                    bpm: firstCSV(row, keys: ["bpm", "tempo", "tempo_bpm"]),
                    key: firstCSV(row, keys: ["key", "keyscale"]),
                    language: firstCSV(row, keys: ["language", "lang"]),
                    timeSignature: firstCSV(row, keys: ["timesignature", "time_signature", "meter"]),
                    role: firstCSV(row, keys: ["role"]),
                    notes: firstCSV(row, keys: ["notes"]),
                    captionSource: firstCSV(row, keys: ["caption", "prompt"]) == nil ? nil : "csv",
                    lyricsSource: lyrics == nil ? nil : "csv",
                    metadataSource: "csv"
                )
                for key in lookupKeys(for: path, datasetDirectory: datasetDirectory) {
                    lookup[key] = fields
                }
            }
        }
        return lookup
    }

    private static func lookupKeys(for clip: DatasetClipReview, datasetDirectory: URL) -> [String] {
        var keys = lookupKeys(for: clip.file, datasetDirectory: datasetDirectory)
        if let audioPath = clip.primaryAudioPath {
            keys.append(contentsOf: lookupKeys(for: audioPath, datasetDirectory: datasetDirectory))
        }
        return unique(keys)
    }

    private static func lookupKeys(for path: String, datasetDirectory: URL) -> [String] {
        let cleaned = path.hasPrefix("./") ? String(path.dropFirst(2)) : path
        let resolved = resolve(cleaned, relativeTo: datasetDirectory)
        return unique([
            path,
            cleaned,
            resolved.path(percentEncoded: false),
            URL(fileURLWithPath: cleaned).lastPathComponent,
            URL(fileURLWithPath: cleaned).deletingPathExtension().lastPathComponent,
        ])
    }

    private static func readCSVRows(from url: URL) -> [[String: String]] {
        guard let contents = try? String(contentsOf: url, encoding: .utf8) else { return [] }
        let rows = contents
            .components(separatedBy: .newlines)
            .filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
            .map(splitCSVLine)
        guard let headers = rows.first?.map({ $0.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() }) else {
            return []
        }
        return rows.dropFirst().map { row in
            Dictionary(uniqueKeysWithValues: headers.enumerated().compactMap { index, header in
                guard row.indices.contains(index) else { return nil }
                return (header, row[index].trimmingCharacters(in: .whitespacesAndNewlines))
            })
        }
    }

    private static func splitCSVLine(_ line: String) -> [String] {
        var fields: [String] = []
        var current = ""
        var isQuoted = false
        var iterator = line.makeIterator()

        while let char = iterator.next() {
            if char == "\"" {
                if isQuoted, let next = iterator.next() {
                    if next == "\"" {
                        current.append("\"")
                    } else {
                        isQuoted = false
                        if next != "," {
                            current.append(next)
                        } else {
                            fields.append(current)
                            current = ""
                        }
                    }
                } else {
                    isQuoted.toggle()
                }
            } else if char == "," && !isQuoted {
                fields.append(current)
                current = ""
            } else {
                current.append(char)
            }
        }
        fields.append(current)
        return fields
    }

    private static func firstCSV(_ row: [String: String], keys: [String]) -> String? {
        for key in keys {
            if let value = row[key.lowercased()]?.trimmingCharacters(in: .whitespacesAndNewlines),
               !value.isEmpty {
                return value
            }
        }
        return nil
    }

    private static func firstExisting(_ urls: [URL]) -> URL? {
        urls.first { FileManager.default.fileExists(atPath: $0.path(percentEncoded: false)) }
    }

    private static func readTrimmedText(from url: URL?) -> String? {
        guard let url,
              let text = try? String(contentsOf: url, encoding: .utf8)
                .trimmingCharacters(in: .whitespacesAndNewlines),
              !text.isEmpty else {
            return nil
        }
        return text
    }

    private static func cleanLyricText(_ text: String?) -> String? {
        guard let text else { return nil }
        let cleaned = text
            .components(separatedBy: .newlines)
            .compactMap { line -> String? in
                let trimmedLine = line.trimmingCharacters(in: .whitespacesAndNewlines)
                if trimmedLine.range(
                    of: #"^\[[A-Za-z][A-Za-z0-9_-]*:.*\]$"#,
                    options: .regularExpression
                ) != nil {
                    return nil
                }

                let withoutTimestamps = trimmedLine.replacingOccurrences(
                    of: #"(?:\[[0-9]{1,3}:[0-9]{2}(?:[\.:][0-9]{1,3})?\]|<[0-9]{1,3}:[0-9]{2}(?:[\.:][0-9]{1,3})?>)"#,
                    with: "",
                    options: .regularExpression
                )
                let cleanedLine = withoutTimestamps.trimmingCharacters(in: .whitespacesAndNewlines)
                return cleanedLine.isEmpty ? nil : cleanedLine
            }
            .joined(separator: "\n")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return cleaned.isEmpty ? nil : cleaned
    }

    private static func readJSONObject(from url: URL) -> [String: Any]? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
    }

    private static func readJSONArray(from url: URL) -> [[String: Any]]? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return (try? JSONSerialization.jsonObject(with: data)) as? [[String: Any]]
    }

    private static func writeJSONArray(_ object: [[String: Any]], to url: URL) throws {
        let data = try JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: url, options: .atomic)
    }

    private static func writeJSONObject(_ object: [String: Any], to url: URL) throws {
        let data = try JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: url, options: .atomic)
    }

    private static func isDirectory(_ url: URL) -> Bool {
        (try? url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true
    }

    private static func modificationDate(_ url: URL) -> Date {
        (try? url.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
    }

    private static func stringDictionary(_ value: Any?) -> [String: String]? {
        guard let dict = value as? [String: Any] else { return nil }
        return dict.reduce(into: [:]) { result, pair in
            if let value = pair.value as? String, !value.isEmpty {
                result[pair.key] = value
            }
        }
    }

    private static func firstString(_ dict: [String: Any], keys: [String]) -> String? {
        for key in keys {
            if let value = stringValue(dict[key])?.trimmingCharacters(in: .whitespacesAndNewlines),
               !value.isEmpty {
                return value
            }
        }
        return nil
    }

    private static func stringValue(_ value: Any?) -> String? {
        switch value {
        case let string as String:
            return string
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

    private static func intValue(_ value: Any?) -> Int? {
        switch value {
        case let number as NSNumber:
            return number.intValue
        case let int as Int:
            return int
        case let string as String:
            return Int(string)
        default:
            return nil
        }
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

    private static func inferredRole(from assets: [String: String]) -> String {
        if assets["full_mix"] != nil { return "full_mix" }
        if assets["vocals"] != nil { return "vocal" }
        if assets["instrumental"] != nil { return "instrumental" }
        return "full_mix"
    }

    private static func unique(_ values: [String]) -> [String] {
        var seen = Set<String>()
        return values.filter { !$0.isEmpty && seen.insert($0).inserted }
    }

    private static func csvEscape(_ value: String) -> String {
        if value.contains(",") || value.contains("\"") || value.contains("\n") || value.contains("\r") {
            return "\"\(value.replacingOccurrences(of: "\"", with: "\"\""))\""
        }
        return value
    }
}

private extension SidecarFields {
    func merging(_ other: SidecarFields) -> SidecarFields {
        SidecarFields(
            caption: other.caption ?? caption,
            lyrics: other.lyrics ?? lyrics,
            bpm: other.bpm ?? bpm,
            key: other.key ?? key,
            language: other.language ?? language,
            timeSignature: other.timeSignature ?? timeSignature,
            role: other.role ?? role,
            notes: other.notes ?? notes,
            captionSource: other.captionSource ?? captionSource,
            lyricsSource: other.lyricsSource ?? lyricsSource,
            metadataSource: other.metadataSource ?? metadataSource
        )
    }
}

private extension DatasetClipReview {
    mutating func apply(fields: SidecarFields) -> Int {
        var changed = 0
        let captionChanged = update(&caption, with: fields.caption)
        if captionChanged > 0 {
            captionSource = fields.captionSource ?? "sidecar"
            annotationBackend = backendName(for: fields.captionSource)
        }
        changed += captionChanged

        let lyricsChanged = update(&lyrics, with: fields.lyrics)
        if lyricsChanged > 0 {
            lyricsSource = fields.lyricsSource ?? "sidecar"
            annotationBackend = backendName(for: fields.lyricsSource)
        }
        changed += lyricsChanged

        changed += update(&bpm, with: fields.bpm)
        changed += update(&key, with: fields.key)
        changed += update(&language, with: fields.language)
        changed += update(&timeSignature, with: fields.timeSignature)
        changed += update(&role, with: fields.role)
        changed += update(&notes, with: fields.notes)
        if changed > 0 && !isApproved {
            reviewStatus = "needs_review"
        }
        return changed
    }

    private func backendName(for source: String?) -> String {
        switch source {
        case "csv":
            return "metadata_csv"
        case "sidecar":
            return "sidecar_import"
        default:
            return source ?? "manual"
        }
    }

    private func update(_ field: inout String, with value: String?) -> Int {
        guard let value else { return 0 }
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, trimmed != field else { return 0 }
        field = trimmed
        return 1
    }
}
