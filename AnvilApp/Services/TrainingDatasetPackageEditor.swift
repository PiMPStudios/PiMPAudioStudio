import Foundation

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
    var hasUnsavedChanges: Bool = false

    var approvedCount: Int {
        clips.filter(\.isApproved).count
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
        }
    }

    var isTrainingReady: Bool {
        blockerCount == 0 && !clips.isEmpty
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
    var rawObject: [String: Any]

    var displayName: String {
        URL(fileURLWithPath: file).lastPathComponent
    }

    var primaryAudioPath: String? {
        assets["full_mix"] ?? (file.isEmpty ? nil : file)
    }

    func audioURL(relativeTo datasetDirectory: URL) -> URL? {
        guard let path = assets["full_mix"] ?? (!file.isEmpty ? file : nil) else { return nil }
        return TrainingDatasetPackageEditor.resolve(path, relativeTo: datasetDirectory)
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
        object["analysis"] = analysis

        let persistedReviewStatus = isApproved ? "approved" : (reviewStatus.isEmpty ? "needs_review" : reviewStatus)
        var rawOutputs = (object["annotation"] as? [String: Any])?["raw_output_files"] as? [String: Any] ?? [:]
        if !rawCaptionOutputFile.isEmpty {
            rawOutputs["caption"] = rawCaptionOutputFile
        }
        if !rawLyricsOutputFile.isEmpty {
            rawOutputs["lyrics"] = rawLyricsOutputFile
        }
        object["annotation"] = [
            "schema_version": TrainingDatasetPackageWriter.annotationSchemaVersion,
            "updated_at": ISO8601DateFormatter().string(from: updatedAt),
            "annotation_backend": annotationBackend.isEmpty ? "manual" : annotationBackend,
            "annotation_backend_version": "",
            "caption_source": trimmedCaption.isEmpty ? "none" : (captionSource.isEmpty ? "manual" : captionSource),
            "lyrics_source": trimmedLyrics.isEmpty ? "none" : (lyricsSource.isEmpty ? "manual" : lyricsSource),
            "model_id": annotationModelID,
            "model_revision": annotationModelRevision,
            "language": trimmedLanguage,
            "caption_present": !trimmedCaption.isEmpty,
            "lyrics_present": !trimmedLyrics.isEmpty,
            "review_status": persistedReviewStatus,
            "raw_output_files": rawOutputs,
        ]

        object["training_review"] = [
            "approved": isApproved,
            "status": persistedReviewStatus,
            "updated_at": ISO8601DateFormatter().string(from: updatedAt),
            "notes": trimmedNotes,
        ]
        return object
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

    var errorDescription: String? {
        switch self {
        case .missingCaptions(let url):
            return "\(url.lastPathComponent) does not contain a captions.json file."
        case .invalidCaptions(let url):
            return "\(url.lastPathComponent)'s captions.json is not a readable clip list."
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

    static func loadPackage(at directory: URL) throws -> DatasetPackageDocument {
        let captionsURL = directory.appendingPathComponent(TrainingDatasetPackageWriter.captionsFileName)
        guard FileManager.default.fileExists(atPath: captionsURL.path(percentEncoded: false)) else {
            throw TrainingDatasetPackageEditorError.missingCaptions(directory)
        }
        guard let records = readJSONArray(from: captionsURL) else {
            throw TrainingDatasetPackageEditorError.invalidCaptions(directory)
        }

        let manifest = readJSONObject(from: directory.appendingPathComponent(TrainingDatasetPackageWriter.datasetManifestFileName))
        let validation = TrainingDatasetPackageWriter.validatePackage(at: directory)
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
        if clip.lyrics.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            issues.append(DatasetPackageIssue(severity: .warning, message: "Lyrics/transcript are empty."))
        }
        if clip.bpm.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            issues.append(DatasetPackageIssue(severity: .warning, message: "BPM is missing."))
        }
        if clip.key.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            issues.append(DatasetPackageIssue(severity: .warning, message: "Key is missing."))
        }
        if clip.separationStatus == "not_run" || clip.separationStatus.isEmpty {
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
            rawObject: object
        )
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
            stemURL.appendingPathExtension("txt"),
            rootStemURL.appendingPathExtension("txt"),
        ])))

        for jsonURL in [
            stemURL.appendingPathExtension("json"),
            rootStemURL.appendingPathExtension("json"),
        ] {
            guard let json = readJSONObject(from: jsonURL) else { continue }
            fields.caption = firstString(json, keys: ["caption", "prompt"]) ?? fields.caption
            fields.lyrics = firstString(json, keys: ["lyrics", "transcript"]) ?? fields.lyrics
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
                let fields = SidecarFields(
                    caption: firstCSV(row, keys: ["caption", "prompt"]),
                    lyrics: firstCSV(row, keys: ["lyrics", "transcript"]),
                    bpm: firstCSV(row, keys: ["bpm", "tempo", "tempo_bpm"]),
                    key: firstCSV(row, keys: ["key", "keyscale"]),
                    language: firstCSV(row, keys: ["language", "lang"]),
                    timeSignature: firstCSV(row, keys: ["timesignature", "time_signature", "meter"]),
                    role: firstCSV(row, keys: ["role"]),
                    notes: firstCSV(row, keys: ["notes"]),
                    captionSource: firstCSV(row, keys: ["caption", "prompt"]) == nil ? nil : "csv",
                    lyricsSource: firstCSV(row, keys: ["lyrics", "transcript"]) == nil ? nil : "csv",
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
            .map {
                $0.replacingOccurrences(
                    of: #"^\s*\[[0-9]{1,2}:[0-9]{2}(?:\.[0-9]+)?\]\s*"#,
                    with: "",
                    options: .regularExpression
                )
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
