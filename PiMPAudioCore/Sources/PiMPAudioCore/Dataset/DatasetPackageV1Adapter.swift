import Foundation

public enum DatasetPackageV1AdapterError: LocalizedError {
    case invalidManifest
    case invalidCaptions
    case emptyCaptions
    case invalidAdaptedPackage([DatasetPackageValidationIssueV2])

    public var errorDescription: String? {
        switch self {
        case .invalidManifest:
            return "The legacy dataset manifest is not a JSON object."
        case .invalidCaptions:
            return "The legacy captions file is not a JSON array of objects."
        case .emptyCaptions:
            return "The legacy captions file does not contain any records."
        case .invalidAdaptedPackage(let issues):
            let summary = issues
                .filter { $0.severity == .error }
                .prefix(3)
                .map { "\($0.path): \($0.message)" }
                .joined(separator: "; ")
            return "The legacy dataset cannot be adapted safely: \(summary)"
        }
    }
}

/// Non-destructive compatibility adapter for existing clip-based packages.
///
/// The adapter never rewrites the source package. Legacy full-mix clip rows
/// become `clip` assets because the complete source audio was not preserved in
/// v1 packages. Existing stem paths are retained as derived assets.
public enum DatasetPackageV1Adapter {
    public static let adapterVersion = "1.0"

    public static func adapt(
        manifestData: Data,
        captionsData: Data,
        adaptedAt: String? = nil
    ) throws -> DatasetPackageV2 {
        guard let manifest = try JSONSerialization.jsonObject(with: manifestData) as? [String: Any] else {
            throw DatasetPackageV1AdapterError.invalidManifest
        }
        guard let captions = try JSONSerialization.jsonObject(with: captionsData) as? [[String: Any]] else {
            throw DatasetPackageV1AdapterError.invalidCaptions
        }
        guard !captions.isEmpty else {
            throw DatasetPackageV1AdapterError.emptyCaptions
        }

        let packageName = string(manifest["name"]) ?? "legacy-dataset"
        let sourceMetadata = sourceMetadataByFileName(manifest)
        let clipFallbackDuration = number(manifest["clip_length_seconds"])

        var trackIDsBySourceKey: [String: String] = [:]
        var tracksByID: [String: DatasetSourceTrackV2] = [:]
        var trackOrder: [String] = []
        var assets: [DatasetAssetV2] = []
        var selections: [DatasetTrainingSelectionV2] = []

        for (recordIndex, record) in captions.enumerated() {
            let source = record["source"] as? [String: Any] ?? [:]
            let primaryPath = primaryAudioPath(record) ?? "clips/legacy-\(recordIndex + 1).wav"
            let sourceFileName = firstString(
                source["file_name"],
                record["source_file_name"],
                URL(fileURLWithPath: primaryPath).lastPathComponent
            ) ?? "legacy-source-\(recordIndex + 1)"
            let sourceKey = sourceFileName.lowercased()
            let trackID: String
            if let existing = trackIDsBySourceKey[sourceKey] {
                trackID = existing
            } else {
                trackID = uniqueID(
                    prefix: "track",
                    preferred: sourceFileName,
                    ordinal: trackOrder.count + 1,
                    existing: Set(trackOrder)
                )
                trackIDsBySourceKey[sourceKey] = trackID
                trackOrder.append(trackID)
                let metadata = sourceMetadata[sourceKey] ?? [:]
                tracksByID[trackID] = DatasetSourceTrackV2(
                    id: trackID,
                    originalFileName: sourceFileName,
                    durationSeconds: number(metadata["duration_seconds"]),
                    sourceFile: nil
                )
            }

            let sourceStart = firstNumber(
                record["source_start_seconds"],
                source["source_start_seconds"],
                source["start_seconds"]
            ) ?? 0
            let audio = record["audio"] as? [String: Any] ?? [:]
            let clipDuration = firstNumber(
                record["clip_length_seconds"],
                audio["duration_seconds"],
                clipFallbackDuration
            )
            let sourceRange = DatasetTimeRangeV2(
                startSeconds: sourceStart,
                endSeconds: clipDuration.map { sourceStart + $0 }
            )
            let annotation = legacyAnnotation(record)
            let baseAssetID = uniqueID(
                prefix: "asset",
                preferred: "\(URL(fileURLWithPath: primaryPath).deletingPathExtension().lastPathComponent)-clip",
                ordinal: assets.count + 1,
                existing: Set(assets.map(\.id))
            )
            let baseAsset = DatasetAssetV2(
                id: baseAssetID,
                sourceTrackID: trackID,
                role: .clip,
                file: audioFile(path: primaryPath, metadata: audio),
                sourceRange: sourceRange,
                derivation: DatasetAssetDerivationV2(
                    kind: .clipExtraction,
                    backendID: "pimp-audio-dataset-v1",
                    details: [
                        "legacy_record_id": string(record["id"]) ?? "\(recordIndex + 1)",
                        "source_asset_status": "not_preserved_in_v1",
                    ]
                ),
                annotation: annotation
            )
            assets.append(baseAsset)
            selections.append(
                DatasetTrainingSelectionV2(
                    assetID: baseAssetID,
                    includeInTraining: true,
                    trainingRange: sourceRange,
                    rangePolicy: .legacyClip
                )
            )

            for role in [DatasetAssetRoleV2.vocals, .instrumental] {
                guard let path = legacyAssetPath(role: role, record: record) else { continue }
                let stemID = uniqueID(
                    prefix: "asset",
                    preferred: "\(URL(fileURLWithPath: primaryPath).deletingPathExtension().lastPathComponent)-\(role.rawValue)",
                    ordinal: assets.count + 1,
                    existing: Set(assets.map(\.id))
                )
                let stemAnnotation: DatasetAssetAnnotationV2?
                if role == .instrumental, var instrumentalAnnotation = annotation {
                    instrumentalAnnotation.lyrics = nil
                    stemAnnotation = instrumentalAnnotation
                } else {
                    stemAnnotation = annotation
                }
                assets.append(
                    DatasetAssetV2(
                        id: stemID,
                        sourceTrackID: trackID,
                        role: role,
                        file: DatasetAudioFileV2(
                            path: path,
                            format: fileExtension(path),
                            durationSeconds: clipDuration
                        ),
                        derivedFrom: baseAssetID,
                        sourceRange: sourceRange,
                        derivation: legacyStemDerivation(record),
                        annotation: stemAnnotation
                    )
                )
            }

            if var track = tracksByID[trackID],
               let end = sourceRange.endSeconds {
                track.durationSeconds = max(track.durationSeconds ?? 0, end)
                tracksByID[trackID] = track
            }
        }

        let sourceVersion = string(manifest["pimp_audio_dataset_version"])
            ?? string(manifest["anvil_dataset_version"])
            ?? "1.0"
        let createdAt = string(manifest["created_at"]) ?? adaptedAt ?? "unknown"
        let package = DatasetPackageV2(
            id: slug(packageName),
            name: packageName,
            createdAt: createdAt,
            sourceTracks: trackOrder.compactMap { tracksByID[$0] },
            assets: assets,
            trainingPlans: [
                DatasetTrainingPlanV2(
                    id: "legacy-v1-default",
                    backendID: "legacy-v1",
                    name: "Legacy v1 clip selection",
                    isDefault: true,
                    selections: selections
                ),
            ],
            migration: DatasetMigrationV2(
                sourceVersion: sourceVersion,
                adapterVersion: adapterVersion,
                migratedAt: adaptedAt,
                notes: [
                    "Read through an in-memory compatibility adapter; the v1 package was not modified.",
                    "Legacy full-mix clip rows remain clip assets because complete source audio was not preserved.",
                    "Legacy vocal and instrumental files remain optional derived assets.",
                ]
            )
        )
        let validation = DatasetPackageV2Validator.validate(package)
        guard validation.isValid else {
            throw DatasetPackageV1AdapterError.invalidAdaptedPackage(validation.issues)
        }
        return package
    }

    private static func sourceMetadataByFileName(_ manifest: [String: Any]) -> [String: [String: Any]] {
        guard let source = manifest["source"] as? [String: Any],
              let files = source["files"] as? [[String: Any]] else {
            return [:]
        }
        var result: [String: [String: Any]] = [:]
        for file in files {
            guard let name = string(file["file_name"]) else { continue }
            result[name.lowercased()] = file
        }
        return result
    }

    private static func primaryAudioPath(_ record: [String: Any]) -> String? {
        let assets = record["assets"] as? [String: Any] ?? [:]
        return firstString(assets["full_mix"], record["file"], record["path"], record["audio"])
    }

    private static func legacyAssetPath(
        role: DatasetAssetRoleV2,
        record: [String: Any]
    ) -> String? {
        let assets = record["assets"] as? [String: Any] ?? [:]
        if let path = string(assets[role.rawValue]) {
            return path
        }
        let separation = record["separation"] as? [String: Any] ?? [:]
        let stems = separation["stems"] as? [String: Any] ?? [:]
        return string(stems[role.rawValue])
    }

    private static func audioFile(
        path: String,
        metadata: [String: Any]
    ) -> DatasetAudioFileV2 {
        DatasetAudioFileV2(
            path: path,
            format: string(metadata["format"]) ?? fileExtension(path),
            byteSize: integer(metadata["byte_size"]),
            durationSeconds: number(metadata["duration_seconds"]),
            sampleRate: integer(metadata["sample_rate"]).map(Int.init),
            channels: integer(metadata["channels"]).map(Int.init)
        )
    }

    private static func legacyStemDerivation(_ record: [String: Any]) -> DatasetAssetDerivationV2 {
        let separation = record["separation"] as? [String: Any] ?? [:]
        return DatasetAssetDerivationV2(
            kind: .stemSeparation,
            backendID: string(separation["backend"]) ?? "legacy-v1-separation",
            backendVersion: string(separation["backend_version"]),
            modelID: string(separation["model"]),
            modelRevision: string(separation["model_revision"])
        )
    }

    private static func legacyAnnotation(_ record: [String: Any]) -> DatasetAssetAnnotationV2? {
        let caption = firstString(record["caption"], record["prompt"]) ?? ""
        let lyrics = firstString(record["lyrics"], record["transcript"]) ?? ""
        let annotation = record["annotation"] as? [String: Any] ?? [:]
        let transcription = record["transcription"] as? [String: Any] ?? [:]
        let review = record["training_review"] as? [String: Any] ?? [:]
        let rawOutputs = (annotation["raw_output_files"] as? [String: Any] ?? [:])
            .compactMapValues(string)
        let captionProvenance = DatasetAnnotationProvenanceV2(
            backendID: firstString(annotation["annotation_backend"], "manual") ?? "manual",
            backendVersion: string(annotation["annotation_backend_version"]),
            modelID: string(annotation["model_id"]),
            modelRevision: string(annotation["model_revision"]),
            generatedAt: string(annotation["updated_at"]),
            rawOutputFiles: rawOutputs.filter { $0.key != "lyrics" }
        )
        let lyricsProvenance = DatasetAnnotationProvenanceV2(
            backendID: firstString(transcription["backend"], annotation["annotation_backend"], "manual") ?? "manual",
            backendVersion: firstString(
                transcription["backend_version"],
                annotation["annotation_backend_version"]
            ),
            modelID: firstString(transcription["model"], annotation["model_id"]),
            modelRevision: firstString(transcription["model_revision"], annotation["model_revision"]),
            generatedAt: firstString(transcription["updated_at"], annotation["updated_at"]),
            rawOutputFiles: rawOutputs.filter { $0.key != "caption" }
        )
        let language = firstString(record["language"], annotation["language"])
        let captionEntry = caption.isEmpty ? nil : DatasetTextAnnotationV2(
            text: caption,
            source: firstString(record["caption_source"], annotation["caption_source"], "legacy-v1") ?? "legacy-v1",
            language: language,
            provenance: captionProvenance
        )
        let lyricsEntry = lyrics.isEmpty ? nil : DatasetTextAnnotationV2(
            text: lyrics,
            source: firstString(record["lyrics_source"], annotation["lyrics_source"], "legacy-v1") ?? "legacy-v1",
            language: language,
            provenance: lyricsProvenance
        )
        guard captionEntry != nil || lyricsEntry != nil || !review.isEmpty || !annotation.isEmpty else {
            return nil
        }
        let approved = boolean(review["approved"]) ?? false
        return DatasetAssetAnnotationV2(
            caption: captionEntry,
            lyrics: lyricsEntry,
            reviewStatus: firstString(
                review["status"],
                annotation["review_status"],
                approved ? "approved" : "needs_review"
            ) ?? "needs_review",
            approved: approved,
            reviewedAt: string(review["updated_at"]),
            notes: string(review["notes"])
        )
    }

    private static func uniqueID(
        prefix: String,
        preferred: String,
        ordinal: Int,
        existing: Set<String>
    ) -> String {
        let base = "\(prefix)-\(String(format: "%04d", ordinal))-\(slug(preferred))"
        guard existing.contains(base) else { return base }
        var suffix = 2
        while existing.contains("\(base)-\(suffix)") {
            suffix += 1
        }
        return "\(base)-\(suffix)"
    }

    private static func slug(_ value: String) -> String {
        let normalized = value.lowercased().unicodeScalars.map { scalar -> Character in
            let value = scalar.value
            let isASCIIAlphaNumeric = (48...57).contains(value) || (97...122).contains(value)
            return isASCIIAlphaNumeric ? Character(scalar) : "-"
        }
        let collapsed = String(normalized).split(separator: "-").joined(separator: "-")
        return collapsed.isEmpty ? "dataset" : collapsed
    }

    private static func fileExtension(_ path: String) -> String? {
        let value = URL(fileURLWithPath: path).pathExtension.lowercased()
        return value.isEmpty ? nil : value
    }

    private static func firstString(_ values: Any?...) -> String? {
        values.lazy.compactMap(string).first
    }

    private static func firstNumber(_ values: Any?...) -> Double? {
        values.lazy.compactMap(number).first
    }

    private static func string(_ value: Any?) -> String? {
        guard let value = value as? String else { return nil }
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }

    private static func number(_ value: Any?) -> Double? {
        switch value {
        case let value as NSNumber:
            return value.doubleValue
        case let value as String:
            return Double(value)
        default:
            return nil
        }
    }

    private static func integer(_ value: Any?) -> Int64? {
        switch value {
        case let value as NSNumber:
            return value.int64Value
        case let value as String:
            return Int64(value)
        default:
            return nil
        }
    }

    private static func boolean(_ value: Any?) -> Bool? {
        switch value {
        case let value as Bool:
            return value
        case let value as NSNumber:
            return value.boolValue
        case let value as String:
            return ["true", "yes", "1"].contains(value.lowercased())
        default:
            return nil
        }
    }
}
