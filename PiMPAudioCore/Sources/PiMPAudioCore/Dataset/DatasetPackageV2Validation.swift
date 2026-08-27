import Foundation

public enum DatasetPackageValidationSeverityV2: String, Codable, Hashable, Sendable {
    case error
    case warning
}

public struct DatasetPackageValidationIssueV2: Codable, Hashable, Sendable {
    public var severity: DatasetPackageValidationSeverityV2
    public var path: String
    public var message: String

    public init(
        severity: DatasetPackageValidationSeverityV2,
        path: String,
        message: String
    ) {
        self.severity = severity
        self.path = path
        self.message = message
    }
}

public struct DatasetPackageValidationResultV2: Codable, Hashable, Sendable {
    public var issues: [DatasetPackageValidationIssueV2]

    public init(issues: [DatasetPackageValidationIssueV2]) {
        self.issues = issues
    }

    public var errors: [DatasetPackageValidationIssueV2] {
        issues.filter { $0.severity == .error }
    }

    public var warnings: [DatasetPackageValidationIssueV2] {
        issues.filter { $0.severity == .warning }
    }

    public var isValid: Bool {
        errors.isEmpty
    }
}

public enum DatasetPackageV2Validator {
    public static func validate(_ package: DatasetPackageV2) -> DatasetPackageValidationResultV2 {
        var issues: [DatasetPackageValidationIssueV2] = []
        validatePackageMetadata(package, issues: &issues)
        validateTracks(package.sourceTracks, issues: &issues)
        validateAssets(package.assets, tracks: package.sourceTracks, issues: &issues)
        validateTrainingPlans(package.trainingPlans, assets: package.assets, issues: &issues)
        validateTrainingWeight(package, issues: &issues)
        validateMigration(package.migration, issues: &issues)
        return DatasetPackageValidationResultV2(issues: issues)
    }

    private static func validatePackageMetadata(
        _ package: DatasetPackageV2,
        issues: inout [DatasetPackageValidationIssueV2]
    ) {
        if package.pimpAudioDatasetVersion != DatasetPackageV2.currentVersion {
            error(
                "pimp_audio_dataset_version",
                "Expected dataset package version \(DatasetPackageV2.currentVersion).",
                issues: &issues
            )
        }
        requireIdentifier(package.id, at: "id", issues: &issues)
        requireText(package.name, at: "name", issues: &issues)
        requireText(package.createdAt, at: "created_at", issues: &issues)

        if package.sourceTracks.isEmpty {
            error("source_tracks", "A dataset must contain at least one source track.", issues: &issues)
        }
        if package.assets.isEmpty {
            error("assets", "A dataset must contain at least one audio asset.", issues: &issues)
        }
        if package.trainingPlans.isEmpty {
            error("training_plans", "A dataset must contain at least one training plan.", issues: &issues)
        }
    }

    private static func validateTracks(
        _ tracks: [DatasetSourceTrackV2],
        issues: inout [DatasetPackageValidationIssueV2]
    ) {
        validateUniqueIDs(tracks.map(\.id), collectionPath: "source_tracks", issues: &issues)

        for (index, track) in tracks.enumerated() {
            let path = "source_tracks[\(index)]"
            requireIdentifier(track.id, at: "\(path).source_track_id", issues: &issues)
            requireText(track.originalFileName, at: "\(path).original_file_name", issues: &issues)
            validatePositiveOptional(
                track.durationSeconds,
                at: "\(path).duration_seconds",
                issues: &issues
            )
            if let sourceFile = track.sourceFile {
                validateAudioFile(sourceFile, at: "\(path).source_file", issues: &issues)
            }
        }
    }

    private static func validateAssets(
        _ assets: [DatasetAssetV2],
        tracks: [DatasetSourceTrackV2],
        issues: inout [DatasetPackageValidationIssueV2]
    ) {
        validateUniqueIDs(assets.map(\.id), collectionPath: "assets", issues: &issues)
        let trackIDs = Set(tracks.map(\.id))
        let assetsByID = indexedByID(assets, id: \.id)

        for (index, asset) in assets.enumerated() {
            let path = "assets[\(index)]"
            requireIdentifier(asset.id, at: "\(path).asset_id", issues: &issues)
            requireIdentifier(asset.sourceTrackID, at: "\(path).source_track_id", issues: &issues)
            if !trackIDs.contains(asset.sourceTrackID) {
                error(
                    "\(path).source_track_id",
                    "Asset references unknown source track \(asset.sourceTrackID).",
                    issues: &issues
                )
            }

            validateAudioFile(asset.file, at: "\(path).file", issues: &issues)
            validateRange(asset.sourceRange, at: "\(path).source_range", issues: &issues)
            validateAnnotation(asset.annotation, at: "\(path).annotation", issues: &issues)

            if let parentID = asset.derivedFrom {
                if asset.derivation == nil {
                    error(
                        "\(path).derivation",
                        "Derived assets must record derivation provenance.",
                        issues: &issues
                    )
                } else if asset.derivation?.kind == .imported {
                    error(
                        "\(path).derivation.kind",
                        "An imported asset cannot also derive from another asset.",
                        issues: &issues
                    )
                }
                if parentID == asset.id {
                    error(
                        "\(path).derived_from",
                        "An asset cannot derive from itself.",
                        issues: &issues
                    )
                } else if let parent = assetsByID[parentID] {
                    if parent.sourceTrackID != asset.sourceTrackID {
                        error(
                            "\(path).derived_from",
                            "Derived assets must remain attached to the same source track.",
                            issues: &issues
                        )
                    }
                } else {
                    error(
                        "\(path).derived_from",
                        "Asset references unknown parent asset \(parentID).",
                        issues: &issues
                    )
                }
            }
        }

        validateDerivationCycles(assets, issues: &issues)
    }

    private static func validateTrainingPlans(
        _ plans: [DatasetTrainingPlanV2],
        assets: [DatasetAssetV2],
        issues: inout [DatasetPackageValidationIssueV2]
    ) {
        validateUniqueIDs(plans.map(\.id), collectionPath: "training_plans", issues: &issues)
        let assetsByID = indexedByID(assets, id: \.id)
        let defaultCount = plans.filter(\.isDefault).count
        if !plans.isEmpty, defaultCount != 1 {
            error(
                "training_plans",
                "Exactly one training plan must be marked as the default.",
                issues: &issues
            )
        }

        for (planIndex, plan) in plans.enumerated() {
            let path = "training_plans[\(planIndex)]"
            requireIdentifier(plan.id, at: "\(path).id", issues: &issues)
            requireIdentifier(plan.backendID, at: "\(path).backend_id", issues: &issues)
            requireText(plan.name, at: "\(path).name", issues: &issues)
            if plan.selections.isEmpty {
                error("\(path).selections", "A training plan cannot be empty.", issues: &issues)
            }

            var seenAssetIDs = Set<String>()
            var includedCount = 0
            for (selectionIndex, selection) in plan.selections.enumerated() {
                let selectionPath = "\(path).selections[\(selectionIndex)]"
                requireIdentifier(
                    selection.assetID,
                    at: "\(selectionPath).asset_id",
                    issues: &issues
                )
                if !seenAssetIDs.insert(selection.assetID).inserted {
                    error(
                        "\(selectionPath).asset_id",
                        "A training plan can select each asset only once.",
                        issues: &issues
                    )
                }
                guard let asset = assetsByID[selection.assetID] else {
                    error(
                        "\(selectionPath).asset_id",
                        "Training selection references unknown asset \(selection.assetID).",
                        issues: &issues
                    )
                    continue
                }

                if selection.includeInTraining {
                    includedCount += 1
                }
                validateRange(
                    selection.trainingRange,
                    at: "\(selectionPath).training_range",
                    issues: &issues
                )
                validateTrainingRange(
                    selection.trainingRange,
                    isWithin: asset.sourceRange,
                    at: "\(selectionPath).training_range",
                    issues: &issues
                )
                validateSampleWindow(
                    selection.sampleWindowSeconds,
                    trainingRange: selection.trainingRange,
                    at: "\(selectionPath).sample_window_seconds",
                    issues: &issues
                )
                if selection.rangePolicy == .fullSource,
                   selection.trainingRange != asset.sourceRange {
                    warning(
                        "\(selectionPath).range_policy",
                        "full_source policy does not match the asset's complete source range.",
                        issues: &issues
                    )
                }
            }

            if includedCount == 0 {
                error(
                    "\(path).selections",
                    "A training plan must include at least one asset.",
                    issues: &issues
                )
            }
        }
    }

    private static func validateTrainingWeight(
        _ package: DatasetPackageV2,
        issues: inout [DatasetPackageValidationIssueV2]
    ) {
        let assetsByID = indexedByID(package.assets, id: \.id)
        for (planIndex, plan) in package.trainingPlans.enumerated() {
            let includedAssets = plan.selections
                .filter(\.includeInTraining)
                .compactMap { assetsByID[$0.assetID] }
            let grouped = Dictionary(grouping: includedAssets, by: \.sourceTrackID)
            for (sourceTrackID, assets) in grouped where assets.count > 1 {
                let roles = assets.map(\.role.rawValue).sorted().joined(separator: ", ")
                warning(
                    "training_plans[\(planIndex)].selections",
                    "Source track \(sourceTrackID) contributes \(assets.count) assets (\(roles)); this increases its effective training weight.",
                    issues: &issues
                )
            }
        }
    }

    private static func validateAudioFile(
        _ file: DatasetAudioFileV2,
        at path: String,
        issues: inout [DatasetPackageValidationIssueV2]
    ) {
        if !isSafePackageRelativePath(file.path) {
            error(
                "\(path).path",
                "Audio paths must be safe, package-relative paths without traversal.",
                issues: &issues
            )
        }
        validatePositiveOptional(file.durationSeconds, at: "\(path).duration_seconds", issues: &issues)
        validatePositiveOptional(file.byteSize.map(Double.init), at: "\(path).byte_size", issues: &issues)
        validatePositiveOptional(file.sampleRate.map(Double.init), at: "\(path).sample_rate", issues: &issues)
        validatePositiveOptional(file.channels.map(Double.init), at: "\(path).channels", issues: &issues)
        if let format = file.format {
            requireText(format, at: "\(path).format", issues: &issues)
        }
        if let sha256 = file.sha256, !isSHA256(sha256) {
            error("\(path).sha256", "SHA-256 values must contain 64 hexadecimal characters.", issues: &issues)
        }
    }

    private static func validateAnnotation(
        _ annotation: DatasetAssetAnnotationV2?,
        at path: String,
        issues: inout [DatasetPackageValidationIssueV2]
    ) {
        guard let annotation else { return }
        requireText(annotation.reviewStatus, at: "\(path).review_status", issues: &issues)
        for (kind, text) in [("caption", annotation.caption), ("lyrics", annotation.lyrics)] {
            guard let text else { continue }
            requireText(text.text, at: "\(path).\(kind).text", issues: &issues)
            requireText(text.source, at: "\(path).\(kind).source", issues: &issues)
            guard let provenance = text.provenance else { continue }
            requireIdentifier(
                provenance.backendID,
                at: "\(path).\(kind).provenance.backend_id",
                issues: &issues
            )
            for (key, rawPath) in provenance.rawOutputFiles {
                if key.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    error(
                        "\(path).\(kind).provenance.raw_output_files",
                        "Raw-output keys cannot be empty.",
                        issues: &issues
                    )
                }
                if !isSafePackageRelativePath(rawPath) {
                    error(
                        "\(path).\(kind).provenance.raw_output_files.\(key)",
                        "Raw-output paths must remain package-relative.",
                        issues: &issues
                    )
                }
            }
        }
    }

    private static func validateMigration(
        _ migration: DatasetMigrationV2?,
        issues: inout [DatasetPackageValidationIssueV2]
    ) {
        guard let migration else { return }
        requireText(migration.sourceVersion, at: "migration.source_version", issues: &issues)
        requireText(migration.adapterVersion, at: "migration.adapter_version", issues: &issues)
    }

    private static func validateRange(
        _ range: DatasetTimeRangeV2,
        at path: String,
        issues: inout [DatasetPackageValidationIssueV2]
    ) {
        if !range.startSeconds.isFinite || range.startSeconds < 0 {
            error("\(path).start_seconds", "Range start must be finite and non-negative.", issues: &issues)
        }
        guard let end = range.endSeconds else {
            warning(
                "\(path).end_seconds",
                "Range end is unknown; truncation and coverage cannot be fully reviewed.",
                issues: &issues
            )
            return
        }
        if !end.isFinite || end <= range.startSeconds {
            error(
                "\(path).end_seconds",
                "Range end must be finite and greater than its start.",
                issues: &issues
            )
        }
    }

    private static func validateTrainingRange(
        _ trainingRange: DatasetTimeRangeV2,
        isWithin sourceRange: DatasetTimeRangeV2,
        at path: String,
        issues: inout [DatasetPackageValidationIssueV2]
    ) {
        if trainingRange.startSeconds < sourceRange.startSeconds {
            error(path, "Training range starts before the asset's source range.", issues: &issues)
        }
        if let trainingEnd = trainingRange.endSeconds,
           let sourceEnd = sourceRange.endSeconds,
           trainingEnd > sourceEnd {
            error(path, "Training range ends after the asset's source range.", issues: &issues)
        }
    }

    private static func validateSampleWindow(
        _ sampleWindow: Double?,
        trainingRange: DatasetTimeRangeV2,
        at path: String,
        issues: inout [DatasetPackageValidationIssueV2]
    ) {
        guard let sampleWindow else { return }
        if !sampleWindow.isFinite || sampleWindow <= 0 {
            error(path, "Sample-window duration must be finite and positive.", issues: &issues)
            return
        }
        if let duration = trainingRange.durationSeconds, sampleWindow > duration {
            error(path, "Sample-window duration cannot exceed the training range.", issues: &issues)
        }
    }

    private static func validateDerivationCycles(
        _ assets: [DatasetAssetV2],
        issues: inout [DatasetPackageValidationIssueV2]
    ) {
        var parentByID: [String: String] = [:]
        for asset in assets where parentByID[asset.id] == nil {
            parentByID[asset.id] = asset.derivedFrom
        }
        for asset in assets {
            var current: String? = asset.id
            var visited = Set<String>()
            while let id = current, let parent = parentByID[id] {
                if !visited.insert(id).inserted || parent == asset.id {
                    error(
                        "assets.\(asset.id).derived_from",
                        "Asset derivation relationships cannot contain cycles.",
                        issues: &issues
                    )
                    break
                }
                current = parent
            }
        }
    }

    private static func validateUniqueIDs(
        _ ids: [String],
        collectionPath: String,
        issues: inout [DatasetPackageValidationIssueV2]
    ) {
        var seen = Set<String>()
        for (index, id) in ids.enumerated() where !seen.insert(id).inserted {
            error(
                "\(collectionPath)[\(index)]",
                "Identifier \(id) is duplicated.",
                issues: &issues
            )
        }
    }

    private static func indexedByID<Element>(
        _ elements: [Element],
        id: KeyPath<Element, String>
    ) -> [String: Element] {
        var result: [String: Element] = [:]
        for element in elements where result[element[keyPath: id]] == nil {
            result[element[keyPath: id]] = element
        }
        return result
    }

    private static func validatePositiveOptional(
        _ value: Double?,
        at path: String,
        issues: inout [DatasetPackageValidationIssueV2]
    ) {
        guard let value else { return }
        if !value.isFinite || value <= 0 {
            error(path, "Value must be finite and positive.", issues: &issues)
        }
    }

    private static func requireIdentifier(
        _ value: String,
        at path: String,
        issues: inout [DatasetPackageValidationIssueV2]
    ) {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty {
            error(path, "Identifier cannot be empty.", issues: &issues)
            return
        }
        if trimmed.unicodeScalars.contains(where: { !isIdentifierScalar($0) }) {
            error(
                path,
                "Identifiers may contain only letters, numbers, period, underscore, and hyphen.",
                issues: &issues
            )
        }
    }

    private static func requireText(
        _ value: String,
        at path: String,
        issues: inout [DatasetPackageValidationIssueV2]
    ) {
        if value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            error(path, "Value cannot be empty.", issues: &issues)
        }
    }

    private static func isSafePackageRelativePath(_ path: String) -> Bool {
        let trimmed = path.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty,
              trimmed == path,
              !trimmed.hasPrefix("/"),
              !trimmed.hasPrefix("~"),
              !trimmed.contains("\\"),
              !trimmed.contains("\0"),
              URL(string: trimmed)?.scheme == nil else {
            return false
        }
        let components = trimmed.split(
            separator: "/",
            omittingEmptySubsequences: false
        )
        return !components.contains(where: { $0.isEmpty || $0 == "." || $0 == ".." })
    }

    private static func isSHA256(_ value: String) -> Bool {
        guard value.count == 64 else { return false }
        return value.unicodeScalars.allSatisfy {
            CharacterSet(charactersIn: "0123456789abcdefABCDEF").contains($0)
        }
    }

    private static func isIdentifierScalar(_ scalar: UnicodeScalar) -> Bool {
        let value = scalar.value
        return (48 ... 57).contains(value)
            || (65 ... 90).contains(value)
            || (97 ... 122).contains(value)
            || scalar == "."
            || scalar == "_"
            || scalar == "-"
    }

    private static func error(
        _ path: String,
        _ message: String,
        issues: inout [DatasetPackageValidationIssueV2]
    ) {
        issues.append(.init(severity: .error, path: path, message: message))
    }

    private static func warning(
        _ path: String,
        _ message: String,
        issues: inout [DatasetPackageValidationIssueV2]
    ) {
        issues.append(.init(severity: .warning, path: path, message: message))
    }
}

public enum DatasetPackageV2CodecError: LocalizedError {
    case invalidPackage([DatasetPackageValidationIssueV2])

    public var errorDescription: String? {
        switch self {
        case let .invalidPackage(issues):
            let summary = issues
                .filter { $0.severity == .error }
                .prefix(3)
                .map { "\($0.path): \($0.message)" }
                .joined(separator: "; ")
            return "Dataset package v2 validation failed: \(summary)"
        }
    }
}

public enum DatasetPackageV2Codec {
    public static func encodeValidated(_ package: DatasetPackageV2) throws -> Data {
        let validation = DatasetPackageV2Validator.validate(package)
        guard validation.isValid else {
            throw DatasetPackageV2CodecError.invalidPackage(validation.issues)
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        return try encoder.encode(package)
    }

    public static func decodeValidated(_ data: Data) throws -> DatasetPackageV2 {
        let package = try JSONDecoder().decode(DatasetPackageV2.self, from: data)
        let validation = DatasetPackageV2Validator.validate(package)
        guard validation.isValid else {
            throw DatasetPackageV2CodecError.invalidPackage(validation.issues)
        }
        return package
    }
}
