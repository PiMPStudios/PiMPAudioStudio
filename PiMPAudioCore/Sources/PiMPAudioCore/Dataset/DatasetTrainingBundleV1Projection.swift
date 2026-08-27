import Foundation

public struct DatasetTrainingBundleV1: Codable, Hashable, Sendable {
    public var pimpAudioTrainingBundleVersion: String
    public var sourcePackageVersion: String
    public var createdAt: String
    public var datasetDirectory: String
    public var include: [String]
    public var trainingLyricsSource: String
    public var annotationSchemaVersion: String
    public var trainingPlanID: String
    public var backendID: String
    public var clipCount: Int
    public var clips: [DatasetTrainingClipV1]

    public init(
        pimpAudioTrainingBundleVersion: String = "1.0",
        sourcePackageVersion: String,
        createdAt: String,
        datasetDirectory: String = ".",
        include: [String],
        trainingLyricsSource: String = "per_asset_annotation",
        annotationSchemaVersion: String = "2.0",
        trainingPlanID: String,
        backendID: String,
        clips: [DatasetTrainingClipV1]
    ) {
        self.pimpAudioTrainingBundleVersion = pimpAudioTrainingBundleVersion
        self.sourcePackageVersion = sourcePackageVersion
        self.createdAt = createdAt
        self.datasetDirectory = datasetDirectory
        self.include = include
        self.trainingLyricsSource = trainingLyricsSource
        self.annotationSchemaVersion = annotationSchemaVersion
        self.trainingPlanID = trainingPlanID
        self.backendID = backendID
        clipCount = clips.count
        self.clips = clips
    }

    enum CodingKeys: String, CodingKey {
        case pimpAudioTrainingBundleVersion = "pimp_audio_training_bundle_version"
        case sourcePackageVersion = "source_package_version"
        case createdAt = "created_at"
        case datasetDirectory = "dataset_dir"
        case include
        case trainingLyricsSource = "training_lyrics_source"
        case annotationSchemaVersion = "annotation_schema_version"
        case trainingPlanID = "training_plan_id"
        case backendID = "backend_id"
        case clipCount = "clip_count"
        case clips
    }
}

public struct DatasetTrainingClipV1: Codable, Hashable, Sendable, Identifiable {
    public var id: String
    public var file: String
    public var assets: [String: String]
    public var caption: String
    public var prompt: String
    public var lyrics: String
    public var transcript: String
    public var annotation: DatasetAssetAnnotationV2?
    public var sourceTrackID: String
    public var assetID: String
    public var assetRole: DatasetAssetRoleV2
    public var sourceRange: DatasetTimeRangeV2
    public var trainingRange: DatasetTimeRangeV2
    public var rangePolicy: DatasetTrainingRangePolicyV2
    public var sampleWindowSeconds: Double?
    public var includeInTraining: Bool

    public init(
        id: String,
        file: String,
        assets: [String: String],
        caption: String,
        prompt: String,
        lyrics: String,
        transcript: String,
        annotation: DatasetAssetAnnotationV2?,
        sourceTrackID: String,
        assetID: String,
        assetRole: DatasetAssetRoleV2,
        sourceRange: DatasetTimeRangeV2,
        trainingRange: DatasetTimeRangeV2,
        rangePolicy: DatasetTrainingRangePolicyV2,
        sampleWindowSeconds: Double?,
        includeInTraining: Bool
    ) {
        self.id = id
        self.file = file
        self.assets = assets
        self.caption = caption
        self.prompt = prompt
        self.lyrics = lyrics
        self.transcript = transcript
        self.annotation = annotation
        self.sourceTrackID = sourceTrackID
        self.assetID = assetID
        self.assetRole = assetRole
        self.sourceRange = sourceRange
        self.trainingRange = trainingRange
        self.rangePolicy = rangePolicy
        self.sampleWindowSeconds = sampleWindowSeconds
        self.includeInTraining = includeInTraining
    }

    enum CodingKeys: String, CodingKey {
        case id
        case file
        case assets
        case caption
        case prompt
        case lyrics
        case transcript
        case annotation
        case sourceTrackID = "source_track_id"
        case assetID = "asset_id"
        case assetRole = "asset_role"
        case sourceRange = "source_range"
        case trainingRange = "training_range"
        case rangePolicy = "range_policy"
        case sampleWindowSeconds = "sample_window_seconds"
        case includeInTraining = "include_in_training"
    }
}

public enum DatasetTrainingBundleV1ProjectionError: LocalizedError {
    case invalidPackage([DatasetPackageValidationIssueV2])
    case missingPlan(String)
    case noIncludedAssets(String)

    public var errorDescription: String? {
        switch self {
        case .invalidPackage:
            "The dataset package is invalid and cannot produce a training bundle."
        case let .missingPlan(id):
            "Dataset training plan \(id) does not exist."
        case let .noIncludedAssets(id):
            "Dataset training plan \(id) does not include any assets."
        }
    }
}

/// Projects package-v2 selections into the existing training-bundle shape.
///
/// This is the compatibility seam used while local training and the remote
/// Side-Step worker migrate to direct package-v2 consumption.
public enum DatasetTrainingBundleV1Projection {
    public static func makeBundle(
        from package: DatasetPackageV2,
        planID: String? = nil
    ) throws -> DatasetTrainingBundleV1 {
        let validation = DatasetPackageV2Validator.validate(package)
        guard validation.isValid else {
            throw DatasetTrainingBundleV1ProjectionError.invalidPackage(validation.issues)
        }

        let plan: DatasetTrainingPlanV2
        if let planID {
            guard let selected = package.trainingPlans.first(where: { $0.id == planID }) else {
                throw DatasetTrainingBundleV1ProjectionError.missingPlan(planID)
            }
            plan = selected
        } else if let selected = package.trainingPlans.first(where: \.isDefault) {
            plan = selected
        } else {
            throw DatasetTrainingBundleV1ProjectionError.missingPlan("default")
        }

        let assetsByID = Dictionary(uniqueKeysWithValues: package.assets.map { ($0.id, $0) })
        let tracksByID = Dictionary(uniqueKeysWithValues: package.sourceTracks.map { ($0.id, $0) })
        let includedSelections = plan.selections.filter(\.includeInTraining)
        guard !includedSelections.isEmpty else {
            throw DatasetTrainingBundleV1ProjectionError.noIncludedAssets(plan.id)
        }

        let clips = includedSelections.compactMap { selection -> DatasetTrainingClipV1? in
            guard let asset = assetsByID[selection.assetID],
                  let track = tracksByID[asset.sourceTrackID] else {
                return nil
            }
            let caption = nonEmpty(asset.annotation?.caption?.text)
                ?? URL(fileURLWithPath: asset.file.path).deletingPathExtension().lastPathComponent
            let lyrics = nonEmpty(asset.annotation?.lyrics?.text)
                ?? (asset.role == .instrumental ? "[Instrumental]" : "")
            return DatasetTrainingClipV1(
                id: asset.id,
                file: asset.file.path,
                assets: [asset.role.rawValue: asset.file.path],
                caption: caption,
                prompt: caption,
                lyrics: lyrics,
                transcript: lyrics,
                annotation: asset.annotation,
                sourceTrackID: track.id,
                assetID: asset.id,
                assetRole: asset.role,
                sourceRange: asset.sourceRange,
                trainingRange: selection.trainingRange,
                rangePolicy: selection.rangePolicy,
                sampleWindowSeconds: selection.sampleWindowSeconds,
                includeInTraining: true
            )
        }

        guard !clips.isEmpty else {
            throw DatasetTrainingBundleV1ProjectionError.noIncludedAssets(plan.id)
        }
        return DatasetTrainingBundleV1(
            sourcePackageVersion: package.pimpAudioDatasetVersion,
            createdAt: package.createdAt,
            include: stableUnique(clips.map(\.assetRole.rawValue)),
            trainingPlanID: plan.id,
            backendID: plan.backendID,
            clips: clips
        )
    }

    public static func encodeBundle(
        from package: DatasetPackageV2,
        planID: String? = nil
    ) throws -> Data {
        let bundle = try makeBundle(from: package, planID: planID)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        return try encoder.encode(bundle)
    }

    private static func stableUnique(_ values: [String]) -> [String] {
        var seen = Set<String>()
        return values.filter { seen.insert($0).inserted }
    }

    private static func nonEmpty(_ value: String?) -> String? {
        guard let value else { return nil }
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }
}
