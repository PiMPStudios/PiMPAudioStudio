import Foundation

/// Canonical PiMP Audio Studio dataset contract.
///
/// Version 2 separates source tracks from their audio assets so that full
/// mixes, stems, and clips can coexist without losing their relationships or
/// silently multiplying a source track's training weight.
public struct DatasetPackageV2: Codable, Hashable, Sendable {
    public static let currentVersion = "2.0"

    public var pimpAudioDatasetVersion: String
    public var id: String
    public var name: String
    public var createdAt: String
    public var sourceTracks: [DatasetSourceTrackV2]
    public var assets: [DatasetAssetV2]
    public var trainingPlans: [DatasetTrainingPlanV2]
    public var migration: DatasetMigrationV2?

    public init(
        pimpAudioDatasetVersion: String = Self.currentVersion,
        id: String,
        name: String,
        createdAt: String,
        sourceTracks: [DatasetSourceTrackV2],
        assets: [DatasetAssetV2],
        trainingPlans: [DatasetTrainingPlanV2],
        migration: DatasetMigrationV2? = nil
    ) {
        self.pimpAudioDatasetVersion = pimpAudioDatasetVersion
        self.id = id
        self.name = name
        self.createdAt = createdAt
        self.sourceTracks = sourceTracks
        self.assets = assets
        self.trainingPlans = trainingPlans
        self.migration = migration
    }

    enum CodingKeys: String, CodingKey {
        case pimpAudioDatasetVersion = "pimp_audio_dataset_version"
        case id
        case name
        case createdAt = "created_at"
        case sourceTracks = "source_tracks"
        case assets
        case trainingPlans = "training_plans"
        case migration
    }
}

public struct DatasetSourceTrackV2: Codable, Hashable, Sendable, Identifiable {
    public var id: String
    public var originalFileName: String
    public var durationSeconds: Double?
    public var sourceFile: DatasetAudioFileV2?

    public init(
        id: String,
        originalFileName: String,
        durationSeconds: Double? = nil,
        sourceFile: DatasetAudioFileV2? = nil
    ) {
        self.id = id
        self.originalFileName = originalFileName
        self.durationSeconds = durationSeconds
        self.sourceFile = sourceFile
    }

    enum CodingKeys: String, CodingKey {
        case id = "source_track_id"
        case originalFileName = "original_file_name"
        case durationSeconds = "duration_seconds"
        case sourceFile = "source_file"
    }
}

public struct DatasetAudioFileV2: Codable, Hashable, Sendable {
    public var path: String
    public var format: String?
    public var byteSize: Int64?
    public var durationSeconds: Double?
    public var sampleRate: Int?
    public var channels: Int?
    public var sha256: String?

    public init(
        path: String,
        format: String? = nil,
        byteSize: Int64? = nil,
        durationSeconds: Double? = nil,
        sampleRate: Int? = nil,
        channels: Int? = nil,
        sha256: String? = nil
    ) {
        self.path = path
        self.format = format
        self.byteSize = byteSize
        self.durationSeconds = durationSeconds
        self.sampleRate = sampleRate
        self.channels = channels
        self.sha256 = sha256
    }

    enum CodingKeys: String, CodingKey {
        case path
        case format
        case byteSize = "byte_size"
        case durationSeconds = "duration_seconds"
        case sampleRate = "sample_rate"
        case channels
        case sha256
    }
}

public enum DatasetAssetRoleV2: String, Codable, CaseIterable, Hashable, Sendable {
    case fullMix = "full_mix"
    case vocals
    case instrumental
    case clip
}

public struct DatasetAssetV2: Codable, Hashable, Sendable, Identifiable {
    public var id: String
    public var sourceTrackID: String
    public var role: DatasetAssetRoleV2
    public var file: DatasetAudioFileV2
    public var derivedFrom: String?
    public var sourceRange: DatasetTimeRangeV2
    public var derivation: DatasetAssetDerivationV2?
    public var annotation: DatasetAssetAnnotationV2?

    public init(
        id: String,
        sourceTrackID: String,
        role: DatasetAssetRoleV2,
        file: DatasetAudioFileV2,
        derivedFrom: String? = nil,
        sourceRange: DatasetTimeRangeV2,
        derivation: DatasetAssetDerivationV2? = nil,
        annotation: DatasetAssetAnnotationV2? = nil
    ) {
        self.id = id
        self.sourceTrackID = sourceTrackID
        self.role = role
        self.file = file
        self.derivedFrom = derivedFrom
        self.sourceRange = sourceRange
        self.derivation = derivation
        self.annotation = annotation
    }

    enum CodingKeys: String, CodingKey {
        case id = "asset_id"
        case sourceTrackID = "source_track_id"
        case role = "asset_role"
        case file
        case derivedFrom = "derived_from"
        case sourceRange = "source_range"
        case derivation
        case annotation
    }
}

public struct DatasetTimeRangeV2: Codable, Hashable, Sendable {
    public var startSeconds: Double
    public var endSeconds: Double?

    public init(startSeconds: Double = 0, endSeconds: Double?) {
        self.startSeconds = startSeconds
        self.endSeconds = endSeconds
    }

    public var durationSeconds: Double? {
        endSeconds.map { $0 - startSeconds }
    }

    enum CodingKeys: String, CodingKey {
        case startSeconds = "start_seconds"
        case endSeconds = "end_seconds"
    }
}

public enum DatasetAssetDerivationKindV2: String, Codable, Hashable, Sendable {
    case imported
    case clipExtraction = "clip_extraction"
    case stemSeparation = "stem_separation"
}

public struct DatasetAssetDerivationV2: Codable, Hashable, Sendable {
    public var kind: DatasetAssetDerivationKindV2
    public var backendID: String?
    public var backendVersion: String?
    public var modelID: String?
    public var modelRevision: String?
    public var createdAt: String?
    public var details: [String: String]

    public init(
        kind: DatasetAssetDerivationKindV2,
        backendID: String? = nil,
        backendVersion: String? = nil,
        modelID: String? = nil,
        modelRevision: String? = nil,
        createdAt: String? = nil,
        details: [String: String] = [:]
    ) {
        self.kind = kind
        self.backendID = backendID
        self.backendVersion = backendVersion
        self.modelID = modelID
        self.modelRevision = modelRevision
        self.createdAt = createdAt
        self.details = details
    }

    enum CodingKeys: String, CodingKey {
        case kind
        case backendID = "backend_id"
        case backendVersion = "backend_version"
        case modelID = "model_id"
        case modelRevision = "model_revision"
        case createdAt = "created_at"
        case details
    }
}

public struct DatasetAssetAnnotationV2: Codable, Hashable, Sendable {
    public var caption: DatasetTextAnnotationV2?
    public var lyrics: DatasetTextAnnotationV2?
    public var reviewStatus: String
    public var approved: Bool
    public var reviewedAt: String?
    public var notes: String?

    public init(
        caption: DatasetTextAnnotationV2? = nil,
        lyrics: DatasetTextAnnotationV2? = nil,
        reviewStatus: String = "needs_review",
        approved: Bool = false,
        reviewedAt: String? = nil,
        notes: String? = nil
    ) {
        self.caption = caption
        self.lyrics = lyrics
        self.reviewStatus = reviewStatus
        self.approved = approved
        self.reviewedAt = reviewedAt
        self.notes = notes
    }

    enum CodingKeys: String, CodingKey {
        case caption
        case lyrics
        case reviewStatus = "review_status"
        case approved
        case reviewedAt = "reviewed_at"
        case notes
    }
}

public struct DatasetTextAnnotationV2: Codable, Hashable, Sendable {
    public var text: String
    public var source: String
    public var language: String?
    public var provenance: DatasetAnnotationProvenanceV2?

    public init(
        text: String,
        source: String,
        language: String? = nil,
        provenance: DatasetAnnotationProvenanceV2? = nil
    ) {
        self.text = text
        self.source = source
        self.language = language
        self.provenance = provenance
    }
}

public struct DatasetAnnotationProvenanceV2: Codable, Hashable, Sendable {
    public var backendID: String
    public var backendVersion: String?
    public var modelID: String?
    public var modelRevision: String?
    public var promptVersion: String?
    public var quantization: String?
    public var generatedAt: String?
    public var rawOutputFiles: [String: String]

    public init(
        backendID: String,
        backendVersion: String? = nil,
        modelID: String? = nil,
        modelRevision: String? = nil,
        promptVersion: String? = nil,
        quantization: String? = nil,
        generatedAt: String? = nil,
        rawOutputFiles: [String: String] = [:]
    ) {
        self.backendID = backendID
        self.backendVersion = backendVersion
        self.modelID = modelID
        self.modelRevision = modelRevision
        self.promptVersion = promptVersion
        self.quantization = quantization
        self.generatedAt = generatedAt
        self.rawOutputFiles = rawOutputFiles
    }

    enum CodingKeys: String, CodingKey {
        case backendID = "backend_id"
        case backendVersion = "backend_version"
        case modelID = "model_id"
        case modelRevision = "model_revision"
        case promptVersion = "prompt_version"
        case quantization
        case generatedAt = "generated_at"
        case rawOutputFiles = "raw_output_files"
    }
}

public enum DatasetTrainingRangePolicyV2: String, Codable, Hashable, Sendable {
    case fullSource = "full_source"
    case backendLimit = "backend_limit"
    case manual
    case legacyClip = "legacy_clip"
}

public struct DatasetTrainingSelectionV2: Codable, Hashable, Sendable {
    public var assetID: String
    public var includeInTraining: Bool
    public var trainingRange: DatasetTimeRangeV2
    public var rangePolicy: DatasetTrainingRangePolicyV2
    public var sampleWindowSeconds: Double?

    public init(
        assetID: String,
        includeInTraining: Bool,
        trainingRange: DatasetTimeRangeV2,
        rangePolicy: DatasetTrainingRangePolicyV2,
        sampleWindowSeconds: Double? = nil
    ) {
        self.assetID = assetID
        self.includeInTraining = includeInTraining
        self.trainingRange = trainingRange
        self.rangePolicy = rangePolicy
        self.sampleWindowSeconds = sampleWindowSeconds
    }

    enum CodingKeys: String, CodingKey {
        case assetID = "asset_id"
        case includeInTraining = "include_in_training"
        case trainingRange = "training_range"
        case rangePolicy = "range_policy"
        case sampleWindowSeconds = "sample_window_seconds"
    }
}

public struct DatasetTrainingPlanV2: Codable, Hashable, Sendable, Identifiable {
    public var id: String
    public var backendID: String
    public var name: String
    public var isDefault: Bool
    public var selections: [DatasetTrainingSelectionV2]

    public init(
        id: String,
        backendID: String,
        name: String,
        isDefault: Bool,
        selections: [DatasetTrainingSelectionV2]
    ) {
        self.id = id
        self.backendID = backendID
        self.name = name
        self.isDefault = isDefault
        self.selections = selections
    }

    enum CodingKeys: String, CodingKey {
        case id
        case backendID = "backend_id"
        case name
        case isDefault = "is_default"
        case selections
    }
}

public struct DatasetMigrationV2: Codable, Hashable, Sendable {
    public var sourceVersion: String
    public var adapterVersion: String
    public var migratedAt: String?
    public var notes: [String]

    public init(
        sourceVersion: String,
        adapterVersion: String,
        migratedAt: String? = nil,
        notes: [String] = []
    ) {
        self.sourceVersion = sourceVersion
        self.adapterVersion = adapterVersion
        self.migratedAt = migratedAt
        self.notes = notes
    }

    enum CodingKeys: String, CodingKey {
        case sourceVersion = "source_version"
        case adapterVersion = "adapter_version"
        case migratedAt = "migrated_at"
        case notes
    }
}
