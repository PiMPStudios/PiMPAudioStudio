import Foundation

enum RemoteTrainingTask: String, Codable, CaseIterable, Hashable, Sendable {
    case preprocess
    case annotate
    case trainLoRA = "train_lora"
    case trainLoKR = "train_lokr"
}

enum RemoteTrainingAdapterMode: String, Codable, Hashable, Sendable {
    case lora
    case lokr
}

enum RemoteTrainingJobStatus: String, Codable, Hashable, Sendable {
    case queued
    case running
    case succeeded
    case failed
    case cancelled
}

struct RemoteTrainingPackageDescriptor: Codable, Hashable, Sendable {
    var anvilRemotePackageVersion: String
    var createdAt: String?
    var packageName: String
    var packageRoot: String
    var clipCount: Int
    var defaultBundle: String
    var bundles: [RemoteTrainingBundleReference]
    var providerTargets: [String]
    var preprocessCache: String
    var copyRules: RemoteTrainingCopyRules
    var workerContract: RemoteWorkerContract?

    enum CodingKeys: String, CodingKey {
        case anvilRemotePackageVersion = "anvil_remote_package_version"
        case createdAt = "created_at"
        case packageName = "package_name"
        case packageRoot = "package_root"
        case clipCount = "clip_count"
        case defaultBundle = "default_bundle"
        case bundles
        case providerTargets = "provider_targets"
        case preprocessCache = "preprocess_cache"
        case copyRules = "copy_rules"
        case workerContract = "worker_contract"
    }
}

struct RemoteTrainingBundleReference: Codable, Hashable, Sendable {
    var name: String
    var path: String
    var include: [String]
    var trainingLyrics: String?
    var trainingLyricsSource: String?
    var optional: Bool?

    enum CodingKeys: String, CodingKey {
        case name
        case path
        case include
        case trainingLyrics = "training_lyrics"
        case trainingLyricsSource = "training_lyrics_source"
        case optional
    }
}

struct RemoteTrainingCopyRules: Codable, Hashable, Sendable {
    var include: [String]
    var exclude: [String]
}

struct RemoteWorkerContract: Codable, Hashable, Sendable {
    var resolveRelativePathsFrom: String
    var entryBundle: String
    var preprocessCache: String
    var secretsRequired: [String]
    var expectedGPUClass: String?
    var jobDescriptor: String?
    var resultDescriptor: String?
    var logsFile: String?
    var metricsFile: String?

    enum CodingKeys: String, CodingKey {
        case resolveRelativePathsFrom = "resolve_relative_paths_from"
        case entryBundle = "entry_bundle"
        case preprocessCache = "preprocess_cache"
        case secretsRequired = "secrets_required"
        case expectedGPUClass = "expected_gpu_class"
        case jobDescriptor = "job_descriptor"
        case resultDescriptor = "result_descriptor"
        case logsFile = "logs_file"
        case metricsFile = "metrics_file"
    }
}

struct RemoteTrainingJobRequest: Codable, Hashable, Sendable {
    static let schemaVersion = "1.0"

    var anvilRemoteJobVersion: String
    var jobID: String
    var createdAt: String
    var task: RemoteTrainingTask
    var package: RemoteTrainingJobPackage
    var training: RemoteTrainingJobTraining?
    var runtime: RemoteTrainingJobRuntime
    var outputs: RemoteTrainingJobOutputs
    var safety: RemoteTrainingJobSafety

    enum CodingKeys: String, CodingKey {
        case anvilRemoteJobVersion = "anvil_remote_job_version"
        case jobID = "job_id"
        case createdAt = "created_at"
        case task
        case package
        case training
        case runtime
        case outputs
        case safety
    }
}

struct RemoteTrainingJobPackage: Codable, Hashable, Sendable {
    var packageName: String
    var packageRoot: String
    var descriptor: String
    var entryBundle: String
    var selectedBundle: RemoteTrainingBundleReference
    var preprocessCache: String
    var copyRules: RemoteTrainingCopyRules

    enum CodingKeys: String, CodingKey {
        case packageName = "package_name"
        case packageRoot = "package_root"
        case descriptor
        case entryBundle = "entry_bundle"
        case selectedBundle = "selected_bundle"
        case preprocessCache = "preprocess_cache"
        case copyRules = "copy_rules"
    }
}

struct RemoteTrainingJobTraining: Codable, Hashable, Sendable {
    var adapterMode: RemoteTrainingAdapterMode
    var adapterName: String
    var baseModelID: String
    var baseModelLabel: String
    var rank: Int?
    var epochs: Int?
    var batchSize: Int?
    var gradientAccumulationSteps: Int?
    var learningRate: Double?
    var targetLayers: [String]
    var trainingMode: String

    enum CodingKeys: String, CodingKey {
        case adapterMode = "adapter_mode"
        case adapterName = "adapter_name"
        case baseModelID = "base_model_id"
        case baseModelLabel = "base_model_label"
        case rank
        case epochs
        case batchSize = "batch_size"
        case gradientAccumulationSteps = "gradient_accumulation_steps"
        case learningRate = "learning_rate"
        case targetLayers = "target_layers"
        case trainingMode = "training_mode"
    }
}

struct RemoteTrainingJobRuntime: Codable, Hashable, Sendable {
    var gpuClass: String
    var minVRAMGB: Int?
    var precision: String
    var maxRuntimeSeconds: Int?
    var maxCostUSD: Double?
    var containerImage: String?
    var modelCacheVolume: String?

    enum CodingKeys: String, CodingKey {
        case gpuClass = "gpu_class"
        case minVRAMGB = "min_vram_gb"
        case precision
        case maxRuntimeSeconds = "max_runtime_seconds"
        case maxCostUSD = "max_cost_usd"
        case containerImage = "container_image"
        case modelCacheVolume = "model_cache_volume"
    }
}

struct RemoteTrainingJobOutputs: Codable, Hashable, Sendable {
    var logsFile: String
    var metricsFile: String
    var resultFile: String
    var checkpointsDirectory: String
    var finalAdapterDirectory: String
    var updatedPackageDirectory: String

    enum CodingKeys: String, CodingKey {
        case logsFile = "logs_file"
        case metricsFile = "metrics_file"
        case resultFile = "result_file"
        case checkpointsDirectory = "checkpoints_directory"
        case finalAdapterDirectory = "final_adapter_directory"
        case updatedPackageDirectory = "updated_package_directory"
    }
}

struct RemoteTrainingJobSafety: Codable, Hashable, Sendable {
    var uploadRequiresUserApproval: Bool
    var redactLocalPaths: Bool
    var writeProviderSecretsToPackage: Bool
    var secretsRequired: [String]

    enum CodingKeys: String, CodingKey {
        case uploadRequiresUserApproval = "upload_requires_user_approval"
        case redactLocalPaths = "redact_local_paths"
        case writeProviderSecretsToPackage = "write_provider_secrets_to_package"
        case secretsRequired = "secrets_required"
    }
}

struct RemoteTrainingJobResult: Codable, Hashable, Sendable {
    static let schemaVersion = "1.0"

    var anvilRemoteResultVersion: String
    var jobID: String
    var task: RemoteTrainingTask
    var status: RemoteTrainingJobStatus
    var startedAt: String?
    var finishedAt: String?
    var artifacts: [RemoteTrainingJobArtifact]
    var metrics: [String: Double]
    var error: RemoteTrainingJobError?

    enum CodingKeys: String, CodingKey {
        case anvilRemoteResultVersion = "anvil_remote_result_version"
        case jobID = "job_id"
        case task
        case status
        case startedAt = "started_at"
        case finishedAt = "finished_at"
        case artifacts
        case metrics
        case error
    }
}

struct RemoteTrainingJobArtifact: Codable, Hashable, Sendable {
    var kind: String
    var path: String
    var mediaType: String?
    var byteSize: Int64?
    var sha256: String?

    enum CodingKeys: String, CodingKey {
        case kind
        case path
        case mediaType = "media_type"
        case byteSize = "byte_size"
        case sha256
    }
}

struct RemoteTrainingJobError: Codable, Hashable, Sendable {
    var code: String
    var message: String
    var recoverable: Bool
}

enum RemoteTrainingJobCodec {
    static let jobFileName = "remote_job.json"
    static let resultFileName = "remote_artifacts/result.json"
    static let logsFileName = "remote_artifacts/logs.jsonl"
    static let metricsFileName = "remote_artifacts/metrics.jsonl"

    static var encoder: JSONEncoder {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return encoder
    }

    static var decoder: JSONDecoder {
        JSONDecoder()
    }
}

extension RemoteTrainingJobRequest {
    static func makeTemplate(
        from descriptor: RemoteTrainingPackageDescriptor,
        task: RemoteTrainingTask = .trainLoRA,
        selectedBundleName: String? = nil,
        jobID: String = UUID().uuidString,
        createdAt: Date = Date(),
        adapterName: String? = nil,
        baseModelID: String = "ace-step-xl-sft",
        baseModelLabel: String = "ACE-Step XL SFT",
        maxRuntimeSeconds: Int? = nil,
        maxCostUSD: Double? = nil
    ) -> RemoteTrainingJobRequest {
        let selectedBundle = descriptor.bundle(named: selectedBundleName) ?? descriptor.defaultBundleReference
        let training = trainingConfig(
            task: task,
            packageName: descriptor.packageName,
            adapterName: adapterName,
            baseModelID: baseModelID,
            baseModelLabel: baseModelLabel
        )
        let outputs = RemoteTrainingJobOutputs(
            logsFile: descriptor.workerContract?.logsFile ?? RemoteTrainingJobCodec.logsFileName,
            metricsFile: descriptor.workerContract?.metricsFile ?? RemoteTrainingJobCodec.metricsFileName,
            resultFile: descriptor.workerContract?.resultDescriptor ?? RemoteTrainingJobCodec.resultFileName,
            checkpointsDirectory: "remote_artifacts/checkpoints",
            finalAdapterDirectory: "remote_artifacts/final_adapter",
            updatedPackageDirectory: "remote_artifacts/package"
        )
        return RemoteTrainingJobRequest(
            anvilRemoteJobVersion: schemaVersion,
            jobID: jobID,
            createdAt: ISO8601DateFormatter().string(from: createdAt),
            task: task,
            package: RemoteTrainingJobPackage(
                packageName: descriptor.packageName,
                packageRoot: descriptor.packageRoot,
                descriptor: TrainingDatasetPackageWriter.remotePackageFileName,
                entryBundle: selectedBundle.path,
                selectedBundle: selectedBundle,
                preprocessCache: descriptor.preprocessCache,
                copyRules: descriptor.copyRules
            ),
            training: training,
            runtime: RemoteTrainingJobRuntime(
                gpuClass: descriptor.workerContract?.expectedGPUClass ?? "h100_or_h200",
                minVRAMGB: nil,
                precision: "bf16_or_fp16",
                maxRuntimeSeconds: maxRuntimeSeconds,
                maxCostUSD: maxCostUSD,
                containerImage: nil,
                modelCacheVolume: nil
            ),
            outputs: outputs,
            safety: RemoteTrainingJobSafety(
                uploadRequiresUserApproval: true,
                redactLocalPaths: true,
                writeProviderSecretsToPackage: false,
                secretsRequired: descriptor.workerContract?.secretsRequired ?? []
            )
        )
    }

    private static func trainingConfig(
        task: RemoteTrainingTask,
        packageName: String,
        adapterName: String?,
        baseModelID: String,
        baseModelLabel: String
    ) -> RemoteTrainingJobTraining? {
        let adapterMode: RemoteTrainingAdapterMode
        switch task {
        case .trainLoRA:
            adapterMode = .lora
        case .trainLoKR:
            adapterMode = .lokr
        case .preprocess, .annotate:
            return nil
        }

        return RemoteTrainingJobTraining(
            adapterMode: adapterMode,
            adapterName: adapterName ?? "\(packageName)-\(adapterMode.rawValue)",
            baseModelID: baseModelID,
            baseModelLabel: baseModelLabel,
            rank: nil,
            epochs: nil,
            batchSize: nil,
            gradientAccumulationSteps: nil,
            learningRate: nil,
            targetLayers: ["self_attn", "cross_attn"],
            trainingMode: "corrected"
        )
    }
}

private extension RemoteTrainingPackageDescriptor {
    func bundle(named name: String?) -> RemoteTrainingBundleReference? {
        guard let name else { return nil }
        return bundles.first { $0.name == name || $0.path == name }
    }

    var defaultBundleReference: RemoteTrainingBundleReference {
        bundles.first { $0.path == defaultBundle } ??
            bundles.first { $0.name == "full_mix" } ??
            RemoteTrainingBundleReference(
                name: "full_mix",
                path: defaultBundle,
                include: ["full_mix"],
                trainingLyrics: "[Instrumental]",
                trainingLyricsSource: "constant",
                optional: nil
            )
    }
}
