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
    var pimpAudioRemotePackageVersion: String
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
        case pimpAudioRemotePackageVersion = "pimp_audio_remote_package_version"
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

    private enum LegacyCodingKeys: String, CodingKey {
        case anvilRemotePackageVersion = "anvil_remote_package_version"
    }

    init(
        pimpAudioRemotePackageVersion: String,
        createdAt: String?,
        packageName: String,
        packageRoot: String,
        clipCount: Int,
        defaultBundle: String,
        bundles: [RemoteTrainingBundleReference],
        providerTargets: [String],
        preprocessCache: String,
        copyRules: RemoteTrainingCopyRules,
        workerContract: RemoteWorkerContract?
    ) {
        self.pimpAudioRemotePackageVersion = pimpAudioRemotePackageVersion
        self.createdAt = createdAt
        self.packageName = packageName
        self.packageRoot = packageRoot
        self.clipCount = clipCount
        self.defaultBundle = defaultBundle
        self.bundles = bundles
        self.providerTargets = providerTargets
        self.preprocessCache = preprocessCache
        self.copyRules = copyRules
        self.workerContract = workerContract
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let legacyContainer = try decoder.container(keyedBy: LegacyCodingKeys.self)
        self.pimpAudioRemotePackageVersion = try container.decodeIfPresent(
            String.self,
            forKey: .pimpAudioRemotePackageVersion
        ) ?? legacyContainer.decode(String.self, forKey: .anvilRemotePackageVersion)
        self.createdAt = try container.decodeIfPresent(String.self, forKey: .createdAt)
        self.packageName = try container.decode(String.self, forKey: .packageName)
        self.packageRoot = try container.decode(String.self, forKey: .packageRoot)
        self.clipCount = try container.decode(Int.self, forKey: .clipCount)
        self.defaultBundle = try container.decode(String.self, forKey: .defaultBundle)
        self.bundles = try container.decode([RemoteTrainingBundleReference].self, forKey: .bundles)
        self.providerTargets = try container.decode([String].self, forKey: .providerTargets)
        self.preprocessCache = try container.decode(String.self, forKey: .preprocessCache)
        self.copyRules = try container.decode(RemoteTrainingCopyRules.self, forKey: .copyRules)
        self.workerContract = try container.decodeIfPresent(RemoteWorkerContract.self, forKey: .workerContract)
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

    var pimpAudioRemoteJobVersion: String
    var jobID: String
    var createdAt: String
    var task: RemoteTrainingTask
    var package: RemoteTrainingJobPackage
    var training: RemoteTrainingJobTraining?
    var runtime: RemoteTrainingJobRuntime
    var outputs: RemoteTrainingJobOutputs
    var safety: RemoteTrainingJobSafety

    enum CodingKeys: String, CodingKey {
        case pimpAudioRemoteJobVersion = "pimp_audio_remote_job_version"
        case jobID = "job_id"
        case createdAt = "created_at"
        case task
        case package
        case training
        case runtime
        case outputs
        case safety
    }

    private enum LegacyCodingKeys: String, CodingKey {
        case anvilRemoteJobVersion = "anvil_remote_job_version"
    }

    init(
        pimpAudioRemoteJobVersion: String,
        jobID: String,
        createdAt: String,
        task: RemoteTrainingTask,
        package: RemoteTrainingJobPackage,
        training: RemoteTrainingJobTraining?,
        runtime: RemoteTrainingJobRuntime,
        outputs: RemoteTrainingJobOutputs,
        safety: RemoteTrainingJobSafety
    ) {
        self.pimpAudioRemoteJobVersion = pimpAudioRemoteJobVersion
        self.jobID = jobID
        self.createdAt = createdAt
        self.task = task
        self.package = package
        self.training = training
        self.runtime = runtime
        self.outputs = outputs
        self.safety = safety
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let legacyContainer = try decoder.container(keyedBy: LegacyCodingKeys.self)
        self.pimpAudioRemoteJobVersion = try container.decodeIfPresent(
            String.self,
            forKey: .pimpAudioRemoteJobVersion
        ) ?? legacyContainer.decode(String.self, forKey: .anvilRemoteJobVersion)
        self.jobID = try container.decode(String.self, forKey: .jobID)
        self.createdAt = try container.decode(String.self, forKey: .createdAt)
        self.task = try container.decode(RemoteTrainingTask.self, forKey: .task)
        self.package = try container.decode(RemoteTrainingJobPackage.self, forKey: .package)
        self.training = try container.decodeIfPresent(RemoteTrainingJobTraining.self, forKey: .training)
        self.runtime = try container.decode(RemoteTrainingJobRuntime.self, forKey: .runtime)
        self.outputs = try container.decode(RemoteTrainingJobOutputs.self, forKey: .outputs)
        self.safety = try container.decode(RemoteTrainingJobSafety.self, forKey: .safety)
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
    var workerModelVariant: String?
    var recipe: String?
    var rank: Int?
    var alpha: Double?
    var epochs: Int?
    var saveEvery: Int?
    var batchSize: Int?
    var gradientAccumulationSteps: Int?
    var learningRate: Double?
    var dropout: Double?
    var targetModules: [String]?
    var targetLayers: [String]
    var attentionType: String?
    var cfgRatio: Double?
    var seed: Int?
    var maxDurationSeconds: Int?
    var lokrLinearDim: Int?
    var lokrLinearAlpha: Double?
    var lokrWeightDecompose: Bool?
    var trainingMode: String

    enum CodingKeys: String, CodingKey {
        case adapterMode = "adapter_mode"
        case adapterName = "adapter_name"
        case baseModelID = "base_model_id"
        case baseModelLabel = "base_model_label"
        case workerModelVariant = "worker_model_variant"
        case recipe
        case rank
        case alpha
        case epochs
        case saveEvery = "save_every"
        case batchSize = "batch_size"
        case gradientAccumulationSteps = "gradient_accumulation_steps"
        case learningRate = "learning_rate"
        case dropout
        case targetModules = "target_modules"
        case targetLayers = "target_layers"
        case attentionType = "attention_type"
        case cfgRatio = "cfg_ratio"
        case seed
        case maxDurationSeconds = "max_duration_seconds"
        case lokrLinearDim = "lokr_linear_dim"
        case lokrLinearAlpha = "lokr_linear_alpha"
        case lokrWeightDecompose = "lokr_weight_decompose"
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

    var pimpAudioRemoteResultVersion: String
    var jobID: String
    var task: RemoteTrainingTask
    var status: RemoteTrainingJobStatus
    var startedAt: String?
    var finishedAt: String?
    var artifacts: [RemoteTrainingJobArtifact]
    var metrics: [String: Double]
    var error: RemoteTrainingJobError?

    enum CodingKeys: String, CodingKey {
        case pimpAudioRemoteResultVersion = "pimp_audio_remote_result_version"
        case jobID = "job_id"
        case task
        case status
        case startedAt = "started_at"
        case finishedAt = "finished_at"
        case artifacts
        case metrics
        case error
    }

    private enum LegacyCodingKeys: String, CodingKey {
        case anvilRemoteResultVersion = "anvil_remote_result_version"
    }

    init(
        pimpAudioRemoteResultVersion: String,
        jobID: String,
        task: RemoteTrainingTask,
        status: RemoteTrainingJobStatus,
        startedAt: String?,
        finishedAt: String?,
        artifacts: [RemoteTrainingJobArtifact],
        metrics: [String: Double],
        error: RemoteTrainingJobError?
    ) {
        self.pimpAudioRemoteResultVersion = pimpAudioRemoteResultVersion
        self.jobID = jobID
        self.task = task
        self.status = status
        self.startedAt = startedAt
        self.finishedAt = finishedAt
        self.artifacts = artifacts
        self.metrics = metrics
        self.error = error
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let legacyContainer = try decoder.container(keyedBy: LegacyCodingKeys.self)
        self.pimpAudioRemoteResultVersion = try container.decodeIfPresent(
            String.self,
            forKey: .pimpAudioRemoteResultVersion
        ) ?? legacyContainer.decode(String.self, forKey: .anvilRemoteResultVersion)
        self.jobID = try container.decode(String.self, forKey: .jobID)
        self.task = try container.decode(RemoteTrainingTask.self, forKey: .task)
        self.status = try container.decode(RemoteTrainingJobStatus.self, forKey: .status)
        self.startedAt = try container.decodeIfPresent(String.self, forKey: .startedAt)
        self.finishedAt = try container.decodeIfPresent(String.self, forKey: .finishedAt)
        self.artifacts = try container.decode([RemoteTrainingJobArtifact].self, forKey: .artifacts)
        self.metrics = try container.decode([String: Double].self, forKey: .metrics)
        self.error = try container.decodeIfPresent(RemoteTrainingJobError.self, forKey: .error)
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
    static let readmeFileName = "README_REMOTE_GPU.md"
    static let runScriptFileName = "run_remote_training.sh"

    static var encoder: JSONEncoder {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return encoder
    }

    static var decoder: JSONDecoder {
        JSONDecoder()
    }

    static func archiveExtraFiles(for job: RemoteTrainingJobRequest) throws -> [String: Data] {
        [
            jobFileName: try encoder.encode(job),
            readmeFileName: Data(remoteReadme(for: job).utf8),
            runScriptFileName: Data(remoteRunScript(for: job).utf8),
        ]
    }

    private static func remoteReadme(for job: RemoteTrainingJobRequest) -> String {
        let training = job.training
        let adapterMode = training?.adapterMode.rawValue ?? job.task.rawValue
        let recipe = training?.recipe ?? "custom"
        let bundle = job.package.selectedBundle.name
        let adapterName = training?.adapterName ?? job.package.packageName
        let workerVariant = training?.workerModelVariant ?? training?.baseModelID ?? "ace-step-xl-sft"
        let epochs = training?.epochs.map(String.init) ?? "worker default"
        let saveEvery = training?.saveEvery.map(String.init) ?? "worker default"
        let targetModules = training?.targetModules?.joined(separator: ", ") ?? "worker default"

        return """
        # PiMP Audio Studio Remote GPU Job

        This archive contains a reviewed dataset package plus `remote_job.json`.
        It is intended for a PiMP Audio Studio remote worker or a compatible RunPod/DigitalOcean template.

        ## Job Summary

        - Adapter: \(adapterName)
        - Mode: \(adapterMode)
        - Recipe: \(recipe)
        - Bundle: \(bundle)
        - Base model: \(training?.baseModelLabel ?? "ACE-Step XL SFT") (\(workerVariant))
        - Epochs: \(epochs)
        - Save every: \(saveEvery) epoch(s)
        - Target modules: \(targetModules)

        ## Expected Remote Flow

        1. Upload this zip to a GPU worker.
        2. Unzip it or pass the zip path to `run_remote_training.sh`.
        3. The worker reads `remote_job.json`.
        4. The worker writes logs, metrics, checkpoints, and final adapters under `remote_artifacts/`.
        5. Download the folder named `pimp_native_adapter` from the checkpoint or final artifact you want to test.

        ## One Command

        From the unzipped package root:

        ```bash
        bash run_remote_training.sh .
        ```

        Or pass the zip directly:

        ```bash
        bash run_remote_training.sh /workspace/jobs/\(job.package.packageName).zip
        ```

        The script expects an installed `pimp-remote-worker` command or an official PiMP Audio Studio worker template.
        Provider API keys, SSH keys, and account IDs are intentionally not stored in this archive.
        """
    }

    private static func remoteRunScript(for job: RemoteTrainingJobRequest) -> String {
        """
        #!/usr/bin/env bash
        set -euo pipefail

        INPUT="${1:-.}"
        WORKDIR=""

        if [ -f "$INPUT" ]; then
          WORKDIR="/workspace/pimp-remote-jobs/\(job.jobID)"
          mkdir -p "$WORKDIR"
          unzip -oq "$INPUT" -d "$WORKDIR"
          JOB_FILE="$(find "$WORKDIR" -maxdepth 3 -name remote_job.json -print -quit)"
          if [ -n "$JOB_FILE" ]; then
            PACKAGE_ROOT="$(dirname "$JOB_FILE")"
          fi
        else
          PACKAGE_ROOT="$INPUT"
        fi

        if [ -z "${PACKAGE_ROOT:-}" ] || [ ! -f "$PACKAGE_ROOT/remote_job.json" ]; then
          echo "Could not find remote_job.json. Run this from the package root or pass the exported zip path."
          exit 2
        fi

        cd "$PACKAGE_ROOT"
        echo "[PiMP] package_root=$PWD"
        echo "[PiMP] job=$(grep -m1 '"job_id"' remote_job.json | sed 's/[",]//g' | xargs || true)"
        echo "[PiMP] task=$(grep -m1 '"task"' remote_job.json | sed 's/[",]//g' | xargs || true)"

        if command -v pimp-remote-worker >/dev/null 2>&1; then
          exec pimp-remote-worker train remote_job.json
        fi

        cat <<'MSG'
        No `pimp-remote-worker` command was found.

        This bundle is ready for a PiMP Audio Studio remote worker/template, but this server image does not have the worker installed yet.

        Use an official PiMP Audio Studio RunPod/DigitalOcean template, or install the worker, then run:

            pimp-remote-worker train remote_job.json

        The worker should write:
          - remote_artifacts/logs.jsonl
          - remote_artifacts/metrics.jsonl
          - remote_artifacts/checkpoints/
          - remote_artifacts/final_adapter/
          - remote_artifacts/result.json
        MSG
        exit 127
        """
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
        baseModelID: String = "ace-step-dit-xl-sft",
        baseModelLabel: String = "ACE-Step XL SFT",
        workerModelVariant: String? = "ace-step-xl-sft",
        recipe: String? = nil,
        rank: Int? = nil,
        alpha: Double? = nil,
        epochs: Int? = nil,
        saveEvery: Int? = nil,
        batchSize: Int? = nil,
        gradientAccumulationSteps: Int? = nil,
        learningRate: Double? = nil,
        dropout: Double? = nil,
        targetModules: [String]? = nil,
        attentionType: String? = nil,
        cfgRatio: Double? = nil,
        seed: Int? = nil,
        maxDurationSeconds: Int? = nil,
        lokrLinearDim: Int? = nil,
        lokrLinearAlpha: Double? = nil,
        lokrWeightDecompose: Bool? = nil,
        maxRuntimeSeconds: Int? = nil,
        maxCostUSD: Double? = nil
    ) -> RemoteTrainingJobRequest {
        let selectedBundle = descriptor.bundle(named: selectedBundleName) ?? descriptor.defaultBundleReference
        let training = trainingConfig(
            task: task,
            packageName: descriptor.packageName,
            adapterName: adapterName,
            baseModelID: baseModelID,
            baseModelLabel: baseModelLabel,
            workerModelVariant: workerModelVariant,
            recipe: recipe,
            rank: rank,
            alpha: alpha,
            epochs: epochs,
            saveEvery: saveEvery,
            batchSize: batchSize,
            gradientAccumulationSteps: gradientAccumulationSteps,
            learningRate: learningRate,
            dropout: dropout,
            targetModules: targetModules,
            attentionType: attentionType,
            cfgRatio: cfgRatio,
            seed: seed,
            maxDurationSeconds: maxDurationSeconds,
            lokrLinearDim: lokrLinearDim,
            lokrLinearAlpha: lokrLinearAlpha,
            lokrWeightDecompose: lokrWeightDecompose
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
            pimpAudioRemoteJobVersion: schemaVersion,
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
        baseModelLabel: String,
        workerModelVariant: String?,
        recipe: String?,
        rank: Int?,
        alpha: Double?,
        epochs: Int?,
        saveEvery: Int?,
        batchSize: Int?,
        gradientAccumulationSteps: Int?,
        learningRate: Double?,
        dropout: Double?,
        targetModules: [String]?,
        attentionType: String?,
        cfgRatio: Double?,
        seed: Int?,
        maxDurationSeconds: Int?,
        lokrLinearDim: Int?,
        lokrLinearAlpha: Double?,
        lokrWeightDecompose: Bool?
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
            workerModelVariant: workerModelVariant,
            recipe: recipe,
            rank: rank,
            alpha: alpha,
            epochs: epochs,
            saveEvery: saveEvery,
            batchSize: batchSize,
            gradientAccumulationSteps: gradientAccumulationSteps,
            learningRate: learningRate,
            dropout: dropout,
            targetModules: targetModules,
            targetLayers: ["self_attn", "cross_attn"],
            attentionType: attentionType,
            cfgRatio: cfgRatio,
            seed: seed,
            maxDurationSeconds: maxDurationSeconds,
            lokrLinearDim: lokrLinearDim,
            lokrLinearAlpha: lokrLinearAlpha,
            lokrWeightDecompose: lokrWeightDecompose,
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
