import Foundation
import PiMPAudioCore

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
    var datasetManifest: String?
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
        case datasetManifest = "dataset_manifest"
        case bundles
        case providerTargets = "provider_targets"
        case preprocessCache = "preprocess_cache"
        case copyRules = "copy_rules"
        case workerContract = "worker_contract"
    }

    init(
        pimpAudioRemotePackageVersion: String,
        createdAt: String?,
        packageName: String,
        packageRoot: String,
        clipCount: Int,
        defaultBundle: String,
        datasetManifest: String? = nil,
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
        self.datasetManifest = datasetManifest
        self.bundles = bundles
        self.providerTargets = providerTargets
        self.preprocessCache = preprocessCache
        self.copyRules = copyRules
        self.workerContract = workerContract
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
}

struct RemoteTrainingJobPackage: Codable, Hashable, Sendable {
    var packageName: String
    var packageRoot: String
    var descriptor: String
    var datasetManifest: String?
    var entryBundle: String
    var selectedBundle: RemoteTrainingBundleReference
    var preprocessCache: String
    var copyRules: RemoteTrainingCopyRules

    enum CodingKeys: String, CodingKey {
        case packageName = "package_name"
        case packageRoot = "package_root"
        case descriptor
        case datasetManifest = "dataset_manifest"
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
    var adapterName: String?
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
        case adapterName = "adapter_name"
        case startedAt = "started_at"
        case finishedAt = "finished_at"
        case artifacts
        case metrics
        case error
    }

    init(
        pimpAudioRemoteResultVersion: String,
        jobID: String,
        task: RemoteTrainingTask,
        status: RemoteTrainingJobStatus,
        adapterName: String? = nil,
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
        self.adapterName = adapterName
        self.startedAt = startedAt
        self.finishedAt = finishedAt
        self.artifacts = artifacts
        self.metrics = metrics
        self.error = error
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
            runScriptFileName: Data(remoteRunScript().utf8),
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

        1. Upload this zip and the portable `remote-worker/` folder to the GPU pod.
        2. SSH into the pod.
        3. Extract the job and run `run_remote_training.sh`.
        4. The script locates `remote_job.json`, installs the uploaded worker when needed, and hands the job to `pimp-remote-worker`.
        5. The worker writes logs, metrics, checkpoints, and final adapters under `remote_artifacts/`.
        6. Download/import `remote_artifacts/final_adapter/` or a folder under `remote_artifacts/checkpoints/` to test a specific epoch.

        A stock official RunPod PyTorch template is only a clean CUDA pod. It can host the job, but it does not include `pimp-remote-worker` yet. PiMP Audio Studio's pod row copies both the job zip and the portable worker folder; the extract/run command then installs that worker before training.

        ## One Command

        From the unzipped package root:

        ```bash
        bash run_remote_training.sh .
        ```

        Or pass the zip directly:

        ```bash
        bash run_remote_training.sh /workspace/jobs/\(job.package.packageName).zip
        ```

        If the worker has not been installed on the pod yet:

        ```bash
        scp -P <ssh-port> -i ~/.ssh/id_ed25519 -r remote-worker root@<pod-ip>:/workspace/pimp-remote-worker-src
        ssh -p <ssh-port> -i ~/.ssh/id_ed25519 root@<pod-ip> 'bash /workspace/pimp-remote-worker-src/install.sh'
        ```

        The script delegates zip validation and extraction to `pimp-remote-worker`, which rejects unsafe archive members before writing them. It expects either an installed `pimp-remote-worker` command, `/workspace/pimp-remote-worker-src/install.sh` from the upload command, or an official PiMP Audio Studio worker template.
        Provider API keys, SSH keys, and account IDs are intentionally not stored in this archive.
        """
    }

    private static func remoteRunScript() -> String {
        """
        #!/usr/bin/env bash
        set -euo pipefail

        INPUT="${1:-.}"

        if ! command -v pimp-remote-worker >/dev/null 2>&1 && [ -f /workspace/pimp-remote-worker-src/install.sh ]; then
          echo "[PiMP] installing uploaded remote worker"
          bash /workspace/pimp-remote-worker-src/install.sh
        fi
        if [ -f /workspace/pimp-remote-worker/env.sh ]; then
          source /workspace/pimp-remote-worker/env.sh
        fi

        if command -v pimp-remote-worker >/dev/null 2>&1; then
          exec pimp-remote-worker train "$INPUT"
        fi

        cat <<'MSG'
        No `pimp-remote-worker` command was found.

        This bundle is ready for a PiMP Audio Studio remote worker/template, but this server image does not have the worker installed yet and no uploaded worker installer was found at /workspace/pimp-remote-worker-src/install.sh.
        If you launched a stock official RunPod PyTorch pod, copy the upload command from the app's pod row first; it uploads both this job zip and the portable PiMP worker.

        Manual equivalent:

            scp -P <ssh-port> -i ~/.ssh/id_ed25519 -r remote-worker root@<pod-ip>:/workspace/pimp-remote-worker-src
            ssh -p <ssh-port> -i ~/.ssh/id_ed25519 root@<pod-ip> 'bash /workspace/pimp-remote-worker-src/install.sh'

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

struct RemoteTrainingJobValidationSummary: Hashable, Sendable {
    var selectedAssetCount: Int
    var assetRoles: [String]
    var trainingPlanID: String?
    var maximumTrainingDurationSeconds: Double?
}

enum RemoteTrainingJobValidationError: LocalizedError {
    case invalidJob(String)
    case unsafePath(String)
    case missingFile(String)
    case invalidPackageV2(String)
    case inconsistentBundle(String)

    var errorDescription: String? {
        switch self {
        case .invalidJob(let detail):
            return "Remote job validation failed: \(detail)"
        case .unsafePath(let path):
            return "Remote job contains an unsafe package-relative path: \(path)"
        case .missingFile(let path):
            return "Remote job package is missing \(path)."
        case .invalidPackageV2(let detail):
            return "Remote package-v2 validation failed: \(detail)"
        case .inconsistentBundle(let detail):
            return "Remote training bundle does not match dataset_manifest.json: \(detail)"
        }
    }
}

enum RemoteTrainingJobValidator {
    private struct BundleClip {
        var assetID: String
        var role: String
        var file: String
        var trainingRange: DatasetTimeRangeV2?
        var rangePolicy: String?
    }

    static func validate(
        _ job: RemoteTrainingJobRequest,
        packageDirectory: URL
    ) throws -> RemoteTrainingJobValidationSummary {
        guard job.pimpAudioRemoteJobVersion == RemoteTrainingJobRequest.schemaVersion else {
            throw RemoteTrainingJobValidationError.invalidJob(
                "unsupported schema \(job.pimpAudioRemoteJobVersion)"
            )
        }
        guard job.package.packageRoot == "." else {
            throw RemoteTrainingJobValidationError.invalidJob(
                "package_root must be the directory containing remote_job.json"
            )
        }
        guard !job.safety.writeProviderSecretsToPackage else {
            throw RemoteTrainingJobValidationError.invalidJob(
                "provider secrets may not be written to the archive"
            )
        }

        let descriptorURL = try packageURL(
            job.package.descriptor,
            root: packageDirectory
        )
        let bundleURL = try packageURL(
            job.package.entryBundle,
            root: packageDirectory
        )
        for requiredURL in [descriptorURL, bundleURL] {
            guard FileManager.default.fileExists(
                atPath: requiredURL.path(percentEncoded: false)
            ) else {
                throw RemoteTrainingJobValidationError.missingFile(
                    requiredURL.lastPathComponent
                )
            }
        }

        let bundleData = try Data(contentsOf: bundleURL)
        guard let bundle = try JSONSerialization.jsonObject(
            with: bundleData
        ) as? [String: Any] else {
            throw RemoteTrainingJobValidationError.invalidJob(
                "entry bundle is not a JSON object"
            )
        }
        let selectedRoles = Set(
            job.package.selectedBundle.include.map {
                $0.trimmingCharacters(in: .whitespacesAndNewlines)
            }.filter { !$0.isEmpty }
        )
        guard !selectedRoles.isEmpty else {
            throw RemoteTrainingJobValidationError.invalidJob(
                "selected_bundle.include must contain at least one asset_role"
            )
        }

        let fallbackRole = selectedRoles.count == 1
            ? selectedRoles.first
            : nil
        let selectedClips = (bundle["clips"] as? [[String: Any]] ?? [])
            .compactMap { clip -> BundleClip? in
                if let include = clip["include_in_training"] as? Bool,
                   !include {
                    return nil
                }
                let role = (clip["asset_role"] as? String)
                    ?? fallbackRole
                    ?? ""
                guard selectedRoles.contains(role) else { return nil }
                let assetID = (clip["asset_id"] as? String)
                    ?? (clip["id"] as? String)
                    ?? ""
                let file = (clip["file"] as? String) ?? ""
                guard !assetID.isEmpty, !file.isEmpty else { return nil }
                let rangeObject = clip["training_range"] as? [String: Any]
                let start = (rangeObject?["start_seconds"] as? NSNumber)?
                    .doubleValue
                let end = (rangeObject?["end_seconds"] as? NSNumber)?
                    .doubleValue
                let trainingRange = start.map {
                    DatasetTimeRangeV2(startSeconds: $0, endSeconds: end)
                }
                return BundleClip(
                    assetID: assetID,
                    role: role,
                    file: file,
                    trainingRange: trainingRange,
                    rangePolicy: clip["range_policy"] as? String
                )
            }
        guard !selectedClips.isEmpty else {
            throw RemoteTrainingJobValidationError.inconsistentBundle(
                "no included assets match \(selectedRoles.sorted().joined(separator: ", "))"
            )
        }

        let sourcePackageVersion = bundle["source_package_version"] as? String
        let trainingPlanID = bundle["training_plan_id"] as? String
        guard sourcePackageVersion == DatasetPackageV2.currentVersion else {
            return RemoteTrainingJobValidationSummary(
                selectedAssetCount: selectedClips.count,
                assetRoles: selectedRoles.sorted(),
                trainingPlanID: trainingPlanID,
                maximumTrainingDurationSeconds: selectedClips
                    .compactMap { $0.trainingRange?.durationSeconds }
                    .max()
            )
        }

        let manifestPath = job.package.datasetManifest
            ?? TrainingDatasetPackageWriter.datasetManifestFileName
        let manifestURL = try packageURL(manifestPath, root: packageDirectory)
        guard FileManager.default.fileExists(
            atPath: manifestURL.path(percentEncoded: false)
        ) else {
            throw RemoteTrainingJobValidationError.missingFile(manifestPath)
        }

        let package: DatasetPackageV2
        do {
            package = try DatasetPackageV2Codec.decodeValidated(
                Data(contentsOf: manifestURL)
            )
        } catch {
            throw RemoteTrainingJobValidationError.invalidPackageV2(
                error.localizedDescription
            )
        }
        guard let trainingPlanID,
              let plan = package.trainingPlans.first(where: {
                  $0.id == trainingPlanID
              }) else {
            throw RemoteTrainingJobValidationError.inconsistentBundle(
                "training plan \(trainingPlanID ?? "missing") is absent"
            )
        }

        let assetsByID = Dictionary(
            uniqueKeysWithValues: package.assets.map { ($0.id, $0) }
        )
        let selectionsByID = Dictionary(
            uniqueKeysWithValues: plan.selections.map { ($0.assetID, $0) }
        )
        for clip in selectedClips {
            guard let asset = assetsByID[clip.assetID] else {
                throw RemoteTrainingJobValidationError.inconsistentBundle(
                    "asset \(clip.assetID) is absent"
                )
            }
            guard asset.role.rawValue == clip.role,
                  selectedRoles.contains(asset.role.rawValue) else {
                throw RemoteTrainingJobValidationError.inconsistentBundle(
                    "asset \(clip.assetID) has role \(asset.role.rawValue), not \(clip.role)"
                )
            }
            guard asset.file.path == clip.file else {
                throw RemoteTrainingJobValidationError.inconsistentBundle(
                    "asset \(clip.assetID) file path differs from the manifest"
                )
            }
            guard let clipTrainingRange = clip.trainingRange,
                  let selection = selectionsByID[clip.assetID],
                  selection.trainingRange == clipTrainingRange,
                  selection.rangePolicy.rawValue == clip.rangePolicy else {
                throw RemoteTrainingJobValidationError.inconsistentBundle(
                    "asset \(clip.assetID) training range differs from plan \(plan.id)"
                )
            }
            guard clipTrainingRange.startSeconds >= asset.sourceRange.startSeconds,
                  let trainingEnd = clipTrainingRange.endSeconds,
                  trainingEnd > clipTrainingRange.startSeconds,
                  clipTrainingRange.durationSeconds ?? 0 <= 240 else {
                throw RemoteTrainingJobValidationError.inconsistentBundle(
                    "asset \(clip.assetID) has an invalid ACE-Step training range"
                )
            }
            if let sourceEnd = asset.sourceRange.endSeconds,
               trainingEnd > sourceEnd {
                throw RemoteTrainingJobValidationError.inconsistentBundle(
                    "asset \(clip.assetID) training range exceeds its source range"
                )
            }
            let audioURL = try packageURL(asset.file.path, root: packageDirectory)
            guard FileManager.default.fileExists(
                atPath: audioURL.path(percentEncoded: false)
            ) else {
                throw RemoteTrainingJobValidationError.missingFile(asset.file.path)
            }
        }

        return RemoteTrainingJobValidationSummary(
            selectedAssetCount: selectedClips.count,
            assetRoles: selectedRoles.sorted(),
            trainingPlanID: plan.id,
            maximumTrainingDurationSeconds: selectedClips
                .compactMap { $0.trainingRange?.durationSeconds }
                .max()
        )
    }

    private static func packageURL(_ path: String, root: URL) throws -> URL {
        let trimmed = path.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty,
              !trimmed.hasPrefix("/"),
              !trimmed.hasPrefix("~"),
              !trimmed.contains("\\") else {
            throw RemoteTrainingJobValidationError.unsafePath(path)
        }
        let components = trimmed.split(separator: "/", omittingEmptySubsequences: false)
        guard !components.contains(where: { $0 == ".." || $0.isEmpty }) else {
            throw RemoteTrainingJobValidationError.unsafePath(path)
        }
        let resolvedRoot = root.standardizedFileURL.resolvingSymlinksInPath()
        let resolved = root.appendingPathComponent(trimmed)
            .standardizedFileURL
            .resolvingSymlinksInPath()
        guard resolved.path == resolvedRoot.path
            || resolved.path.hasPrefix(resolvedRoot.path + "/") else {
            throw RemoteTrainingJobValidationError.unsafePath(path)
        }
        return resolved
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
        lokrWeightDecompose: Bool? = nil
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
                datasetManifest: descriptor.datasetManifest,
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
                // Not enforced by the worker or RunPod monitor yet — omit rather than
                // publish advisory limits that look like hard guards.
                maxRuntimeSeconds: nil,
                maxCostUSD: nil,
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
