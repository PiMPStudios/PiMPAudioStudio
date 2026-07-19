import Foundation
import PiMPAudioCore

// MARK: - MCP remote training job lifecycle (Tier A)

extension PiMPAudioCommandFacade {
    /// Create/export a remote training job zip from a reviewed dataset package.
    /// Defaults to dryRun; requires confirm for actual archive write.
    func createRemoteTrainingJob(arguments: [String: Any]) async throws -> [String: Any] {
        let dryRun = boolValue(arguments["dryRun"] ?? arguments["validateOnly"]) ?? true
        let packageDirectory = try resolveTrainingDatasetDirectory(arguments: arguments)
        let document = try TrainingDatasetPackageEditor.loadPackage(at: packageDirectory)
        guard document.isTrainingReady else {
            throw PiMPAudioCommandFacadeError.invalidTrainingConfiguration(
                "Package needs dataset review before export: \(document.blockerCount) blocker(s)."
            )
        }

        let kind = try mcpTrainingKind(from: arguments["kind"] ?? arguments["adapterKind"])
        let recipe = try mcpRemoteJobRecipe(from: arguments["recipe"])
        let bundle = try mcpRemoteJobBundle(from: arguments["bundle"] ?? arguments["selectedBundle"])
        let baseModel = try mcpRemoteJobBaseModel(from: arguments["baseModel"] ?? arguments["baseModelID"])
        let settings = recipe.settings(for: kind)

        guard FileManager.default.fileExists(
            atPath: packageDirectory.appendingPathComponent(bundle.fileName).path(percentEncoded: false)
        ) else {
            throw PiMPAudioCommandFacadeError.invalidTrainingConfiguration(
                "\(bundle.title) is not available for this package. Run stem enrichment first or choose full_mix."
            )
        }

        let descriptor = try remotePackageDescriptor(at: packageDirectory)
        let epochs = integerValue(arguments["epochs"]) ?? settings.epochs
        let saveEvery = integerValue(arguments["saveEvery"] ?? arguments["saveInterval"]) ?? settings.saveEvery
        let seed = integerValue(arguments["seed"]) ?? 42
        let adapterName = mcpRemoteAdapterName(
            arguments: arguments,
            packageName: packageDirectory.lastPathComponent,
            kind: kind,
            recipe: recipe
        )

        let job = RemoteTrainingJobRequest.makeTemplate(
            from: descriptor,
            task: kind == .lokr ? .trainLoKR : .trainLoRA,
            selectedBundleName: bundle.rawValue,
            adapterName: adapterName,
            baseModelID: baseModel.appModelID,
            baseModelLabel: baseModel.title,
            workerModelVariant: baseModel.workerVariant,
            recipe: recipe.rawValue,
            rank: kind == .lora ? settings.rank : nil,
            alpha: kind == .lora ? settings.alpha : nil,
            epochs: epochs,
            saveEvery: saveEvery,
            batchSize: settings.batchSize,
            gradientAccumulationSteps: settings.gradientAccumulationSteps,
            learningRate: settings.learningRate,
            dropout: settings.dropout,
            targetModules: settings.targetModules,
            attentionType: settings.attentionType,
            cfgRatio: settings.cfgRatio,
            seed: seed,
            maxDurationSeconds: settings.maxDurationSeconds,
            lokrLinearDim: kind == .lokr ? settings.lokrLinearDim : nil,
            lokrLinearAlpha: kind == .lokr ? settings.lokrLinearAlpha : nil,
            lokrWeightDecompose: kind == .lokr ? settings.lokrWeightDecompose : nil
        )
        let validation = try RemoteTrainingJobValidator.validate(
            job,
            packageDirectory: packageDirectory
        )
        let destination = try resolveRemoteJobDestination(
            arguments: arguments,
            packageDirectory: packageDirectory,
            kind: kind,
            recipe: recipe
        )

        var payload: [String: Any] = [
            "dryRun": dryRun,
            "jobID": job.jobID,
            "adapterName": adapterName,
            "kind": kind.rawValue,
            "recipe": recipe.rawValue,
            "bundle": bundle.rawValue,
            "baseModelID": baseModel.appModelID,
            "baseModelLabel": baseModel.title,
            "epochs": epochs,
            "saveEvery": saveEvery,
            "seed": seed,
            "packagePath": packageDirectory.path(percentEncoded: false),
            "destinationPath": destination.path(percentEncoded: false),
            "selectedAssetCount": validation.selectedAssetCount,
            "assetRoles": validation.assetRoles,
            "submitted": false
        ]

        guard !dryRun else {
            payload["confirmation"] = confirmationRequirementPayload(toolName: "create_remote_training_job")
            payload["note"] = "Dry run only. Call with dryRun: false and confirm: true to write the remote job zip."
            return payload
        }
        guard boolValue(arguments["confirm"]) == true else {
            throw PiMPAudioCommandFacadeError.confirmationRequired("create_remote_training_job")
        }

        let extraFiles = try RemoteTrainingJobCodec.archiveExtraFiles(for: job)
        let archiveURL = try await TrainingDatasetPackageWriter.exportPackageArchive(
            outputDirectory: packageDirectory,
            to: destination,
            extraFiles: extraFiles
        )
        UserDefaults.standard.set(
            archiveURL.path(percentEncoded: false),
            forKey: AppUserDefaultsKeys.runPodLastArchivePath
        )

        payload["submitted"] = true
        payload["archivePath"] = archiveURL.path(percentEncoded: false)
        payload["note"] = "Remote job zip written. Use upload_remote_training_job with this archivePath and a running pod."
        return payload
    }

    func uploadRemoteTrainingJob(arguments: [String: Any]) async throws -> [String: Any] {
        guard boolValue(arguments["confirm"]) == true else {
            throw PiMPAudioCommandFacadeError.confirmationRequired("upload_remote_training_job")
        }
        guard let runPodService else {
            throw PiMPAudioCommandFacadeError.runPodServiceUnavailable
        }
        let pod = try resolveRunPodPod(arguments: arguments, service: runPodService)
        let archiveURL = try resolveRemoteArchiveURL(arguments: arguments)
        let sshKeyPath = try resolveMCPSSHKeyPath(arguments: arguments)

        await runPodService.uploadRemoteJob(
            to: pod,
            archiveURL: archiveURL,
            sshKeyPath: sshKeyPath
        )

        return remoteJobActionPayload(
            action: "upload",
            service: runPodService,
            pod: pod,
            archiveURL: archiveURL,
            submitted: runPodService.lastError == nil
        )
    }

    func startRemoteTrainingJob(arguments: [String: Any]) async throws -> [String: Any] {
        guard boolValue(arguments["confirm"]) == true else {
            throw PiMPAudioCommandFacadeError.confirmationRequired("start_remote_training_job")
        }
        guard let runPodService else {
            throw PiMPAudioCommandFacadeError.runPodServiceUnavailable
        }
        let pod = try resolveRunPodPod(arguments: arguments, service: runPodService)
        let archiveURL = try resolveRemoteArchiveURL(arguments: arguments)
        let sshKeyPath = try resolveMCPSSHKeyPath(arguments: arguments)

        await runPodService.startRemoteJob(
            on: pod,
            archiveURL: archiveURL,
            sshKeyPath: sshKeyPath
        )

        return remoteJobActionPayload(
            action: "start",
            service: runPodService,
            pod: pod,
            archiveURL: archiveURL,
            submitted: runPodService.lastError == nil
        )
    }

    func remoteTrainingJobStatus(arguments: [String: Any]) async throws -> [String: Any] {
        guard let runPodService else {
            throw PiMPAudioCommandFacadeError.runPodServiceUnavailable
        }

        let refresh = boolValue(arguments["refresh"] ?? arguments["monitor"]) ?? true
        let archiveURL = try? resolveRemoteArchiveURL(arguments: arguments)
        var pod: RunPodPodSummary?
        if let podID = stringValue(arguments["podID"] ?? arguments["pod"] ?? arguments["id"])?
            .trimmingCharacters(in: .whitespacesAndNewlines),
           !podID.isEmpty {
            pod = try resolveRunPodPod(arguments: arguments, service: runPodService)
        }

        if refresh,
           let pod,
           let archiveURL {
            let sshKeyPath = try resolveMCPSSHKeyPath(arguments: arguments)
            await runPodService.refreshRemoteJob(
                on: pod,
                archiveURL: archiveURL,
                sshKeyPath: sshKeyPath
            )
        } else if boolValue(arguments["refreshPods"]) == true {
            await runPodService.refreshPods()
        }

        let state: RunPodRemoteJobState?
        if let pod {
            state = runPodService.remoteJobState(for: pod, archiveURL: archiveURL)
        } else {
            state = nil
        }

        return [
            "pod": pod.map(runPodPodPayload) ?? NSNull(),
            "archivePath": archiveURL?.path(percentEncoded: false) ?? NSNull(),
            "job": state.map(remoteJobStatePayload) ?? NSNull(),
            "remoteJobs": runPodService.remoteJobStates.map { id, job in
                var entry = remoteJobStatePayload(job)
                entry["podID"] = id
                return entry
            },
            "lastMessage": runPodService.lastMessage ?? NSNull(),
            "lastError": runPodService.lastError ?? NSNull()
        ]
    }

    func pullRemoteTrainingJob(arguments: [String: Any]) async throws -> [String: Any] {
        guard boolValue(arguments["confirm"]) == true else {
            throw PiMPAudioCommandFacadeError.confirmationRequired("pull_remote_training_job")
        }
        guard let runPodService else {
            throw PiMPAudioCommandFacadeError.runPodServiceUnavailable
        }
        let pod = try resolveRunPodPod(arguments: arguments, service: runPodService)
        let archiveURL = try resolveRemoteArchiveURL(arguments: arguments)
        let sshKeyPath = try resolveMCPSSHKeyPath(arguments: arguments)
        let destinationParent = try resolvePullDestinationParent(arguments: arguments, archiveURL: archiveURL)
        let importAdapter = boolValue(arguments["importAdapter"] ?? arguments["import"]) ?? true

        let pullback = try await runPodService.downloadRemoteJobArtifacts(
            from: pod,
            archiveURL: archiveURL,
            sshKeyPath: sshKeyPath,
            destinationParentURL: destinationParent
        )

        var payload: [String: Any] = [
            "submitted": true,
            "podID": pod.id,
            "archivePath": archiveURL.path(percentEncoded: false),
            "pullbackDirectory": pullback.directoryURL.path(percentEncoded: false),
            "artifactsPath": pullback.artifactsURL.path(percentEncoded: false),
            "finalAdapterPath": pullback.finalAdapterURL.path(percentEncoded: false),
            "resultStatus": pullback.result.status.rawValue,
            "adapterNameFromResult": pullback.result.adapterName ?? NSNull(),
            "imported": false
        ]

        if importAdapter {
            let importer = CommunityService()
            let adapter = try await importer.importLocalAdapter(
                directoryURL: pullback.finalAdapterURL,
                preferredName: pullback.result.adapterName
            )
            runPodService.markRemoteAdapterImported(
                podID: pod.id,
                archiveURL: archiveURL,
                adapterName: adapter.name
            )
            payload["imported"] = true
            payload["adapter"] = [
                "id": adapter.id,
                "name": adapter.name,
                "kind": adapter.kind.rawValue,
                "baseModelID": adapter.baseModelID
            ]
        }

        payload["job"] = runPodService.remoteJobState(for: pod, archiveURL: archiveURL)
            .map(remoteJobStatePayload) ?? NSNull()
        payload["lastMessage"] = runPodService.lastMessage ?? NSNull()
        payload["lastError"] = runPodService.lastError ?? NSNull()
        return payload
    }

    // MARK: - Remote job helpers

    private func remotePackageDescriptor(at packageDirectory: URL) throws -> RemoteTrainingPackageDescriptor {
        let descriptorURL = packageDirectory.appendingPathComponent(
            TrainingDatasetPackageWriter.remotePackageFileName
        )
        let data = try Data(contentsOf: descriptorURL)
        return try RemoteTrainingJobCodec.decoder.decode(
            RemoteTrainingPackageDescriptor.self,
            from: data
        )
    }

    private func resolveRemoteJobDestination(
        arguments: [String: Any],
        packageDirectory: URL,
        kind: AdapterKind,
        recipe: MCPRemoteJobRecipe
    ) throws -> URL {
        if let raw = stringValue(
            arguments["destinationPath"]
                ?? arguments["outputPath"]
                ?? arguments["archivePath"]
        )?.trimmingCharacters(in: .whitespacesAndNewlines),
           !raw.isEmpty {
            let url = URL(fileURLWithPath: NSString(string: raw).expandingTildeInPath)
            if url.pathExtension.lowercased() == "zip" {
                return url
            }
            return url.appendingPathComponent(
                "\(packageDirectory.lastPathComponent)_remote_\(kind.rawValue)_\(recipe.rawValue).zip"
            )
        }

        return packageDirectory
            .deletingLastPathComponent()
            .appendingPathComponent(
                "\(packageDirectory.lastPathComponent)_remote_\(kind.rawValue)_\(recipe.rawValue).zip"
            )
    }

    private func resolveRemoteArchiveURL(arguments: [String: Any]) throws -> URL {
        if let raw = stringValue(
            arguments["archivePath"]
                ?? arguments["jobZip"]
                ?? arguments["zipPath"]
                ?? arguments["path"]
        )?.trimmingCharacters(in: .whitespacesAndNewlines),
           !raw.isEmpty {
            let url = URL(fileURLWithPath: NSString(string: raw).expandingTildeInPath)
            guard FileManager.default.fileExists(atPath: url.path(percentEncoded: false)) else {
                throw PiMPAudioCommandFacadeError.invalidTrainingConfiguration(
                    "Remote job zip not found at \(url.path(percentEncoded: false))."
                )
            }
            return url
        }

        if let remembered = UserDefaults.standard.string(forKey: AppUserDefaultsKeys.runPodLastArchivePath)?
            .trimmingCharacters(in: .whitespacesAndNewlines),
           !remembered.isEmpty {
            let url = URL(fileURLWithPath: remembered)
            if FileManager.default.fileExists(atPath: url.path(percentEncoded: false)) {
                return url
            }
        }

        throw PiMPAudioCommandFacadeError.invalidTrainingConfiguration(
            "Provide archivePath to a remote job zip, or create one with create_remote_training_job first."
        )
    }

    private func resolveMCPSSHKeyPath(arguments: [String: Any]) throws -> String {
        if let raw = stringValue(arguments["sshKeyPath"] ?? arguments["sshKey"] ?? arguments["privateKeyPath"])?
            .trimmingCharacters(in: .whitespacesAndNewlines),
           !raw.isEmpty {
            return raw
        }

        // Prefer the same managed key path the Remote GPU UI writes into.
        let store = RunPodSSHKeyStore()
        if store.hasManagedPrivateKey {
            return store.managedPrivateKeyURL.path(percentEncoded: false)
        }

        // Fall back to the same defaults key the Remote GPU UI uses.
        if let stored = UserDefaults.standard.string(forKey: AppUserDefaultsKeys.runPodSSHKeyPath)?
            .trimmingCharacters(in: .whitespacesAndNewlines),
           !stored.isEmpty {
            return stored
        }

        throw PiMPAudioCommandFacadeError.invalidTrainingConfiguration(
            "No SSH private key configured. Choose a key in Build → Training → Remote GPU, or pass sshKeyPath."
        )
    }

    private func resolveRunPodPod(
        arguments: [String: Any],
        service: RunPodService
    ) throws -> RunPodPodSummary {
        let podID = try requiredString("podID", aliases: ["pod", "id"], in: arguments)
        if let pod = service.pods.first(where: { $0.id == podID }) {
            return pod
        }
        throw PiMPAudioCommandFacadeError.invalidTrainingConfiguration(
            "Unknown RunPod podID '\(podID)'. Call runpod_status with refreshPods: true first."
        )
    }

    private func resolvePullDestinationParent(
        arguments: [String: Any],
        archiveURL: URL
    ) throws -> URL {
        if let raw = stringValue(
            arguments["destinationDirectory"]
                ?? arguments["outputDirectory"]
                ?? arguments["destinationPath"]
        )?.trimmingCharacters(in: .whitespacesAndNewlines),
           !raw.isEmpty {
            let url = URL(fileURLWithPath: NSString(string: raw).expandingTildeInPath, isDirectory: true)
            try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
            return url
        }

        let parent = archiveURL
            .deletingLastPathComponent()
            .appendingPathComponent("Remote Training Results", isDirectory: true)
        try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
        return parent
    }

    private func remoteJobActionPayload(
        action: String,
        service: RunPodService,
        pod: RunPodPodSummary,
        archiveURL: URL,
        submitted: Bool
    ) -> [String: Any] {
        [
            "action": action,
            "submitted": submitted,
            "podID": pod.id,
            "pod": runPodPodPayload(pod),
            "archivePath": archiveURL.path(percentEncoded: false),
            "job": service.remoteJobState(for: pod, archiveURL: archiveURL)
                .map(remoteJobStatePayload) ?? NSNull(),
            "lastMessage": service.lastMessage ?? NSNull(),
            "lastError": service.lastError ?? NSNull()
        ]
    }

    private func remoteJobStatePayload(_ state: RunPodRemoteJobState) -> [String: Any] {
        [
            "archivePath": state.archivePath,
            "phase": state.phase.rawValue,
            "message": state.message,
            "logTail": state.logTail,
            "localArtifactsPath": state.localArtifactsPath ?? NSNull(),
            "importedAdapterName": state.importedAdapterName ?? NSNull(),
            "resultStatus": state.result?.status.rawValue ?? NSNull(),
            "resultAdapterName": state.result?.adapterName ?? NSNull()
        ]
    }

    private func mcpRemoteAdapterName(
        arguments: [String: Any],
        packageName: String,
        kind: AdapterKind,
        recipe: MCPRemoteJobRecipe
    ) -> String {
        if let name = stringValue(arguments["adapterName"] ?? arguments["name"])?
            .trimmingCharacters(in: .whitespacesAndNewlines),
           !name.isEmpty {
            return name
        }
        let friendly = packageName.replacingOccurrences(of: "_local_", with: " ")
        return "\(friendly) \(kind == .lokr ? "LoKR" : "LoRA") \(recipe.title)"
    }

    private func mcpRemoteJobRecipe(from value: Any?) throws -> MCPRemoteJobRecipe {
        guard let raw = stringValue(value)?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased(),
              !raw.isEmpty else {
            return .wide
        }
        switch raw {
        case "narrow", "balanced", "balancedlocal":
            return .narrow
        case "wide", "expanded":
            return .wide
        default:
            throw PiMPAudioCommandFacadeError.invalidTrainingRecipe(raw)
        }
    }

    private func mcpRemoteJobBundle(from value: Any?) throws -> MCPRemoteJobBundle {
        guard let raw = stringValue(value)?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased(),
              !raw.isEmpty else {
            return .fullMix
        }
        switch raw {
        case "full_mix", "fullmix", "full-mix", "mix":
            return .fullMix
        case "vocals", "vocal":
            return .vocals
        case "instrumental", "instruments":
            return .instrumental
        default:
            throw PiMPAudioCommandFacadeError.invalidTrainingConfiguration(
                "Unsupported bundle '\(raw)'. Use full_mix, vocals, or instrumental."
            )
        }
    }

    private func mcpRemoteJobBaseModel(from value: Any?) throws -> MCPRemoteJobBaseModel {
        guard let raw = stringValue(value)?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased(),
              !raw.isEmpty else {
            return .xlSFT
        }
        switch raw {
        case "xlsft", "xl-sft", "ace-step-dit-xl-sft", "ace-step-xl-sft", "sft":
            return .xlSFT
        case "xlbase", "xl-base", "ace-step-dit-xl-base", "ace-step-xl-base", "base":
            return .xlBase
        case "xlturbo", "xl-turbo", "ace-step-dit-xl-turbo", "ace-step-xl-turbo", "turbo":
            return .xlTurbo
        default:
            throw PiMPAudioCommandFacadeError.invalidTrainingConfiguration(
                "Unsupported baseModel '\(raw)'. Use xlSFT, xlBase, or xlTurbo."
            )
        }
    }
}

// MARK: - Shared remote job recipe mirrors (UI-private enums are not accessible)

private enum MCPRemoteJobRecipe: String {
    case narrow
    case wide

    var title: String {
        switch self {
        case .narrow: return "Narrow"
        case .wide: return "Wide"
        }
    }

    func settings(for adapterKind: AdapterKind) -> MCPRemoteJobRecipeSettings {
        switch (adapterKind, self) {
        case (.lora, .narrow):
            return MCPRemoteJobRecipeSettings(
                rank: 64, alpha: 64, lokrLinearDim: nil, lokrLinearAlpha: nil,
                lokrWeightDecompose: false, epochs: 100, saveEvery: 10, batchSize: 1,
                gradientAccumulationSteps: 2, learningRate: 0.00008, dropout: 0,
                targetModules: ["q_proj", "v_proj"], attentionType: "both",
                cfgRatio: 0.15, maxDurationSeconds: 240
            )
        case (.lora, .wide):
            return MCPRemoteJobRecipeSettings(
                rank: 96, alpha: 192, lokrLinearDim: nil, lokrLinearAlpha: nil,
                lokrWeightDecompose: false, epochs: 150, saveEvery: 10, batchSize: 1,
                gradientAccumulationSteps: 2, learningRate: 0.000065, dropout: 0,
                targetModules: ["q_proj", "k_proj", "v_proj", "o_proj"], attentionType: "both",
                cfgRatio: 0.15, maxDurationSeconds: 240
            )
        case (.lokr, .narrow):
            return MCPRemoteJobRecipeSettings(
                rank: 32, alpha: 64, lokrLinearDim: 32, lokrLinearAlpha: 64,
                lokrWeightDecompose: true, epochs: 80, saveEvery: 10, batchSize: 1,
                gradientAccumulationSteps: 2, learningRate: 0.00008, dropout: 0,
                targetModules: ["q_proj", "v_proj"], attentionType: "both",
                cfgRatio: 0.15, maxDurationSeconds: 240
            )
        case (.lokr, .wide):
            return MCPRemoteJobRecipeSettings(
                rank: 128, alpha: 256, lokrLinearDim: 128, lokrLinearAlpha: 256,
                lokrWeightDecompose: true, epochs: 150, saveEvery: 10, batchSize: 1,
                gradientAccumulationSteps: 2, learningRate: 0.000065, dropout: 0,
                targetModules: ["q_proj", "k_proj", "v_proj", "o_proj"], attentionType: "both",
                cfgRatio: 0.15, maxDurationSeconds: 240
            )
        }
    }
}

private struct MCPRemoteJobRecipeSettings {
    let rank: Int
    let alpha: Double
    let lokrLinearDim: Int?
    let lokrLinearAlpha: Double?
    let lokrWeightDecompose: Bool
    let epochs: Int
    let saveEvery: Int
    let batchSize: Int
    let gradientAccumulationSteps: Int
    let learningRate: Double
    let dropout: Double
    let targetModules: [String]
    let attentionType: String
    let cfgRatio: Double
    let maxDurationSeconds: Int
}

private enum MCPRemoteJobBundle: String {
    case fullMix = "full_mix"
    case instrumental
    case vocals

    var title: String {
        switch self {
        case .fullMix: return "Full Mix"
        case .instrumental: return "Instrumental"
        case .vocals: return "Vocals"
        }
    }

    var fileName: String {
        switch self {
        case .fullMix:
            return TrainingDatasetPackageWriter.fullMixBundleFileName
        case .instrumental:
            return TrainingDatasetPackageWriter.instrumentalBundleFileName
        case .vocals:
            return TrainingDatasetPackageWriter.vocalBundleFileName
        }
    }
}

private enum MCPRemoteJobBaseModel {
    case xlSFT
    case xlBase
    case xlTurbo

    var title: String {
        switch self {
        case .xlSFT: return "ACE-Step XL SFT"
        case .xlBase: return "ACE-Step XL Base"
        case .xlTurbo: return "ACE-Step XL Turbo"
        }
    }

    var appModelID: String {
        switch self {
        case .xlSFT: return "ace-step-dit-xl-sft"
        case .xlBase: return "ace-step-dit-xl-base"
        case .xlTurbo: return "ace-step-dit-xl-turbo"
        }
    }

    var workerVariant: String {
        switch self {
        case .xlSFT: return "ace-step-xl-sft"
        case .xlBase: return "ace-step-xl-base"
        case .xlTurbo: return "ace-step-xl-turbo"
        }
    }
}
