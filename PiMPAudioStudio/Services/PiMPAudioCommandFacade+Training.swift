import Foundation
import AVFoundation
import PiMPAudioCore

extension PiMPAudioCommandFacade {
    func trainingStatus() -> [String: Any] {
        guard let trainingService else {
            return [
                "available": false,
                "state": "unavailable",
                "message": "Training service is not wired into MCP."
            ]
        }

        let activeRun: Any
        if let log = trainingService.activeRunLogSnapshot {
            activeRun = trainingRunLogPayload(log)
        } else {
            activeRun = NSNull()
        }
        return [
            "available": true,
            "state": trainingStatePayload(trainingService.state),
            "canSaveCheckpointOnStop": trainingService.canSaveOnStop,
            "activeRun": activeRun,
            "diagnostics": diagnosticsStatusPayload(),
            "resources": [
                "datasets": "pimpaudio://training-datasets",
                "runs": "pimpaudio://training-runs",
                "operations": "pimpaudio://operations"
            ]
        ]
    }

    func listTrainingDatasets(arguments: [String: Any]) -> [String: Any] {
        let limit = max(1, min(integerValue(arguments["limit"]) ?? 50, 200))
        let roots = trainingDatasetSearchRoots()
        var seen = Set<String>()
        var packages: [DatasetPackageSummary] = []

        for root in roots {
            for package in TrainingDatasetPackageEditor.listPackages(in: root) {
                guard seen.insert(package.id).inserted else { continue }
                packages.append(package)
            }
        }

        let rows = packages
            .sorted { lhs, rhs in
                modificationDate(lhs.directory) > modificationDate(rhs.directory)
            }
            .prefix(limit)
            .map(trainingDatasetPayload)

        return [
            "count": rows.count,
            "roots": roots.map { $0.path(percentEncoded: false) },
            "datasets": Array(rows)
        ]
    }

    func listTrainingRuns(arguments: [String: Any]) -> [String: Any] {
        let limit = max(1, min(integerValue(arguments["limit"]) ?? 25, 100))
        let includeMissingLogs = boolValue(arguments["includeAdaptersWithoutLogs"]) ?? false
        let adapters = AdapterRegistry.listAdapters()
        let rows = adapters.compactMap { adapter -> [String: Any]? in
            if let log = AdapterRegistry.loadTrainingRunLog(adapterID: adapter.id) {
                return trainingRunPayload(adapter: adapter, log: log)
            }
            guard includeMissingLogs else { return nil }
            return trainingRunPayload(adapter: adapter, log: nil)
        }
        .sorted { lhs, rhs in
            String(describing: lhs["updatedAt"] ?? "") > String(describing: rhs["updatedAt"] ?? "")
        }
        .prefix(limit)

        return [
            "count": rows.count,
            "runs": Array(rows)
        ]
    }

    func inspectTrainingDataset(arguments: [String: Any]) throws -> [String: Any] {
        let sampleLimit = max(0, min(integerValue(arguments["sampleLimit"]) ?? 12, 50))
        let directory = try resolveTrainingDatasetDirectory(arguments: arguments)
        let document = try TrainingDatasetPackageEditor.loadPackage(at: directory)
        return trainingDatasetDetailPayload(document, sampleLimit: sampleLimit)
    }

    func inspectTrainingRun(arguments: [String: Any]) throws -> [String: Any] {
        let progressLimit = max(0, min(integerValue(arguments["progressLimit"]) ?? 50, 500))
        let eventLimit = max(0, min(integerValue(arguments["eventLimit"]) ?? 25, 100))
        let adapter = try resolveTrainingRunAdapter(arguments: arguments)
        let log = AdapterRegistry.loadTrainingRunLog(adapterID: adapter.id)
        return trainingRunDetailPayload(
            adapter: adapter,
            log: log,
            progressLimit: progressLimit,
            eventLimit: eventLimit
        )
    }

    func compareTrainingRuns(arguments: [String: Any]) throws -> [String: Any] {
        let limit = max(2, min(integerValue(arguments["limit"]) ?? 6, 20))
        let includeMissingLogs = boolValue(arguments["includeAdaptersWithoutLogs"]) ?? false
        let adapters = try resolveTrainingRunComparisonAdapters(
            arguments: arguments,
            limit: limit,
            includeMissingLogs: includeMissingLogs
        )
        let rows = adapters.map { adapter in
            trainingRunComparisonPayload(
                adapter: adapter,
                log: AdapterRegistry.loadTrainingRunLog(adapterID: adapter.id)
            )
        }

        return [
            "count": rows.count,
            "runs": rows,
            "note": "Read-only comparison of installed adapter metadata and training_log.json metrics."
        ]
    }

    func tailTrainingLog(arguments: [String: Any]) throws -> [String: Any] {
        let limit = max(1, min(integerValue(arguments["limit"]) ?? 120, 500))
        let contains = stringValue(arguments["contains"] ?? arguments["filter"])?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? "training"
        let url = try resolveDiagnosticLogURL(arguments: arguments)
        let lines = try tailLines(
            from: url,
            limit: limit,
            containing: contains.isEmpty ? nil : contains
        )

        return [
            "path": url.path(percentEncoded: false),
            "filter": contains.isEmpty ? NSNull() : contains,
            "lineCount": lines.count,
            "lines": lines
        ]
    }

    func trainingPreflight(arguments: [String: Any]) async throws -> [String: Any] {
        guard let trainingService else {
            throw PiMPAudioCommandFacadeError.trainingServiceUnavailable
        }
        let request = try resolveMCPTrainingRequest(arguments: arguments)
        let clearCache = boolValue(arguments["clearCache"]) ?? true
        let report = await trainingService.localTrainingPreflight(
            kind: request.kind,
            sampleCount: request.importedFiles.count,
            config: request.config,
            lokrConfig: request.lokrConfig,
            clearCache: clearCache
        )
        return trainingPreflightResponsePayload(
            request: request,
            report: report,
            submitted: false,
            dryRun: true,
            confirmation: NSNull()
        )
    }

    func startTraining(arguments: [String: Any]) async throws -> [String: Any] {
        guard let trainingService else {
            throw PiMPAudioCommandFacadeError.trainingServiceUnavailable
        }

        let request = try resolveMCPTrainingRequest(arguments: arguments)
        let clearCache = boolValue(arguments["clearCache"]) ?? true
        let report = await trainingService.localTrainingPreflight(
            kind: request.kind,
            sampleCount: request.importedFiles.count,
            config: request.config,
            lokrConfig: request.lokrConfig,
            clearCache: clearCache
        )
        let wantsActualRun = boolValue(arguments["dryRun"] ?? arguments["validateOnly"]) == false
        guard wantsActualRun else {
            return trainingPreflightResponsePayload(
                request: request,
                report: report,
                submitted: false,
                dryRun: true,
                confirmation: confirmationRequirementPayload(toolName: "start_training")
            )
        }
        guard boolValue(arguments["confirm"]) == true else {
            throw PiMPAudioCommandFacadeError.confirmationRequired("start_training")
        }
        guard report.allowed else {
            return trainingPreflightResponsePayload(
                request: request,
                report: report,
                submitted: false,
                dryRun: false,
                confirmation: NSNull()
            )
        }
        guard !trainingStateIsActive(trainingService.state) else {
            throw PiMPAudioCommandFacadeError.trainingAlreadyActive(
                trainingStatePayload(trainingService.state)["status"].map(String.init(describing:)) ?? "active"
            )
        }

        switch request.kind {
        case .lora:
            trainingService.startTraining(
                entries: [],
                audioURLs: [],
                importedFiles: request.importedFiles,
                adapterName: request.adapterName,
                config: request.config
            )
        case .lokr:
            guard let lokrConfig = request.lokrConfig else {
                throw PiMPAudioCommandFacadeError.invalidTrainingConfiguration("LoKR training requires a LoKR config.")
            }
            trainingService.startLoKRTraining(
                entries: [],
                audioURLs: [],
                importedFiles: request.importedFiles,
                adapterName: request.adapterName,
                config: lokrConfig
            )
        }

        var payload = trainingPreflightResponsePayload(
            request: request,
            report: report,
            submitted: true,
            dryRun: false,
            confirmation: [
                "required": true,
                "accepted": true,
                "confirmArgument": "confirm"
            ]
        )
        payload["status"] = trainingStatus()
        payload["note"] = "Training was submitted through the native TrainingService. Poll training_status, tail_training_log, or inspect_training_run after the adapter appears."
        return payload
    }

    func stopTraining(arguments: [String: Any]) throws -> [String: Any] {
        guard boolValue(arguments["confirm"]) == true else {
            throw PiMPAudioCommandFacadeError.confirmationRequired("stop_training")
        }
        guard let trainingService else {
            throw PiMPAudioCommandFacadeError.trainingServiceUnavailable
        }

        let saveCheckpoint = boolValue(
            arguments["saveCheckpoint"] ?? arguments["checkpoint"] ?? arguments["save"]
        ) ?? trainingService.canSaveOnStop
        guard trainingStateIsActive(trainingService.state) else {
            return [
                "submitted": false,
                "stopped": false,
                "reason": "trainingNotActive",
                "status": trainingStatus()
            ]
        }

        trainingService.stopTraining(saveCheckpoint: saveCheckpoint)
        return [
            "submitted": true,
            "stopped": true,
            "saveCheckpoint": saveCheckpoint,
            "status": trainingStatus()
        ]
    }

    func runPodStatus(arguments: [String: Any]) async throws -> [String: Any] {
        guard let runPodService else {
            throw PiMPAudioCommandFacadeError.runPodServiceUnavailable
        }

        if boolValue(arguments["refreshLaunchOptions"] ?? arguments["refreshOptions"]) == true {
            await runPodService.refreshLaunchOptions()
        }
        if boolValue(arguments["refreshPods"]) == true {
            await runPodService.refreshPods()
        }

        return runPodStatusPayload(
            service: runPodService,
            recommendedForGPUType: stringValue(arguments["gpuType"] ?? arguments["gpuTypeID"] ?? arguments["gpu"])
        )
    }

    func refreshRunPodLaunchOptions(arguments: [String: Any]) async throws -> [String: Any] {
        guard let runPodService else {
            throw PiMPAudioCommandFacadeError.runPodServiceUnavailable
        }

        await runPodService.refreshLaunchOptions()
        var payload = runPodStatusPayload(
            service: runPodService,
            recommendedForGPUType: stringValue(arguments["gpuType"] ?? arguments["gpuTypeID"] ?? arguments["gpu"])
        )
        payload["refreshed"] = runPodService.lastError == nil
        return payload
    }

    func launchRunPodPod(arguments: [String: Any]) async throws -> [String: Any] {
        guard let runPodService else {
            throw PiMPAudioCommandFacadeError.runPodServiceUnavailable
        }

        let dryRun = boolValue(arguments["dryRun"] ?? arguments["validateOnly"]) ?? true
        let resolution = try await resolveRunPodLaunchConfiguration(
            arguments: arguments,
            service: runPodService
        )
        _ = try resolution.configuration.requestBody()

        var payload = runPodLaunchPayload(
            configuration: resolution.configuration,
            recommendedTemplate: resolution.recommendedTemplate,
            usedRecommendedTemplate: resolution.usedRecommendedTemplate,
            submitted: false,
            dryRun: dryRun,
            pod: nil,
            error: nil
        )

        guard !dryRun else {
            payload["confirmation"] = confirmationRequirementPayload(toolName: "launch_runpod_pod")
            payload["status"] = runPodStatusPayload(service: runPodService, recommendedForGPUType: resolution.configuration.gpuTypeID)
            return payload
        }
        guard boolValue(arguments["confirm"]) == true else {
            throw PiMPAudioCommandFacadeError.confirmationRequired("launch_runpod_pod")
        }

        guard runPodService.hasAPIKey else {
            let message = "Add a RunPod API key before launching a pod."
            return runPodLaunchPayload(
                configuration: resolution.configuration,
                recommendedTemplate: resolution.recommendedTemplate,
                usedRecommendedTemplate: resolution.usedRecommendedTemplate,
                submitted: false,
                dryRun: false,
                pod: nil,
                error: message
            )
        }

        let pod = await runPodService.launchPod(configuration: resolution.configuration)
        payload = runPodLaunchPayload(
            configuration: resolution.configuration,
            recommendedTemplate: resolution.recommendedTemplate,
            usedRecommendedTemplate: resolution.usedRecommendedTemplate,
            submitted: pod != nil,
            dryRun: false,
            pod: pod,
            error: pod == nil ? runPodService.lastError : nil
        )
        payload["status"] = runPodStatusPayload(service: runPodService, recommendedForGPUType: resolution.configuration.gpuTypeID)
        return payload
    }

    func stopRunPodPod(arguments: [String: Any]) async throws -> [String: Any] {
        guard boolValue(arguments["confirm"]) == true else {
            throw PiMPAudioCommandFacadeError.confirmationRequired("stop_runpod_pod")
        }
        guard let runPodService else {
            throw PiMPAudioCommandFacadeError.runPodServiceUnavailable
        }

        let podID = try requiredString("podID", aliases: ["id", "pod"], in: arguments)
        await runPodService.terminatePod(id: podID)
        return [
            "submitted": runPodService.lastError == nil,
            "podID": podID,
            "action": "terminate",
            "error": runPodService.lastError ?? NSNull(),
            "status": runPodStatusPayload(service: runPodService, recommendedForGPUType: nil)
        ]
    }


    func trainingStatePayload(_ state: TrainingState) -> [String: Any] {
        switch state {
        case .idle:
            return ["status": "idle", "active": false]
        case .loadingModels:
            return ["status": "loadingModels", "active": true]
        case .preparingDataset(let current, let total):
            return [
                "status": "preparingDataset",
                "active": true,
                "current": current,
                "total": total,
                "progress": progressFraction(current: current, total: total)
            ]
        case .training(let uiState):
            return [
                "status": "training",
                "active": true,
                "progress": trainingUIStatePayload(uiState)
            ]
        case .saving:
            return ["status": "saving", "active": true]
        case .complete(let adapter):
            return [
                "status": "complete",
                "active": false,
                "adapter": trainingAdapterSummaryPayload(adapter)
            ]
        case .failed(let message):
            return [
                "status": "failed",
                "active": false,
                "message": message
            ]
        case .cancelled:
            return ["status": "cancelled", "active": false]
        case .paused(let checkpoint):
            return [
                "status": "paused",
                "active": false,
                "checkpoint": trainingCheckpointPayload(checkpoint)
            ]
        }
    }

    func trainingUIStatePayload(_ uiState: TrainingUIState) -> [String: Any] {
        [
            "epoch": uiState.epoch,
            "totalEpochs": uiState.totalEpochs,
            "step": uiState.step,
            "loss": Double(uiState.loss),
            "learningRate": Double(uiState.learningRate),
            "lossMovingAverage": uiState.lossMovingAverage.map { Double($0) } ?? NSNull(),
            "gradientNorm": uiState.gradientNorm.map { Double($0) } ?? NSNull(),
            "parameterNorm": uiState.parameterNorm.map { Double($0) } ?? NSNull(),
            "updateNorm": uiState.updateNorm.map { Double($0) } ?? NSNull(),
            "validationLoss": uiState.validationLoss.map { Double($0) } ?? NSNull(),
            "elapsedSeconds": uiState.elapsed,
            "remainingSeconds": uiState.remaining.map { $0 as Any } ?? NSNull(),
            "lossHistoryCount": uiState.lossHistory.count,
            "latestLossHistory": Array(uiState.lossHistory.suffix(25)).map(Double.init),
            "validationHistoryCount": uiState.validationHistory.count,
            "latestValidationHistory": uiState.validationHistory.suffix(25).map(trainingValidationPointPayload)
        ]
    }

    func trainingCheckpointPayload(_ checkpoint: TrainingCheckpoint) -> [String: Any] {
        [
            "epoch": checkpoint.epoch,
            "globalStep": checkpoint.globalStep,
            "totalSteps": checkpoint.totalSteps,
            "bestLoss": checkpoint.bestLoss.map { Double($0) } ?? NSNull(),
            "lossHistoryCount": checkpoint.lossHistory.count,
            "validationHistoryCount": checkpoint.validationHistory.count,
            "weightsPath": checkpoint.weightsURL.path(percentEncoded: false),
            "optimizerStatePath": checkpoint.optimizerStateURL.path(percentEncoded: false),
            "baseModelID": checkpoint.baseModelID
        ]
    }

    func trainingAdapterSummaryPayload(_ adapter: LoRAAdapter) -> [String: Any] {
        let directory = try? AdapterRegistry.adapterDirectory(id: adapter.id)
        return [
            "id": adapter.id,
            "name": adapter.name,
            "kind": adapter.kind.rawValue,
            "baseModelID": adapter.baseModelID,
            "baseModelCategory": adapter.baseModelCategory.rawValue,
            "rank": adapter.config.rank,
            "alpha": Double(adapter.config.alpha),
            "sampleCount": adapter.sampleCount,
            "finalLoss": adapter.finalLoss.map { Double($0) } ?? NSNull(),
            "createdAt": dateFormatter.string(from: adapter.createdAt),
            "hasCheckpoint": AdapterRegistry.hasCheckpoint(adapterID: adapter.id),
            "directory": directory?.path(percentEncoded: false) ?? NSNull()
        ]
    }

    func trainingDatasetSearchRoots() -> [URL] {
        let roots = [
            historyStore.storageDirectory,
            AppStorageLocations.rootDirectory
        ]
        var seen = Set<String>()
        return roots.filter { url in
            let path = url.standardizedFileURL.path(percentEncoded: false)
            return seen.insert(path).inserted
        }
    }

    func trainingDatasetPayload(_ summary: DatasetPackageSummary) -> [String: Any] {
        [
            "id": summary.id,
            "name": summary.name,
            "path": summary.directory.path(percentEncoded: false),
            "createdAt": summary.createdAt ?? NSNull(),
            "modifiedAt": dateFormatter.string(from: modificationDate(summary.directory)),
            "clipCount": summary.clipCount,
            "isValid": summary.isValid,
            "validationSummary": summary.validationSummary,
            "enrichmentStatus": summary.enrichmentStatus,
            "transcriptionStatus": summary.transcriptionStatus,
            "bpmStatus": summary.bpmStatus,
            "keyStatus": summary.keyStatus,
            "timeSignatureStatus": summary.timeSignatureStatus,
            "trainingBundlePath": summary.directory
                .appendingPathComponent(TrainingDatasetPackageWriter.fullMixBundleFileName)
                .path(percentEncoded: false),
            "remotePackagePath": summary.directory
                .appendingPathComponent(TrainingDatasetPackageWriter.remotePackageFileName)
                .path(percentEncoded: false)
        ]
    }

    func resolveMCPTrainingRequest(arguments: [String: Any]) throws -> MCPTrainingRequest {
        let kind = try mcpTrainingKind(from: arguments["kind"] ?? arguments["adapterKind"] ?? arguments["mode"])
        let directory = try resolveTrainingDatasetDirectory(arguments: mcpTrainingDatasetArguments(arguments))
        let document = try TrainingDatasetPackageEditor.loadPackage(at: directory)
        let importedFiles = try TrainingDatasetPackageEditor.importedFiles(for: document)
        let profile = trainingDatasetProfile(for: document)
        let recommendation = TrainingPlanRecommendation.make(adapterKind: kind, profile: profile)
        let recipe = try mcpTrainingRecipe(from: arguments["recipe"] ?? arguments["preset"], kind: kind)
        let defaults = mcpTrainingDefaults(
            recipe: recipe,
            kind: kind,
            recommendation: recommendation
        )

        let dimension = try boundedTrainingInt(
            arguments["rank"] ?? arguments["linearDim"] ?? arguments["dimension"],
            defaultValue: defaults.dimension,
            name: kind == .lokr ? "linearDim" : "rank",
            range: 1...256
        )
        let epochs = try boundedTrainingInt(
            arguments["epochs"] ?? arguments["maxEpochs"],
            defaultValue: defaults.epochs,
            name: "epochs",
            range: 1...1_000
        )
        let layerCount = try boundedTrainingLayerCount(
            arguments["layerCoverage"] ?? arguments["trainingLayerCount"] ?? arguments["layerCount"],
            defaultValue: defaults.layerCount
        )
        let gradientAccumulation = try boundedTrainingInt(
            arguments["gradientAccumulationSteps"] ?? arguments["gradAccumulation"] ?? arguments["gradAccum"],
            defaultValue: defaults.gradientAccumulationSteps,
            name: "gradientAccumulationSteps",
            range: 1...64
        )
        let batchSize = try boundedTrainingInt(
            arguments["batchSize"],
            defaultValue: 1,
            name: "batchSize",
            range: 1...16
        )
        let learningRate = try boundedTrainingFloat(
            arguments["learningRate"] ?? arguments["lr"],
            defaultValue: defaults.learningRate,
            name: "learningRate",
            range: 0.000_001...0.01
        )
        let dropout = try boundedTrainingFloat(
            arguments["dropout"],
            defaultValue: defaults.dropout,
            name: "dropout",
            range: 0...0.95
        )
        let weightDecompose = try mcpTrainingBool(
            from: arguments["weightDecompose"] ?? arguments["weight_decompose"] ?? arguments["dora"],
            defaultValue: kind == .lokr ? defaults.weightDecompose : false,
            name: "weightDecompose"
        )
        let warmupFraction = try boundedTrainingFloat(
            arguments["warmupFraction"] ?? arguments["warmup"],
            defaultValue: 0.10,
            name: "warmupFraction",
            range: 0...0.50
        )
        let alpha = try mcpTrainingAlpha(
            from: arguments["alpha"],
            alphaScale: arguments["alphaScale"] ?? arguments["scale"],
            dimension: dimension,
            defaultScale: defaults.alphaScale
        )
        let targetLayers = try mcpTrainingTargetLayers(
            from: arguments["targetLayers"] ?? arguments["targets"],
            defaultValue: defaults.targetLayers
        )
        let exactInjectionPatterns = try mcpTrainingInjectionPatterns(
            arguments: arguments,
            targetLayers: targetLayers,
            defaultCoverage: defaults.projectionCoverage
        )
        let adapterName = mcpTrainingAdapterName(
            arguments: arguments,
            document: document,
            kind: kind
        )

        let config = LoRAConfig(
            rank: dimension,
            alpha: alpha,
            dropout: dropout,
            targetLayers: targetLayers,
            exactInjectionPatterns: exactInjectionPatterns,
            learningRate: learningRate,
            maxEpochs: epochs,
            batchSize: batchSize,
            warmupFraction: warmupFraction,
            gradientAccumulationSteps: gradientAccumulation,
            trainingLayerCount: layerCount
        )
        let lokrConfig = kind == .lokr
            ? LoKRConfig(
                linearDim: dimension,
                alpha: alpha,
                dropout: dropout,
                targetLayers: targetLayers,
                exactInjectionPatterns: exactInjectionPatterns,
                learningRate: learningRate,
                maxEpochs: epochs,
                batchSize: batchSize,
                warmupFraction: warmupFraction,
                gradientAccumulationSteps: gradientAccumulation,
                trainingLayerCount: layerCount,
                weightDecompose: weightDecompose
            )
            : nil

        return MCPTrainingRequest(
            kind: kind,
            recipe: recipe,
            dataset: document,
            importedFiles: importedFiles,
            adapterName: adapterName,
            config: config,
            lokrConfig: lokrConfig
        )
    }

    func mcpTrainingDatasetArguments(_ arguments: [String: Any]) -> [String: Any] {
        guard arguments["dataset"] == nil,
              arguments["id"] == nil,
              arguments["path"] == nil,
              arguments["directory"] == nil,
              arguments["datasetPath"] == nil else {
            return arguments
        }

        var datasetArguments = arguments
        datasetArguments["name"] = nil
        return datasetArguments
    }

    func mcpTrainingKind(from value: Any?) throws -> AdapterKind {
        guard let raw = stringValue(value)?.trimmingCharacters(in: .whitespacesAndNewlines),
              !raw.isEmpty else {
            return .lora
        }
        switch compactTrainingToken(raw) {
        case "lora":
            return .lora
        case "lokr":
            return .lokr
        default:
            throw PiMPAudioCommandFacadeError.invalidTrainingKind(raw)
        }
    }

    func mcpTrainingRecipe(from value: Any?, kind: AdapterKind) throws -> MCPTrainingRecipe {
        guard let raw = stringValue(value)?.trimmingCharacters(in: .whitespacesAndNewlines),
              !raw.isEmpty else {
            return .narrow
        }

        switch compactTrainingToken(raw) {
        case "narrow", "local", "balanced", "balancedlocal":
            return .narrow
        case "wide", "expanded", "expandedlokr":
            return .wide
        default:
            throw PiMPAudioCommandFacadeError.invalidTrainingRecipe(raw)
        }
    }

    func mcpTrainingDefaults(
        recipe: MCPTrainingRecipe,
        kind: AdapterKind,
        recommendation: TrainingPlanRecommendation?
    ) -> MCPTrainingDefaults {
        switch recipe {
        case .narrow:
            return MCPTrainingDefaults(
                dimension: recommendation?.rank ?? 16,
                epochs: recommendation?.epochs ?? 10,
                layerCount: recommendation?.layerCount ?? 8,
                projectionCoverage: .queryValue,
                alphaScale: 1,
                dropout: 0,
                targetLayers: [.selfAttention, .crossAttention],
                gradientAccumulationSteps: recommendation?.gradientAccumulationSteps ?? 1,
                learningRate: 0.0001,
                weightDecompose: kind == .lokr
            )
        case .wide:
            return MCPTrainingDefaults(
                dimension: kind == .lokr ? 32 : 96,
                epochs: kind == .lokr ? 100 : 150,
                layerCount: kind == .lokr ? 16 : 32,
                projectionCoverage: kind == .lokr ? .queryValue : .fullAttention,
                alphaScale: 2,
                dropout: 0.08,
                targetLayers: [.selfAttention, .crossAttention],
                gradientAccumulationSteps: 4,
                learningRate: kind == .lokr ? 0.0001 : 0.000075,
                weightDecompose: kind == .lokr
            )
        }
    }

    func mcpTrainingBool(
        from value: Any?,
        defaultValue: Bool,
        name: String
    ) throws -> Bool {
        guard value != nil else { return defaultValue }
        guard let parsed = boolValue(value) else {
            throw PiMPAudioCommandFacadeError.invalidBoolean(name)
        }
        return parsed
    }

    func boundedTrainingInt(
        _ value: Any?,
        defaultValue: Int,
        name: String,
        range: ClosedRange<Int>
    ) throws -> Int {
        let resolved = integerValue(value) ?? defaultValue
        guard range.contains(resolved) else {
            throw PiMPAudioCommandFacadeError.invalidTrainingConfiguration(
                "\(name) must be between \(range.lowerBound) and \(range.upperBound)."
            )
        }
        return resolved
    }

    func boundedTrainingLayerCount(_ value: Any?, defaultValue: Int) throws -> Int {
        if let raw = stringValue(value)?.trimmingCharacters(in: .whitespacesAndNewlines),
           !raw.isEmpty,
           integerValue(raw) == nil {
            switch compactTrainingToken(raw) {
            case "last4":
                return 4
            case "last8":
                return 8
            case "last12":
                return 12
            case "last16":
                return 16
            case "all32", "all", "full":
                return 32
            default:
                throw PiMPAudioCommandFacadeError.invalidTrainingConfiguration(
                    "layerCoverage must be a number from 1 to 32, or one of last4, last8, last12, last16, all32."
                )
            }
        }
        return try boundedTrainingInt(value, defaultValue: defaultValue, name: "layerCoverage", range: 1...32)
    }

    func boundedTrainingFloat(
        _ value: Any?,
        defaultValue: Float,
        name: String,
        range: ClosedRange<Float>
    ) throws -> Float {
        let resolved = doubleValue(value).map(Float.init) ?? defaultValue
        guard range.contains(resolved) else {
            throw PiMPAudioCommandFacadeError.invalidTrainingConfiguration(
                "\(name) must be between \(range.lowerBound) and \(range.upperBound)."
            )
        }
        return resolved
    }

    func mcpTrainingAlpha(
        from value: Any?,
        alphaScale: Any?,
        dimension: Int,
        defaultScale: Float
    ) throws -> Float {
        if let explicit = doubleValue(value).map(Float.init) {
            guard explicit > 0 else {
                throw PiMPAudioCommandFacadeError.invalidTrainingConfiguration("alpha must be greater than 0.")
            }
            return explicit
        }

        let scale: Float
        if let raw = stringValue(alphaScale)?.trimmingCharacters(in: .whitespacesAndNewlines),
           !raw.isEmpty {
            switch compactTrainingToken(raw) {
            case "rank", "dim", "dimension", "1", "1x":
                scale = 1
            case "double", "doublerank", "doubledim", "2", "2x":
                scale = 2
            default:
                guard let numeric = doubleValue(raw), numeric > 0 else {
                    throw PiMPAudioCommandFacadeError.invalidTrainingConfiguration(
                        "alphaScale must be a positive number, 1x/rank, or 2x/double."
                    )
                }
                scale = Float(numeric)
            }
        } else if let numeric = doubleValue(alphaScale) {
            guard numeric > 0 else {
                throw PiMPAudioCommandFacadeError.invalidTrainingConfiguration("alphaScale must be greater than 0.")
            }
            scale = Float(numeric)
        } else {
            scale = defaultScale
        }
        return Float(dimension) * scale
    }

    func mcpTrainingTargetLayers(
        from value: Any?,
        defaultValue: Set<TargetLayer>
    ) throws -> Set<TargetLayer> {
        guard let rawLayers = stringArrayValue(value), !rawLayers.isEmpty else {
            return defaultValue
        }

        var layers = Set<TargetLayer>()
        for raw in rawLayers {
            switch compactTrainingToken(raw) {
            case "self", "selfattn", "selfattention":
                layers.insert(.selfAttention)
            case "cross", "crossattn", "crossattention":
                layers.insert(.crossAttention)
            case "mlp", "feedforward", "ffn":
                layers.insert(.mlp)
            case "attention", "attn", "both":
                layers.formUnion([.selfAttention, .crossAttention])
            case "all":
                layers.formUnion(TargetLayer.allCases)
            default:
                throw PiMPAudioCommandFacadeError.invalidTrainingConfiguration(
                    "Unsupported target layer '\(raw)'. Use selfAttention, crossAttention, mlp, attention, or all."
                )
            }
        }
        guard !layers.isEmpty else {
            throw PiMPAudioCommandFacadeError.invalidTrainingConfiguration("targetLayers cannot be empty.")
        }
        return layers
    }

    func mcpTrainingInjectionPatterns(
        arguments: [String: Any],
        targetLayers: Set<TargetLayer>,
        defaultCoverage: MCPTrainingProjectionCoverage
    ) throws -> [String]? {
        if let patterns = stringArrayValue(arguments["exactInjectionPatterns"] ?? arguments["injectionPatterns"]),
           !patterns.isEmpty {
            return patterns
        }

        let coverage: MCPTrainingProjectionCoverage
        if let raw = stringValue(
            arguments["projectionCoverage"] ?? arguments["projections"] ?? arguments["projection"]
        )?.trimmingCharacters(in: .whitespacesAndNewlines),
           !raw.isEmpty {
            switch compactTrainingToken(raw) {
            case "qv", "queryvalue", "queryvalueonly":
                coverage = .queryValue
            case "qkvo", "full", "fullattention", "all":
                coverage = .fullAttention
            default:
                throw PiMPAudioCommandFacadeError.invalidTrainingConfiguration(
                    "projectionCoverage must be qv or qkvo."
                )
            }
        } else {
            coverage = defaultCoverage
        }

        switch coverage {
        case .queryValue:
            return nil
        case .fullAttention:
            var patterns: [String] = []
            if targetLayers.contains(.selfAttention) {
                patterns.append(contentsOf: [
                    "selfAttn.qProj",
                    "selfAttn.kProj",
                    "selfAttn.vProj",
                    "selfAttn.oProj",
                ])
            }
            if targetLayers.contains(.crossAttention) {
                patterns.append(contentsOf: [
                    "crossAttn.qProj",
                    "crossAttn.kProj",
                    "crossAttn.vProj",
                    "crossAttn.oProj",
                ])
            }
            if targetLayers.contains(.mlp) {
                patterns.append(contentsOf: TargetLayer.mlp.defaultPatterns)
            }
            return patterns.isEmpty ? nil : patterns
        }
    }

    func mcpTrainingAdapterName(
        arguments: [String: Any],
        document: DatasetPackageDocument,
        kind: AdapterKind
    ) -> String {
        if let explicit = stringValue(arguments["adapterName"] ?? arguments["name"])?
            .trimmingCharacters(in: .whitespacesAndNewlines),
           !explicit.isEmpty {
            return explicit
        }
        let baseName = document.name
            .replacingOccurrences(of: "_local_", with: " ")
            .replacingOccurrences(of: "_", with: " ")
        return "\(baseName) \(mcpTrainingKindTitle(kind))"
    }

    func trainingPreflightResponsePayload(
        request: MCPTrainingRequest,
        report: TrainingLocalPreflightReport,
        submitted: Bool,
        dryRun: Bool,
        confirmation: Any
    ) -> [String: Any] {
        let lokrPayload: Any = request.lokrConfig.map(lokrConfigPayload) ?? NSNull()
        let effectiveInjectionPatterns = request.lokrConfig?.injectionPatterns ?? request.config.injectionPatterns
        let blockedReason: Any = report.blockedReason.map { $0 as Any } ?? NSNull()
        return [
            "submitted": submitted,
            "dryRun": dryRun,
            "allowed": report.allowed,
            "blockedReason": blockedReason,
            "adapterName": request.adapterName,
            "kind": request.kind.rawValue,
            "recipe": request.recipe.rawValue,
            "dataset": [
                "summary": trainingDatasetPayload(TrainingDatasetPackageEditor.packageSummary(at: request.dataset.directory)),
                "clipCount": request.dataset.clips.count,
                "importedFileCount": request.importedFiles.count
            ],
            "configuration": [
                "lora": loraConfigPayload(request.config),
                "lokr": lokrPayload,
                "effectiveInjectionPatterns": effectiveInjectionPatterns
            ],
            "preflight": trainingLocalPreflightPayload(report),
            "confirmation": confirmation
        ]
    }

    func trainingLocalPreflightPayload(_ report: TrainingLocalPreflightReport) -> [String: Any] {
        let blockedReason: Any = report.blockedReason.map { $0 as Any } ?? NSNull()
        let unsupportedConfigurationReason: Any = report.unsupportedConfigurationReason.map { $0 as Any } ?? NSNull()
        let memoryHeadroomReason: Any = report.memoryHeadroomReason.map { $0 as Any } ?? NSNull()
        let systemAvailableBytes: Any = report.systemAvailableMemoryBytes ?? NSNull()
        let systemAvailable: Any = report.systemAvailableMemoryBytes.map { formatGiB($0) } ?? NSNull()
        return [
            "phase": report.phase,
            "allowed": report.allowed,
            "blockedReason": blockedReason,
            "unsupportedConfigurationReason": unsupportedConfigurationReason,
            "memoryHeadroomReason": memoryHeadroomReason,
            "kind": report.kind.rawValue,
            "activeModelID": report.activeModelID,
            "sampleCount": report.sampleCount,
            "dimension": report.dimension,
            "layerCount": report.layerCount,
            "projectionCount": report.projectionCount,
            "injectionPatterns": report.injectionPatterns,
            "estimatedTrainablePathCount": report.estimatedTrainablePathCount,
            "estimatedTrainableTensorCount": report.estimatedTrainableTensorCount,
            "memory": [
                "cacheBeforeClearBytes": report.cacheBeforeClearBytes,
                "cacheBeforeClear": formatGiB(report.cacheBeforeClearBytes),
                "activeBytes": report.activeMemoryBytes,
                "active": formatGiB(report.activeMemoryBytes),
                "cacheBytes": report.cacheMemoryBytes,
                "cache": formatGiB(report.cacheMemoryBytes),
                "peakBytes": report.peakMemoryBytes,
                "peak": formatGiB(report.peakMemoryBytes),
                "physicalBytes": report.physicalMemoryBytes,
                "physical": formatGiB(report.physicalMemoryBytes),
                "systemAvailableBytes": systemAvailableBytes,
                "systemAvailable": systemAvailable,
                "memoryLimitBytes": report.memoryLimitBytes,
                "memoryLimit": formatGiB(report.memoryLimitBytes),
                "processHeadroomBytes": report.processHeadroomBytes,
                "processHeadroom": formatGiB(report.processHeadroomBytes),
                "effectiveHeadroomBytes": report.effectiveHeadroomBytes,
                "effectiveHeadroom": formatGiB(report.effectiveHeadroomBytes),
                "requiredHeadroomBytes": report.requiredHeadroomBytes,
                "requiredHeadroom": formatGiB(report.requiredHeadroomBytes)
            ],
            "warnings": report.warnings,
            "note": "Preflight uses the current MLX memory snapshot. Training rechecks after ACE-Step model load and again before the first trainer pass."
        ]
    }

    func trainingStateIsActive(_ state: TrainingState) -> Bool {
        switch state {
        case .loadingModels, .preparingDataset, .training, .saving:
            return true
        case .idle, .complete, .failed, .cancelled, .paused:
            return false
        }
    }

    func mcpTrainingKindTitle(_ kind: AdapterKind) -> String {
        switch kind {
        case .lora:
            return "LoRA"
        case .lokr:
            return "LoKR"
        }
    }

    func compactTrainingToken(_ raw: String) -> String {
        raw
            .lowercased()
            .replacingOccurrences(of: " ", with: "")
            .replacingOccurrences(of: "-", with: "")
            .replacingOccurrences(of: "_", with: "")
            .replacingOccurrences(of: "/", with: "")
    }

    func resolveTrainingDatasetDirectory(arguments: [String: Any]) throws -> URL {
        if let rawPath = stringValue(arguments["path"] ?? arguments["directory"] ?? arguments["datasetPath"])?
            .trimmingCharacters(in: .whitespacesAndNewlines),
           !rawPath.isEmpty {
            let expanded = (rawPath as NSString).expandingTildeInPath
            let url = URL(fileURLWithPath: expanded).standardizedFileURL
            guard FileManager.default.fileExists(
                atPath: url.appendingPathComponent(TrainingDatasetPackageWriter.captionsFileName)
                    .path(percentEncoded: false)
            ) else {
                throw PiMPAudioCommandFacadeError.resourceNotFound(rawPath)
            }
            return url
        }

        let packages = trainingDatasetSearchRoots()
            .flatMap { TrainingDatasetPackageEditor.listPackages(in: $0) }
            .sorted { lhs, rhs in
                modificationDate(lhs.directory) > modificationDate(rhs.directory)
            }
        guard !packages.isEmpty else {
            throw PiMPAudioCommandFacadeError.resourceNotFound("training datasets")
        }

        guard let identifier = stringValue(arguments["id"] ?? arguments["name"] ?? arguments["dataset"])?
            .trimmingCharacters(in: .whitespacesAndNewlines),
              !identifier.isEmpty,
              identifier.lowercased() != "latest" else {
            return packages[0].directory
        }

        let needle = identifier.lowercased()
        if let exact = packages.first(where: { package in
            package.id.lowercased() == needle ||
                package.name.lowercased() == needle ||
                package.directory.lastPathComponent.lowercased() == needle ||
                package.directory.path(percentEncoded: false).lowercased() == needle
        }) {
            return exact.directory
        }
        if let partial = packages.first(where: { package in
            package.name.lowercased().contains(needle) ||
                package.directory.lastPathComponent.lowercased().contains(needle)
        }) {
            return partial.directory
        }
        throw PiMPAudioCommandFacadeError.resourceNotFound(identifier)
    }

    func trainingDatasetDetailPayload(
        _ document: DatasetPackageDocument,
        sampleLimit: Int
    ) -> [String: Any] {
        let profile = trainingDatasetProfile(for: document)
        let recommendations: [String: Any] = [
            "lora": TrainingPlanRecommendation.make(adapterKind: .lora, profile: profile)
                .map(trainingPlanRecommendationPayload) ?? NSNull(),
            "lokr": TrainingPlanRecommendation.make(adapterKind: .lokr, profile: profile)
                .map(trainingPlanRecommendationPayload) ?? NSNull()
        ]
        let clips = document.clips.prefix(sampleLimit).map(trainingDatasetClipPayload)

        return [
            "summary": trainingDatasetPayload(TrainingDatasetPackageEditor.packageSummary(at: document.directory)),
            "validation": trainingDatasetValidationPayload(document.validation),
            "profile": trainingDatasetProfilePayload(profile),
            "annotations": trainingDatasetAnnotationStatsPayload(document),
            "recommendations": recommendations,
            "sampleClipLimit": sampleLimit,
            "sampleClips": Array(clips),
            "paths": trainingDatasetPathsPayload(document.directory),
            "resources": [
                "collection": "pimpaudio://training-datasets",
                "latest": "pimpaudio://training-dataset/latest"
            ]
        ]
    }

    func trainingDatasetValidationPayload(_ validation: TrainingDatasetPackageValidationResult) -> [String: Any] {
        [
            "isValid": validation.isValid,
            "summary": validation.summary,
            "checkedAt": validation.checkedAt,
            "clipCount": validation.clipCount,
            "importableClipCount": validation.importableClipCount,
            "missingPackageFiles": validation.missingPackageFiles,
            "missingClipFiles": validation.missingClipFiles,
            "errors": validation.errors
        ]
    }

    func trainingDatasetProfile(for document: DatasetPackageDocument) -> TrainingDatasetProfile {
        var knownDuration: TimeInterval = 0
        var knownDurationSampleCount = 0
        for clip in document.clips {
            guard let duration = clip.durationSeconds, duration > 0 else { continue }
            knownDuration += duration
            knownDurationSampleCount += 1
        }
        return TrainingDatasetProfile(
            sampleCount: document.clips.count,
            knownDuration: knownDuration,
            knownDurationSampleCount: knownDurationSampleCount
        )
    }

    func trainingDatasetProfilePayload(_ profile: TrainingDatasetProfile) -> [String: Any] {
        [
            "sampleCount": profile.sampleCount,
            "knownDurationSeconds": profile.knownDuration,
            "knownDurationMinutes": profile.knownDuration / 60,
            "knownDurationSampleCount": profile.knownDurationSampleCount,
            "durationCoverage": profile.durationCoverage,
            "hasUsableDurationCoverage": profile.hasUsableDurationCoverage
        ]
    }

    func trainingDatasetAnnotationStatsPayload(_ document: DatasetPackageDocument) -> [String: Any] {
        var captionCount = 0
        var lyricsCount = 0
        var durationCount = 0
        var bpmCount = 0
        var keyCount = 0
        var timeSignatureCount = 0
        var approvedCount = 0
        var captionSources: [String: Int] = [:]
        var lyricsSources: [String: Int] = [:]
        var transcriptionStatuses: [String: Int] = [:]
        var audioStatuses: [String: Int] = [:]

        for clip in document.clips {
            if !clip.caption.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { captionCount += 1 }
            if !clip.lyrics.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { lyricsCount += 1 }
            if let duration = clip.durationSeconds, duration > 0 { durationCount += 1 }
            if !clip.bpm.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { bpmCount += 1 }
            if !clip.key.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { keyCount += 1 }
            if !clip.timeSignature.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                timeSignatureCount += 1
            }
            if clip.isApproved { approvedCount += 1 }
            increment(&captionSources, key: clip.captionSource.isEmpty ? "unknown" : clip.captionSource)
            increment(&lyricsSources, key: clip.lyricsSource.isEmpty ? "unknown" : clip.lyricsSource)
            increment(&transcriptionStatuses, key: clip.transcriptionStatus.isEmpty ? "unknown" : clip.transcriptionStatus)
            increment(&audioStatuses, key: clip.audioStatus.isEmpty ? "unknown" : clip.audioStatus)
        }

        let total = document.clips.count
        return [
            "clipCount": total,
            "approvedCount": approvedCount,
            "blockerCount": document.blockerCount,
            "warningCount": document.warningCount,
            "captionCount": captionCount,
            "missingCaptionCount": max(0, total - captionCount),
            "lyricsCount": lyricsCount,
            "missingLyricsCount": max(0, total - lyricsCount),
            "durationKnownCount": durationCount,
            "bpmKnownCount": bpmCount,
            "keyKnownCount": keyCount,
            "timeSignatureKnownCount": timeSignatureCount,
            "captionSources": captionSources,
            "lyricsSources": lyricsSources,
            "transcriptionStatuses": transcriptionStatuses,
            "audioStatuses": audioStatuses
        ]
    }

    func trainingDatasetClipPayload(_ clip: DatasetClipReview) -> [String: Any] {
        [
            "id": clip.id,
            "file": clip.file,
            "displayName": clip.displayName,
            "caption": compact(clip.caption, limit: 220),
            "lyricsLineCount": lyricLineCount(clip.lyrics),
            "durationSeconds": clip.durationSeconds ?? NSNull(),
            "bpm": clip.bpm.isEmpty ? NSNull() : clip.bpm,
            "key": clip.key.isEmpty ? NSNull() : clip.key,
            "timeSignature": clip.timeSignature.isEmpty
                ? NSNull()
                : clip.timeSignature,
            "language": clip.language.isEmpty ? NSNull() : clip.language,
            "role": clip.role.isEmpty ? NSNull() : clip.role,
            "reviewStatus": clip.reviewStatus,
            "isApproved": clip.isApproved,
            "captionSource": clip.captionSource,
            "lyricsSource": clip.lyricsSource,
            "audioStatus": clip.audioStatus,
            "separationStatus": clip.separationStatus,
            "transcriptionStatus": clip.transcriptionStatus
        ]
    }

    func trainingDatasetPathsPayload(_ directory: URL) -> [String: Any] {
        let files = [
            TrainingDatasetPackageWriter.captionsFileName,
            TrainingDatasetPackageWriter.datasetManifestFileName,
            TrainingDatasetPackageWriter.validationReportFileName,
            TrainingDatasetPackageWriter.fullMixBundleFileName,
            TrainingDatasetPackageWriter.vocalBundleFileName,
            TrainingDatasetPackageWriter.instrumentalBundleFileName,
            TrainingDatasetPackageWriter.remotePackageFileName,
            TrainingDatasetPackageWriter.preprocessCacheFileName
        ]
        let entries = files.reduce(into: [String: Any]()) { result, fileName in
            let url = directory.appendingPathComponent(fileName)
            result[fileName] = [
                "path": url.path(percentEncoded: false),
                "exists": FileManager.default.fileExists(atPath: url.path(percentEncoded: false))
            ]
        }
        return [
            "directory": directory.path(percentEncoded: false),
            "files": entries
        ]
    }

    func trainingPlanRecommendationPayload(_ recommendation: TrainingPlanRecommendation) -> [String: Any] {
        [
            "adapterKind": recommendation.adapterKind.rawValue,
            "sampleCount": recommendation.profile.sampleCount,
            "knownDurationSeconds": recommendation.profile.knownDuration,
            "rankOrLinearDim": recommendation.rank,
            "epochs": recommendation.epochs,
            "layerCount": recommendation.layerCount,
            "gradientAccumulationSteps": recommendation.gradientAccumulationSteps,
            "estimatedOptimizerUpdates": recommendation.estimatedOptimizerUpdates,
            "summary": recommendation.summary,
            "rationale": recommendation.rationale
        ]
    }

    func trainingRunPayload(adapter: LoRAAdapter, log: TrainingRunLog?) -> [String: Any] {
        var payload = trainingAdapterSummaryPayload(adapter)
        payload["runLogPath"] = (try? AdapterRegistry.trainingRunLogURL(adapterID: adapter.id))?
            .path(percentEncoded: false) ?? NSNull()
        payload["hasRunLog"] = log != nil
        if let log {
            payload["runLog"] = trainingRunLogPayload(log)
        } else {
            payload["runLog"] = NSNull()
        }
        payload["updatedAt"] = log.map { dateFormatter.string(from: $0.updatedAt) }
            ?? dateFormatter.string(from: adapter.createdAt)
        return payload
    }

    func resolveTrainingRunAdapter(arguments: [String: Any]) throws -> LoRAAdapter {
        let installed = AdapterRegistry.listAdapters()
        guard !installed.isEmpty else {
            throw PiMPAudioCommandFacadeError.resourceNotFound("training runs")
        }

        guard let identifier = stringValue(arguments["id"] ?? arguments["adapterID"] ?? arguments["name"] ?? arguments["adapter"])?
            .trimmingCharacters(in: .whitespacesAndNewlines),
              !identifier.isEmpty,
              identifier.lowercased() != "latest" else {
            return installed.sorted {
                trainingRunUpdatedAt($0) > trainingRunUpdatedAt($1)
            }[0]
        }

        return try resolveAdapter(identifier, installed: installed)
    }

    func resolveTrainingRunComparisonAdapters(
        arguments: [String: Any],
        limit: Int,
        includeMissingLogs: Bool
    ) throws -> [LoRAAdapter] {
        let installed = AdapterRegistry.listAdapters()
        guard !installed.isEmpty else {
            throw PiMPAudioCommandFacadeError.resourceNotFound("training runs")
        }

        let rawIDs = arguments["ids"] ?? arguments["adapterIDs"] ?? arguments["adapters"]
        let identifiers: [String]
        if let strings = rawIDs as? [String] {
            identifiers = strings
        } else if let values = rawIDs as? [Any] {
            identifiers = values.compactMap { value in
                if let dict = value as? [String: Any] {
                    return stringValue(dict["id"] ?? dict["adapterID"] ?? dict["name"])
                }
                return stringValue(value)
            }
        } else {
            identifiers = []
        }

        if !identifiers.isEmpty {
            var seen = Set<String>()
            return try identifiers.compactMap { identifier in
                let adapter = try resolveAdapter(identifier, installed: installed)
                guard seen.insert(adapter.id).inserted else { return nil }
                return adapter
            }
        }

        return installed
            .filter { includeMissingLogs || AdapterRegistry.loadTrainingRunLog(adapterID: $0.id) != nil }
            .sorted { trainingRunUpdatedAt($0) > trainingRunUpdatedAt($1) }
            .prefix(limit)
            .map { $0 }
    }

    func trainingRunUpdatedAt(_ adapter: LoRAAdapter) -> Date {
        AdapterRegistry.loadTrainingRunLog(adapterID: adapter.id)?.updatedAt ?? adapter.createdAt
    }

    func trainingRunDetailPayload(
        adapter: LoRAAdapter,
        log: TrainingRunLog?,
        progressLimit: Int,
        eventLimit: Int
    ) -> [String: Any] {
        var payload = trainingRunPayload(adapter: adapter, log: log)
        payload["configuration"] = trainingAdapterConfigurationPayload(adapter)
        payload["paths"] = trainingRunPathsPayload(adapter)
        payload["health"] = trainingRunHealthPayload(adapter: adapter, log: log)
        if let log {
            payload["progressTailLimit"] = progressLimit
            payload["progressTail"] = log.progress.suffix(progressLimit).map(trainingRunProgressPayload)
            payload["eventTailLimit"] = eventLimit
            payload["eventTail"] = log.events.suffix(eventLimit).map(trainingRunEventPayload)
        } else {
            payload["progressTailLimit"] = progressLimit
            payload["progressTail"] = []
            payload["eventTailLimit"] = eventLimit
            payload["eventTail"] = []
        }
        payload["resources"] = [
            "collection": "pimpaudio://training-runs",
            "latest": "pimpaudio://training-run/latest",
            "adapter": resourceURI(kind: "adapter", identifier: adapter.id)
        ]
        return payload
    }

    func trainingRunComparisonPayload(adapter: LoRAAdapter, log: TrainingRunLog?) -> [String: Any] {
        let health = trainingRunHealthPayload(adapter: adapter, log: log)
        return [
            "id": adapter.id,
            "name": adapter.name,
            "kind": adapter.kind.rawValue,
            "baseModelID": adapter.baseModelID,
            "sampleCount": log?.sampleCount ?? adapter.sampleCount,
            "hasRunLog": log != nil,
            "status": log?.status.rawValue ?? "unknown",
            "updatedAt": dateFormatter.string(from: log?.updatedAt ?? adapter.createdAt),
            "createdAt": dateFormatter.string(from: adapter.createdAt),
            "rank": adapter.config.rank,
            "alpha": Double(adapter.config.alpha),
            "lokrLinearDim": adapter.lokrConfig?.linearDim ?? NSNull(),
            "lokrWeightDecompose": adapter.lokrConfig.map { $0.weightDecompose as Any } ?? NSNull(),
            "learningRate": Double(adapter.lokrConfig?.learningRate ?? adapter.config.learningRate),
            "epochs": adapter.lokrConfig?.maxEpochs ?? adapter.config.maxEpochs,
            "gradientAccumulationSteps": adapter.lokrConfig?.gradientAccumulationSteps ?? adapter.config.gradientAccumulationSteps,
            "trainingLayerCount": adapter.lokrConfig?.trainingLayerCount ?? adapter.config.trainingLayerCount ?? NSNull(),
            "injectionPatterns": adapter.lokrConfig?.injectionPatterns ?? adapter.config.injectionPatterns,
            "trainedPathCount": log?.trainedPaths.count ?? adapter.trainedPaths?.count ?? 0,
            "trainedLayerIndices": log?.trainedLayerIndices ?? adapter.trainedLayerIndices ?? [],
            "finalLoss": log?.finalLoss.map { Double($0) } ?? adapter.finalLoss.map { Double($0) } ?? NSNull(),
            "latestLoss": health["latestLoss"] ?? NSNull(),
            "bestLoss": health["bestLoss"] ?? NSNull(),
            "latestValidationLoss": health["latestValidationLoss"] ?? NSNull(),
            "bestValidationLoss": health["bestValidationLoss"] ?? NSNull(),
            "latestUpdateNorm": health["latestUpdateNorm"] ?? NSNull(),
            "zeroUpdateNormCount": health["zeroUpdateNormCount"] ?? NSNull(),
            "progressCount": log?.progress.count ?? 0,
            "validationCount": health["validationCount"] ?? NSNull(),
            "hasCheckpoint": AdapterRegistry.hasCheckpoint(adapterID: adapter.id)
        ]
    }

    func trainingAdapterConfigurationPayload(_ adapter: LoRAAdapter) -> [String: Any] {
        [
            "kind": adapter.kind.rawValue,
            "lora": loraConfigPayload(adapter.config),
            "lokr": adapter.lokrConfig.map(lokrConfigPayload) ?? NSNull(),
            "effectiveInjectionPatterns": adapter.lokrConfig?.injectionPatterns ?? adapter.config.injectionPatterns,
            "trainedPaths": adapter.trainedPaths ?? [],
            "trainedLayerIndices": adapter.trainedLayerIndices ?? []
        ]
    }

    func loraConfigPayload(_ config: LoRAConfig) -> [String: Any] {
        [
            "rank": config.rank,
            "alpha": Double(config.alpha),
            "scale": Double(config.alpha) / Double(max(1, config.rank)),
            "dropout": Double(config.dropout),
            "targetLayers": targetLayerPayload(config.targetLayers),
            "exactInjectionPatterns": config.exactInjectionPatterns ?? NSNull(),
            "injectionPatterns": config.injectionPatterns,
            "learningRate": Double(config.learningRate),
            "maxEpochs": config.maxEpochs,
            "batchSize": config.batchSize,
            "warmupFraction": Double(config.warmupFraction),
            "gradientAccumulationSteps": config.gradientAccumulationSteps,
            "trainingLayerCount": config.trainingLayerCount ?? NSNull(),
            "trainingSeed": config.trainingSeed.map { NSNumber(value: $0) } ?? NSNull()
        ]
    }

    func lokrConfigPayload(_ config: LoKRConfig) -> [String: Any] {
        [
            "linearDim": config.linearDim,
            "alpha": Double(config.alpha),
            "scale": Double(config.alpha) / Double(max(1, config.linearDim)),
            "dropout": Double(config.dropout),
            "weightDecompose": config.weightDecompose,
            "targetLayers": targetLayerPayload(config.targetLayers),
            "exactInjectionPatterns": config.exactInjectionPatterns ?? NSNull(),
            "injectionPatterns": config.injectionPatterns,
            "learningRate": Double(config.learningRate),
            "maxEpochs": config.maxEpochs,
            "batchSize": config.batchSize,
            "warmupFraction": Double(config.warmupFraction),
            "gradientAccumulationSteps": config.gradientAccumulationSteps,
            "trainingLayerCount": config.trainingLayerCount ?? NSNull(),
            "trainingSeed": config.trainingSeed.map { NSNumber(value: $0) } ?? NSNull()
        ]
    }

    func targetLayerPayload(_ layers: Set<TargetLayer>) -> [String] {
        layers.map(\.rawValue).sorted()
    }

    func trainingRunPathsPayload(_ adapter: LoRAAdapter) -> [String: Any] {
        let directory: URL
        let logURL: URL
        let checkpointURL: URL
        let optimizerURL: URL
        let weightsURL: URL
        do {
            directory = try AdapterRegistry.adapterDirectory(id: adapter.id)
            logURL = try AdapterRegistry.trainingRunLogURL(adapterID: adapter.id)
            checkpointURL = try AdapterRegistry.checkpointURL(adapterID: adapter.id)
            optimizerURL = try AdapterRegistry.optimizerStateURL(adapterID: adapter.id)
            weightsURL = try AdapterRegistry.weightsURL(adapterID: adapter.id)
        } catch {
            return ["error": error.localizedDescription]
        }
        return [
            "directory": directory.path(percentEncoded: false),
            "trainingLog": [
                "path": logURL.path(percentEncoded: false),
                "exists": FileManager.default.fileExists(atPath: logURL.path(percentEncoded: false))
            ],
            "checkpoint": [
                "path": checkpointURL.path(percentEncoded: false),
                "exists": FileManager.default.fileExists(atPath: checkpointURL.path(percentEncoded: false))
            ],
            "optimizerState": [
                "path": optimizerURL.path(percentEncoded: false),
                "exists": FileManager.default.fileExists(atPath: optimizerURL.path(percentEncoded: false))
            ],
            "weights": [
                "path": weightsURL.path(percentEncoded: false),
                "exists": FileManager.default.fileExists(atPath: weightsURL.path(percentEncoded: false))
            ]
        ]
    }

    func trainingRunHealthPayload(adapter: LoRAAdapter, log: TrainingRunLog?) -> [String: Any] {
        guard let log else {
            return [
                "hasRunLog": false,
                "finalLoss": adapter.finalLoss.map { Double($0) } ?? NSNull(),
                "progressCount": 0,
                "validationCount": 0
            ]
        }

        let losses = log.progress.map { Double($0.loss) }
        let validationLosses = (log.validationHistory ?? []).map { Double($0.loss) }
        let updateNorms = log.progress.compactMap(\.updateNorm).map(Double.init)
        let recentLosses = losses.suffix(10)
        let firstLoss = losses.first
        let latestLoss = losses.last ?? log.finalLoss.map(Double.init)
        let bestLoss = losses.min() ?? log.finalLoss.map(Double.init)
        let recentAverage = recentLosses.isEmpty
            ? nil
            : recentLosses.reduce(0, +) / Double(recentLosses.count)
        let zeroUpdateNormCount = updateNorms.filter { abs($0) <= 1.0e-12 }.count

        return [
            "hasRunLog": true,
            "status": log.status.rawValue,
            "progressCount": log.progress.count,
            "validationCount": log.validationHistory?.count ?? 0,
            "eventCount": log.events.count,
            "firstLoss": firstLoss ?? NSNull(),
            "latestLoss": latestLoss ?? NSNull(),
            "bestLoss": bestLoss ?? NSNull(),
            "latestValidationLoss": validationLosses.last ?? NSNull(),
            "bestValidationLoss": validationLosses.min() ?? NSNull(),
            "finalLoss": log.finalLoss.map { Double($0) } ?? NSNull(),
            "lossDelta": (firstLoss != nil && latestLoss != nil) ? latestLoss! - firstLoss! : NSNull(),
            "recentAverageLoss": recentAverage ?? NSNull(),
            "latestLearningRate": log.progress.last.map { Double($0.learningRate) } ?? NSNull(),
            "latestGradientNorm": log.progress.last?.gradientNorm.map { Double($0) } ?? NSNull(),
            "latestParameterNorm": log.progress.last?.parameterNorm.map { Double($0) } ?? NSNull(),
            "latestUpdateNorm": log.progress.last?.updateNorm.map { Double($0) } ?? NSNull(),
            "updateNormSampleCount": updateNorms.count,
            "zeroUpdateNormCount": zeroUpdateNormCount,
            "trainedPathCount": log.trainedPaths.count,
            "trainedLayerCount": Set(log.trainedLayerIndices).count,
            "weightsTensorCount": log.weightsTensorCount ?? NSNull(),
            "errorMessage": log.errorMessage ?? NSNull()
        ]
    }

    func trainingRunLogPayload(_ log: TrainingRunLog) -> [String: Any] {
        [
            "schemaVersion": log.schemaVersion,
            "adapterID": log.adapterID,
            "adapterName": log.adapterName,
            "adapterKind": log.adapterKind.rawValue,
            "status": log.status.rawValue,
            "baseModelID": log.baseModelID,
            "sampleCount": log.sampleCount,
            "resumedFromCheckpoint": log.resumedFromCheckpoint,
            "startedAt": dateFormatter.string(from: log.startedAt),
            "updatedAt": dateFormatter.string(from: log.updatedAt),
            "finishedAt": log.finishedAt.map { dateFormatter.string(from: $0) } ?? NSNull(),
            "progressCount": log.progress.count,
            "latestProgress": log.progress.last.map { trainingRunProgressPayload($0) as Any } ?? NSNull(),
            "lossHistoryCount": log.lossHistory.count,
            "latestLossHistory": Array(log.lossHistory.suffix(25)).map(Double.init),
            "validationHistoryCount": log.validationHistory?.count ?? 0,
            "latestValidationHistory": (log.validationHistory ?? [])
                .suffix(25)
                .map(trainingValidationPointPayload),
            "events": log.events.suffix(10).map(trainingRunEventPayload),
            "finalLoss": log.finalLoss.map { Double($0) } ?? NSNull(),
            "trainedPaths": log.trainedPaths,
            "trainedLayerIndices": log.trainedLayerIndices,
            "weightsTensorCount": log.weightsTensorCount ?? NSNull(),
            "errorMessage": log.errorMessage ?? NSNull()
        ]
    }

    func trainingRunProgressPayload(_ progress: TrainingRunProgressEntry) -> [String: Any] {
        [
            "timestamp": dateFormatter.string(from: progress.timestamp),
            "epoch": progress.epoch,
            "totalEpochs": progress.totalEpochs,
            "step": progress.step,
            "loss": Double(progress.loss),
            "learningRate": Double(progress.learningRate),
            "lossMovingAverage": progress.lossMovingAverage.map { Double($0) } ?? NSNull(),
            "gradientNorm": progress.gradientNorm.map { Double($0) } ?? NSNull(),
            "parameterNorm": progress.parameterNorm.map { Double($0) } ?? NSNull(),
            "updateNorm": progress.updateNorm.map { Double($0) } ?? NSNull(),
            "validationLoss": progress.validationLoss.map { Double($0) } ?? NSNull(),
            "elapsedSeconds": progress.elapsed,
            "estimatedRemainingSeconds": progress.estimatedRemaining.map { $0 as Any } ?? NSNull()
        ]
    }

    func trainingValidationPointPayload(_ point: TrainingValidationPoint) -> [String: Any] {
        [
            "epoch": point.epoch,
            "step": point.step,
            "loss": Double(point.loss)
        ]
    }

    func trainingRunEventPayload(_ event: TrainingRunEvent) -> [String: Any] {
        [
            "timestamp": dateFormatter.string(from: event.timestamp),
            "status": event.status.rawValue,
            "message": event.message
        ]
    }

    func runPodStatusPayload(
        service: RunPodService,
        recommendedForGPUType gpuTypeValue: String?
    ) -> [String: Any] {
        let limit = 100
        let trimmedGPUType = gpuTypeValue?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let recommended = trimmedGPUType.isEmpty
            ? nil
            : service.recommendedTrainingTemplate(for: trimmedGPUType)
        return [
            "hasAPIKey": service.hasAPIKey,
            "keySource": runPodKeySourcePayload(service.keySource),
            "isRefreshingPods": service.isRefreshing,
            "isRefreshingLaunchOptions": service.isRefreshingGPUTypes,
            "isLaunching": service.isLaunching,
            "terminatingPodID": service.terminatingPodID ?? NSNull(),
            "stoppingPodID": service.terminatingPodID ?? NSNull(),
            "lastMessage": service.lastMessage ?? NSNull(),
            "lastError": service.lastError ?? NSNull(),
            "gpuTypeCount": service.gpuTypes.count,
            "templateCount": service.podTemplates.count,
            "podCount": service.pods.count,
            "recommendedTemplate": recommended.map(runPodTemplatePayload) ?? NSNull(),
            "gpuTypes": service.gpuTypes.prefix(limit).map(runPodGPUTypePayload),
            "templates": service.podTemplates.prefix(limit).map(runPodTemplatePayload),
            "pods": service.pods.prefix(limit).map(runPodPodPayload)
        ]
    }

    func resolveRunPodLaunchConfiguration(
        arguments: [String: Any],
        service: RunPodService
    ) async throws -> (
        configuration: RunPodLaunchConfiguration,
        recommendedTemplate: RunPodTemplate?,
        usedRecommendedTemplate: Bool
    ) {
        let gpuType = try requiredString(
            "gpuType",
            aliases: ["gpuTypeID", "gpu"],
            in: arguments
        )
        var templateID = stringValue(arguments["templateID"] ?? arguments["templateId"] ?? arguments["template"])?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let imageName = stringValue(arguments["imageName"] ?? arguments["image"])?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        var containerDiskGB = integerValue(arguments["containerDiskGB"] ?? arguments["containerDiskGb"] ?? arguments["containerDisk"]) ?? 80
        var volumeGB = integerValue(arguments["volumeGB"] ?? arguments["volumeGb"] ?? arguments["volume"]) ?? 80
        let name = stringValue(arguments["name"] ?? arguments["podName"])?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let useRecommendedTemplate = boolValue(arguments["useRecommendedTemplate"] ?? arguments["autoTemplate"]) ?? true

        var recommendedTemplate: RunPodTemplate?
        var usedRecommendedTemplate = false
        if templateID.isEmpty,
           imageName.isEmpty,
           useRecommendedTemplate {
            if service.hasAPIKey,
               service.gpuTypes.isEmpty || service.podTemplates.isEmpty {
                await service.refreshLaunchOptions()
            }
            if let template = service.recommendedTrainingTemplate(for: gpuType) {
                templateID = template.id
                recommendedTemplate = template
                usedRecommendedTemplate = true
                if let templateContainerDiskGB = template.containerDiskInGb {
                    containerDiskGB = max(containerDiskGB, templateContainerDiskGB)
                }
                if let templateVolumeGB = template.volumeInGb {
                    volumeGB = max(volumeGB, templateVolumeGB)
                }
            }
        }

        var dataCenterIDs: [String] = []
        if let explicitCenters = stringArrayValue(
            arguments["dataCenterIDs"] ?? arguments["dataCenterIds"] ?? arguments["dataCenters"]
        ), !explicitCenters.isEmpty {
            dataCenterIDs = explicitCenters
        } else if let regionRaw = stringValue(arguments["region"])?
            .trimmingCharacters(in: .whitespacesAndNewlines),
                  !regionRaw.isEmpty {
            let region = try mcpRunPodRegion(from: regionRaw)
            if service.gpuTypes.isEmpty || service.dataCenters.isEmpty {
                await service.refreshLaunchOptions()
            }
            dataCenterIDs = service.launchDataCenterIDs(for: gpuType, in: region)
        }

        let configuration = RunPodLaunchConfiguration(
            name: name?.isEmpty == false ? name! : runPodDefaultPodName(),
            templateID: templateID,
            imageName: imageName,
            gpuTypeID: gpuType,
            dataCenterIDs: dataCenterIDs,
            containerDiskGB: containerDiskGB,
            volumeGB: volumeGB
        )
        return (configuration, recommendedTemplate, usedRecommendedTemplate)
    }

    private func mcpRunPodRegion(from raw: String) throws -> RunPodRegion {
        let normalized = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        switch normalized {
        case "any", "all", "global":
            return .any
        case "asia", "as":
            return .asia
        case "northamerica", "north_america", "north-america", "na", "us", "usa":
            return .northAmerica
        case "europe", "eu", "emea":
            return .europe
        case "oceania", "australia", "au", "apac":
            return .oceania
        default:
            if let region = RunPodRegion(rawValue: raw) {
                return region
            }
            throw PiMPAudioCommandFacadeError.invalidTrainingConfiguration(
                "Unsupported region '\(raw)'. Use any, asia, northAmerica, europe, or oceania."
            )
        }
    }

    func runPodLaunchPayload(
        configuration: RunPodLaunchConfiguration,
        recommendedTemplate: RunPodTemplate?,
        usedRecommendedTemplate: Bool,
        submitted: Bool,
        dryRun: Bool,
        pod: RunPodPodSummary?,
        error: String?
    ) -> [String: Any] {
        [
            "submitted": submitted,
            "dryRun": dryRun,
            "configuration": runPodLaunchConfigurationPayload(configuration),
            "usedRecommendedTemplate": usedRecommendedTemplate,
            "recommendedTemplate": recommendedTemplate.map(runPodTemplatePayload) ?? NSNull(),
            "pod": pod.map(runPodPodPayload) ?? NSNull(),
            "error": error ?? NSNull(),
            "note": dryRun
                ? "Dry run only. Call launch_runpod_pod with dryRun: false and confirm: true to create billable RunPod GPU time."
                : "Launch submitted through the same RunPodService path as the UI."
        ]
    }

    func runPodLaunchConfigurationPayload(_ configuration: RunPodLaunchConfiguration) -> [String: Any] {
        [
            "name": configuration.name,
            "templateID": configuration.templateID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                ? NSNull()
                : configuration.templateID,
            "imageName": configuration.imageName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                ? NSNull()
                : configuration.imageName,
            "gpuType": configuration.gpuTypeID,
            "dataCenterIDs": configuration.dataCenterIDs,
            "containerDiskGB": configuration.containerDiskGB,
            "volumeGB": configuration.volumeGB,
            "ports": configuration.ports
        ]
    }

    func runPodPodPayload(_ pod: RunPodPodSummary) -> [String: Any] {
        [
            "id": pod.id,
            "name": pod.displayName,
            "status": pod.statusLabel,
            "gpu": pod.gpuLabel,
            "gpuCount": pod.gpuCount ?? NSNull(),
            "hourlyCost": pod.hourlyCostLabel ?? NSNull(),
            "publicIP": pod.publicIp ?? NSNull(),
            "sshHost": pod.sshHost ?? NSNull(),
            "sshPort": pod.sshPort ?? NSNull(),
            "sshUsername": pod.sshDirect == nil ? NSNull() : pod.sshUsername,
            "locked": pod.locked ?? NSNull(),
            "templateID": pod.templateId ?? NSNull(),
            "image": pod.image ?? NSNull(),
            "containerDiskGB": pod.containerDiskInGb ?? NSNull(),
            "volumeGB": pod.volumeInGb ?? NSNull(),
            "volumeMountPath": pod.volumeMountPath ?? NSNull()
        ]
    }

    func runPodGPUTypePayload(_ gpuType: RunPodGPUType) -> [String: Any] {
        [
            "id": gpuType.id,
            "displayName": gpuType.displayName ?? gpuType.id,
            "manufacturer": gpuType.manufacturer ?? NSNull(),
            "memoryGB": gpuType.memoryInGb ?? NSNull(),
            "secureCloud": gpuType.secureCloud ?? NSNull(),
            "stockStatus": gpuType.secureLowestPrice?.stockStatus ?? NSNull(),
            "menuTitle": gpuType.menuTitle,
            "selectionValue": gpuType.selectionValue,
            "detail": gpuType.detailLabel()
        ]
    }

    func runPodTemplatePayload(_ template: RunPodTemplate) -> [String: Any] {
        [
            "id": template.id,
            "name": template.displayNameValue,
            "imageName": template.imageNameValue ?? NSNull(),
            "category": template.category ?? NSNull(),
            "isRunpod": template.isRunpod ?? NSNull(),
            "isPublic": template.isPublic ?? NSNull(),
            "isServerless": template.isServerless ?? NSNull(),
            "isOfficialPodTemplate": template.isOfficialPodTemplate,
            "containerDiskGB": template.containerDiskInGb ?? NSNull(),
            "volumeGB": template.volumeInGb ?? NSNull(),
            "volumeMountPath": template.volumeMountPath ?? NSNull(),
            "ports": template.ports ?? [],
            "summary": template.summaryLabel,
            "detail": template.detailLabel
        ]
    }

    func runPodKeySourcePayload(_ source: APIKeyStore.KeySource) -> String {
        switch source {
        case .keychain:
            return "keychain"
        case .environment:
            return "environment"
        case .none:
            return "none"
        }
    }

    func runPodDefaultPodName() -> String {
        "pimp-audio-remote-\(UUID().uuidString.prefix(8).lowercased())"
    }


}
