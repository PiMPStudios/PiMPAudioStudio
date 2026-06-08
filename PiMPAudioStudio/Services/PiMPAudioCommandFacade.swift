import Foundation
import AVFoundation
import PiMPAudioCore

@MainActor
final class PiMPAudioCommandFacade {
    private let registry: ModelRegistry
    private let promptEnhancer: PromptEnhancer
    private let lyricWriterManager: LyricWriterManager
    private var historyStore: HistoryStore
    private var generationService: GenerationService?
    private var editorService: EditorService?
    private var trainingService: TrainingService?
    private let dateFormatter = ISO8601DateFormatter()
    private let renderFileDateFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd_HHmmss"
        return formatter
    }()
    private let maxOperationRecords = 50
    private var operationRecords: [String: MCPOperationRecord] = [:]
    private var operationOrder: [String] = []
    private var operationTasks: [String: Task<Void, Never>] = [:]
    private var activeGenerationOperationID: String?
    private let generationProgressPlan: [String: (base: Double, span: Double)] = [
        "queued": (0.00, 0.02),
        "idle": (0.00, 0.02),
        "downloadingModel": (0.02, 0.13),
        "loadingWeights": (0.15, 0.20),
        "conditioning": (0.35, 0.05),
        "tokenizingSource": (0.40, 0.05),
        "lmAudioCodeConditioning": (0.45, 0.08),
        "generating": (0.53, 0.32),
        "decoding": (0.85, 0.10),
        "complete": (1.00, 0.00),
        "error": (1.00, 0.00),
        "cancelled": (1.00, 0.00)
    ]
    private let editorLoadProgressPlan: [String: (base: Double, span: Double)] = [
        "queued": (0.00, 0.02),
        "readingFile": (0.02, 0.18),
        "preparingAudio": (0.20, 0.15),
        "encodingLatents": (0.35, 0.60),
        "complete": (1.00, 0.00),
        "error": (1.00, 0.00),
        "cancelled": (1.00, 0.00)
    ]
    private let editorInpaintProgressPlan: [String: (base: Double, span: Double)] = [
        "queued": (0.00, 0.02),
        "buildingMask": (0.02, 0.06),
        "encoding": (0.08, 0.10),
        "diffusion": (0.18, 0.70),
        "decoding": (0.88, 0.10),
        "complete": (1.00, 0.00),
        "error": (1.00, 0.00),
        "cancelled": (1.00, 0.00)
    ]
    private let editorRenderProgressPlan: [String: (base: Double, span: Double)] = [
        "queued": (0.00, 0.02),
        "readingBuffer": (0.02, 0.18),
        "preparingAudio": (0.20, 0.30),
        "writingFile": (0.50, 0.45),
        "complete": (1.00, 0.00),
        "error": (1.00, 0.00),
        "cancelled": (1.00, 0.00)
    ]
    private let editorDSPProgressPlan: [String: (base: Double, span: Double)] = [
        "queued": (0.00, 0.02),
        "readingBuffer": (0.02, 0.13),
        "applyingEffects": (0.15, 0.35),
        "preparingAudio": (0.50, 0.10),
        "encodingLatents": (0.60, 0.35),
        "complete": (1.00, 0.00),
        "error": (1.00, 0.00),
        "cancelled": (1.00, 0.00)
    ]

    init(
        registry: ModelRegistry,
        historyStore: HistoryStore,
        promptEnhancer: PromptEnhancer,
        lyricWriterManager: LyricWriterManager
    ) {
        self.registry = registry
        self.promptEnhancer = promptEnhancer
        self.lyricWriterManager = lyricWriterManager
        self.historyStore = historyStore
    }

    func configure(historyStore: HistoryStore) {
        self.historyStore = historyStore
    }

    func configure(generationService: GenerationService) {
        self.generationService = generationService
    }

    func configure(editorService: EditorService) {
        self.editorService = editorService
    }

    func configure(trainingService: TrainingService) {
        self.trainingService = trainingService
    }

    func listModels() async -> [String: Any] {
        let specs = await registry.allSpecs
        var activeByCategory: [String: String] = [:]
        let categories = Set(specs.map(\.category))
        for category in categories {
            if let activeID = await registry.activeModelID(for: category) {
                activeByCategory[category.rawValue] = activeID
            }
        }

        var models: [[String: Any]] = []
        for spec in specs {
            models.append([
                "id": spec.id,
                "displayName": spec.displayName,
                "category": spec.category.rawValue,
                "family": spec.variantFamily ?? NSNull(),
                "huggingFaceRepo": spec.huggingFaceRepo,
                "totalSizeBytes": spec.totalSizeBytes,
                "totalSizeFormatted": spec.totalSizeFormatted,
                "requiresAuth": spec.requiresAuth,
                "isRunnable": spec.isRunnable,
                "runtimeNote": spec.runtimeNote ?? NSNull(),
                "status": await registry.status(for: spec.id).mcpDescription,
                "isActive": activeByCategory[spec.category.rawValue] == spec.id
            ])
        }
        models.sort {
            String(describing: $0["id"] ?? "") < String(describing: $1["id"] ?? "")
        }

        return [
            "activeModels": activeByCategory,
            "models": models
        ]
    }

    func listAdapters() async -> [String: Any] {
        let adapters = await Task.detached(priority: .userInitiated) {
            AdapterRegistry.listAdapters()
        }.value

        return [
            "adapters": adapters.map { adapter in
                [
                    "id": adapter.id,
                    "name": adapter.name,
                    "baseModelID": adapter.baseModelID,
                    "baseModelCategory": adapter.baseModelCategory.rawValue,
                    "rank": adapter.config.rank,
                    "alpha": adapter.config.alpha,
                    "injectionPatterns": adapter.config.injectionPatterns,
                    "sampleCount": adapter.sampleCount,
                    "finalLoss": adapter.finalLoss.map { Double($0) } ?? NSNull(),
                    "createdAt": dateFormatter.string(from: adapter.createdAt),
                    "hasCheckpoint": AdapterRegistry.hasCheckpoint(adapterID: adapter.id)
                ] as [String: Any]
            }
        ]
    }

    func listHistory(limit: Int) -> [String: Any] {
        let cappedLimit = max(1, min(limit, 200))
        let rows = historyStore.entries.prefix(cappedLimit).map(historySummary)
        return [
            "count": rows.count,
            "history": Array(rows)
        ]
    }

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

    func listResources() async -> [[String: Any]] {
        var resources: [[String: Any]] = [
            resource(
                uri: "pimpaudio://models",
                name: "models",
                title: "PiMP Audio Studio Models",
                description: "Current model registry state and active model selections."
            ),
            resource(
                uri: "pimpaudio://adapters",
                name: "adapters",
                title: "PiMP Audio Studio LoRA Adapters",
                description: "Installed LoRA adapters and checkpoint metadata."
            ),
            resource(
                uri: "pimpaudio://training",
                name: "training",
                title: "Training Status",
                description: "Current local LoRA/LoKR training state, active run summary, and diagnostics log pointers."
            ),
            resource(
                uri: "pimpaudio://training-datasets",
                name: "training-datasets",
                title: "Training Datasets",
                description: "Local dataset packages visible from the current output folder."
            ),
            resource(
                uri: "pimpaudio://training-runs",
                name: "training-runs",
                title: "Training Runs",
                description: "Installed adapter training logs with latest progress and health metrics."
            ),
            resource(
                uri: "pimpaudio://training-dataset/latest",
                name: "latest-training-dataset",
                title: "Latest Training Dataset",
                description: "Detailed inspection payload for the newest visible local dataset package."
            ),
            resource(
                uri: "pimpaudio://training-run/latest",
                name: "latest-training-run",
                title: "Latest Training Run",
                description: "Detailed inspection payload for the newest installed adapter training log."
            ),
            resource(
                uri: "pimpaudio://history",
                name: "history",
                title: "Recent Generation History",
                description: "Recent generation rows with model, seed, prompt, audio path, and trace availability."
            ),
            resource(
                uri: resourceURI(kind: "project", identifier: "current"),
                name: "current",
                title: "Current PiMP Audio Studio Project",
                description: "Current local workstation project/session with editor, take, and operation links."
            ),
            resource(
                uri: resourceURI(kind: "track", identifier: "current"),
                name: "current",
                title: "Current Editor Track",
                description: "Current editor audio track and loaded clip summary."
            ),
            resource(
                uri: resourceURI(kind: "clip", identifier: "current"),
                name: "current",
                title: "Current Editor Clip",
                description: "Loaded editor audio as a clip-like resource."
            ),
            resource(
                uri: resourceURI(kind: "region", identifier: "current-selection"),
                name: "current-selection",
                title: "Current Editor Region",
                description: "Current or most recently requested editor selection/region metadata."
            ),
            resource(
                uri: "pimpaudio://takes",
                name: "takes",
                title: "Recent Takes",
                description: "Recent generated takes with links to history, audio, and trace resources."
            ),
            resource(
                uri: "pimpaudio://presets",
                name: "presets",
                title: "Generation and Editor Presets",
                description: "Saved generation presets plus built-in editor/guidance preset catalogs."
            ),
            resource(
                uri: "pimpaudio://workflows",
                name: "workflows",
                title: "PiMP Audio Studio Workflow Prompts",
                description: "Reusable agent workflow prompt templates for generation, editing, and review."
            ),
            resource(
                uri: "pimpaudio://editor",
                name: "editor",
                title: "Editor Status",
                description: "Current editor readiness, loaded audio metadata, supported actions, and active editor operation."
            ),
            resource(
                uri: "pimpaudio://operations",
                name: "operations",
                title: "MCP Operations",
                description: "Recent MCP-observable operations with progress, status, result, and cancellation metadata."
            )
        ]

        let specs = await registry.allSpecs
        for spec in specs.sorted(by: { $0.id < $1.id }) {
            resources.append(resource(
                uri: resourceURI(kind: "model", identifier: spec.id),
                name: spec.id,
                title: spec.displayName,
                description: "Model registry entry for \(spec.displayName)."
            ))
        }

        let adapters = await loadAdapters()
        for adapter in adapters.sorted(by: { $0.name < $1.name }) {
            resources.append(resource(
                uri: resourceURI(kind: "adapter", identifier: adapter.id),
                name: adapter.id,
                title: adapter.name,
                description: "LoRA adapter metadata for \(adapter.name)."
            ))
        }

        for entry in historyStore.entries.prefix(100) {
            resources.append(resource(
                uri: resourceURI(kind: "history", identifier: entry.id.uuidString),
                name: entry.id.uuidString,
                title: compact(entry.prompt, limit: 80),
                description: "Generation history entry for \(entry.fileName)."
            ))

            resources.append(resource(
                uri: resourceURI(kind: "take", identifier: entry.id.uuidString),
                name: entry.id.uuidString,
                title: "Take: \(compact(entry.prompt, limit: 70))",
                description: "Workstation take view for \(entry.fileName)."
            ))

            if historyStore.traceURL(for: entry) != nil {
                resources.append(resource(
                    uri: resourceURI(kind: "trace", identifier: entry.id.uuidString),
                    name: entry.traceFileName ?? HistoryEntry.traceFileName(forAudioFileName: entry.fileName),
                    title: "Trace: \(compact(entry.prompt, limit: 70))",
                    description: "Generation trace sidecar for \(entry.fileName)."
                ))
            }

            resources.append(resource(
                uri: resourceURI(kind: "audio", identifier: entry.id.uuidString),
                name: entry.fileName,
                title: "Audio: \(compact(entry.prompt, limit: 70))",
                description: "Local audio file metadata and path. Audio bytes are not streamed."
            ))
        }

        for preset in loadGenerationPresets() {
            resources.append(resource(
                uri: resourceURI(kind: "preset", identifier: preset.id.uuidString),
                name: preset.id.uuidString,
                title: preset.name,
                description: "Saved generation preset."
            ))
        }

        for workflow in workflowPromptDefinitions {
            resources.append(resource(
                uri: resourceURI(kind: "workflow", identifier: workflow.id),
                name: workflow.id,
                title: workflow.title,
                description: workflow.description
            ))
        }

        refreshGenerationOperation()
        for operation in operationSnapshot(limit: maxOperationRecords) {
            resources.append(resource(
                uri: resourceURI(kind: "operation", identifier: operation.id),
                name: operation.id,
                title: operation.title,
                description: "\(operation.kind) operation is \(operation.status.rawValue)."
            ))
        }

        return resources
    }

    func readResource(uri: String) async throws -> [String: Any] {
        guard let parsed = PiMPAudioResourceURI(uri) else {
            throw PiMPAudioCommandFacadeError.invalidResourceURI(uri)
        }

        switch parsed.kind {
        case "models":
            return await listModels()
        case "model":
            guard let identifier = parsed.identifier else {
                throw PiMPAudioCommandFacadeError.invalidResourceURI(uri)
            }
            return try await modelResource(id: identifier)
        case "adapters":
            return await listAdapters()
        case "adapter":
            guard let identifier = parsed.identifier else {
                throw PiMPAudioCommandFacadeError.invalidResourceURI(uri)
            }
            return try await adapterResource(id: identifier)
        case "training":
            return trainingStatus()
        case "training-datasets":
            return listTrainingDatasets(arguments: [:])
        case "training-runs":
            return listTrainingRuns(arguments: [:])
        case "training-dataset":
            var arguments: [String: Any] = ["sampleLimit": 12]
            if let identifier = parsed.identifier, identifier != "latest" {
                arguments["id"] = identifier
            }
            return try inspectTrainingDataset(arguments: arguments)
        case "training-run":
            var arguments: [String: Any] = ["progressLimit": 50, "eventLimit": 25]
            if let identifier = parsed.identifier, identifier != "latest" {
                arguments["id"] = identifier
            }
            return try inspectTrainingRun(arguments: arguments)
        case "history":
            guard let identifier = parsed.identifier else {
                return listHistory(limit: 200)
            }
            let entry = try resolveHistoryEntry(identifier)
            return historySummary(entry)
        case "project":
            try validateResourceIdentifier(parsed.identifier, expected: "current", uri: uri)
            return try await projectResource()
        case "track":
            try validateResourceIdentifier(parsed.identifier, expected: "current", uri: uri)
            return await currentTrackResource()
        case "clip":
            try validateResourceIdentifier(parsed.identifier, expected: "current", uri: uri)
            return await currentClipResource()
        case "region":
            try validateResourceIdentifier(parsed.identifier, expected: "current-selection", uri: uri)
            return await currentRegionResource()
        case "takes":
            return takesResource(limit: 200)
        case "take":
            guard let identifier = parsed.identifier else {
                throw PiMPAudioCommandFacadeError.invalidResourceURI(uri)
            }
            return try takeResource(identifier: identifier)
        case "presets":
            return presetsResource()
        case "preset":
            guard let identifier = parsed.identifier else {
                throw PiMPAudioCommandFacadeError.invalidResourceURI(uri)
            }
            return try presetResource(identifier: identifier)
        case "workflows":
            return workflowsResource()
        case "workflow":
            guard let identifier = parsed.identifier else {
                throw PiMPAudioCommandFacadeError.invalidResourceURI(uri)
            }
            return try workflowResource(identifier: identifier)
        case "trace":
            guard let identifier = parsed.identifier else {
                throw PiMPAudioCommandFacadeError.invalidResourceURI(uri)
            }
            return try getTrace(arguments: ["id": identifier])
        case "audio":
            guard let identifier = parsed.identifier else {
                throw PiMPAudioCommandFacadeError.invalidResourceURI(uri)
            }
            return try audioResource(identifier: identifier)
        case "editor":
            return try await editorStatus()
        case "operations":
            return listOperations(arguments: ["limit": maxOperationRecords])
        case "operation":
            guard let identifier = parsed.identifier else {
                throw PiMPAudioCommandFacadeError.invalidResourceURI(uri)
            }
            return try getOperation(arguments: ["id": identifier])
        default:
            throw PiMPAudioCommandFacadeError.resourceNotFound(uri)
        }
    }

    func getTrace(arguments: [String: Any]) throws -> [String: Any] {
        let trace = try resolveTrace(arguments: arguments)
        return try tracePayload(trace)
    }

    func compareTraces(arguments: [String: Any]) throws -> [String: Any] {
        let left = try resolveTrace(arguments: [
            "id": arguments["left"] ?? arguments["leftID"] ?? arguments["leftTrace"] ?? ""
        ])
        let right = try resolveTrace(arguments: [
            "id": arguments["right"] ?? arguments["rightID"] ?? arguments["rightTrace"] ?? ""
        ])
        let groups = GenerationTraceComparator.groups(left: left, right: right)
        let changedGroups = groups.compactMap { group -> [String: Any]? in
            let rows = group.rows.filter(\.isDifferent)
            guard !rows.isEmpty else { return nil }
            return [
                "title": group.title,
                "systemImage": group.systemImage,
                "rows": rows.map(rowPayload)
            ]
        }

        return [
            "left": traceSummary(left),
            "right": traceSummary(right),
            "changedFieldCount": changedGroups.reduce(0) { total, group in
                total + ((group["rows"] as? [[String: Any]])?.count ?? 0)
            },
            "groups": changedGroups
        ]
    }

    func markPreferredTake(arguments: [String: Any]) throws -> [String: Any] {
        let trace = try resolveTrace(arguments: arguments)
        let hasPreferredArgument = hasArgument("isPreferred", in: arguments)
            || hasArgument("preferred", in: arguments)
        let preferredValue = arguments["isPreferred"] ?? arguments["preferred"]
        let isPreferred: Bool
        if hasPreferredArgument {
            guard let parsed = boolValue(preferredValue) else {
                throw PiMPAudioCommandFacadeError.invalidBoolean("isPreferred")
            }
            isPreferred = parsed
        } else {
            isPreferred = true
        }

        let note: String?
        if hasArgument("listeningNote", in: arguments) || hasArgument("note", in: arguments) {
            note = stringValue(arguments["listeningNote"] ?? arguments["note"])
        } else {
            note = trace.listeningNote
        }

        let updated = try historyStore.saveTraceReview(
            traceFileName: trace.traceFileName,
            listeningNote: note,
            isPreferred: isPreferred
        )

        return reviewPayload(updated)
    }

    func enhancePrompt(arguments: [String: Any]) async throws -> [String: Any] {
        let prompt = try requiredString("prompt", in: arguments)
        let mode = generationMode(from: arguments["mode"] ?? arguments["generationMode"]) ?? .music
        let enhanced = await promptEnhancer.enhance(prompt: prompt, mode: mode)

        return [
            "prompt": prompt,
            "enhancedPrompt": enhanced,
            "mode": mode.rawValue,
            "modeDisplayName": mode.displayName,
            "isAvailable": promptEnhancer.isAvailable,
            "changed": normalizedText(enhanced) != normalizedText(prompt)
        ]
    }

    func writeLyrics(arguments: [String: Any]) async throws -> [String: Any] {
        let prompt = try requiredString("prompt", in: arguments)
        let style = stringValue(arguments["style"]) ?? ""
        let duration = try requiredDuration(in: arguments)
        let lyrics = try await lyricWriterManager.writeLyricsText(
            prompt: prompt,
            style: style,
            duration: duration
        )

        return [
            "prompt": prompt,
            "style": style,
            "duration": duration,
            "backend": lyricWriterManager.activeBackend,
            "lyrics": lyrics,
            "lineCount": lyricLineCount(lyrics)
        ]
    }

    func applyAdapter(arguments: [String: Any]) async throws -> [String: Any] {
        guard let generationService else {
            throw PiMPAudioCommandFacadeError.generationServiceUnavailable
        }

        if boolValue(arguments["clear"] ?? arguments["none"]) == true {
            generationService.selectedAdapterID = nil
            generationService.adapterStack = []
            persistAdapterSelection([])
            notifyAdapterSelectionChanged([])
            return adapterSelectionPayload([])
        }

        let installed = await loadAdapters()
        let selection = try parseAdapterSelection(arguments, installed: installed)
        generationService.selectedAdapterID = selection.first?.adapter.id
        generationService.adapterStack = selection.map { (id: $0.adapter.id, weight: $0.strength) }
        persistAdapterSelection(selection)
        notifyAdapterSelectionChanged(selection)
        return adapterSelectionPayload(selection)
    }

    func generateSong(arguments: [String: Any]) async throws -> [String: Any] {
        let dryRun = boolValue(arguments["dryRun"] ?? arguments["validateOnly"]) == true
        let originalPrompt = try requiredString("prompt", aliases: ["text"], in: arguments)
        let mode = generationMode(from: arguments["mode"] ?? arguments["generationMode"]) ?? .music
        let shouldEnhancePrompt = boolValue(
            arguments["enhancePrompt"] ?? arguments["enhance"] ?? arguments["autoEnhance"]
        ) == true
        let shouldWriteLyrics = boolValue(
            arguments["writeLyrics"] ?? arguments["generateLyrics"] ?? arguments["autoLyrics"]
        ) == true

        var workingArguments = arguments
        var finalPrompt = originalPrompt
        var enhancedPrompt: String?
        if shouldEnhancePrompt {
            let enhanced = await promptEnhancer.enhance(prompt: originalPrompt, mode: mode)
            if !normalizedText(enhanced).isEmpty {
                finalPrompt = enhanced
                enhancedPrompt = enhanced
            }
        }
        workingArguments["prompt"] = finalPrompt

        let suppliedLyrics = stringValue(arguments["lyrics"])?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        var finalLyrics = suppliedLyrics
        var lyricPayload: [String: Any]?
        if suppliedLyrics.isEmpty && shouldWriteLyrics {
            let duration = try orchestrationDuration(in: arguments)
            let style = stringValue(arguments["lyricStyle"] ?? arguments["style"]) ?? ""
            finalLyrics = try await lyricWriterManager.writeLyricsText(
                prompt: finalPrompt,
                style: style,
                duration: duration
            )
            workingArguments["lyrics"] = finalLyrics
            lyricPayload = [
                "generated": true,
                "backend": lyricWriterManager.activeBackend,
                "style": style,
                "duration": duration,
                "lineCount": lyricLineCount(finalLyrics)
            ]
        } else {
            lyricPayload = [
                "generated": false,
                "supplied": !suppliedLyrics.isEmpty,
                "lineCount": lyricLineCount(finalLyrics)
            ]
        }

        let adapterPayload: [String: Any]?
        if let adapterArguments = orchestrationAdapterArguments(from: arguments) {
            if dryRun {
                adapterPayload = ["skipped": true, "reason": "dryRun"]
            } else {
                adapterPayload = try await applyAdapter(arguments: adapterArguments)
            }
        } else {
            adapterPayload = nil
        }

        let submission = try generate(arguments: workingArguments)
        return [
            "submitted": submission["submitted"] ?? false,
            "dryRun": submission["dryRun"] ?? dryRun,
            "originalPrompt": originalPrompt,
            "finalPrompt": finalPrompt,
            "promptEnhanced": shouldEnhancePrompt,
            "enhancedPrompt": enhancedPrompt ?? NSNull(),
            "lyrics": finalLyrics.isEmpty ? NSNull() : finalLyrics,
            "lyricWriting": lyricPayload ?? NSNull(),
            "adapterSelection": adapterPayload ?? NSNull(),
            "generation": submission
        ]
    }

    func generate(arguments: [String: Any]) throws -> [String: Any] {
        guard let generationService else {
            throw PiMPAudioCommandFacadeError.generationServiceUnavailable
        }

        let request = try generationRequest(arguments)
        let dryRun = boolValue(arguments["dryRun"] ?? arguments["validateOnly"]) == true
        if dryRun {
            return generationSubmissionPayload(request: request, submitted: false, dryRun: true)
        }

        guard generationService.outputDirectoryPath != nil else {
            throw PiMPAudioCommandFacadeError.outputDirectoryUnavailable
        }
        guard generationService.canSubmitGeneration else {
            throw PiMPAudioCommandFacadeError.generationAlreadyActive(stateName(generationService.state))
        }

        syncGenerationAdapterSelectionFromDefaultsIfNeeded(generationService)
        let operation = registerGenerationOperation(request: request)
        generationService.generate(request: request)
        refreshGenerationOperation()
        return generationSubmissionPayload(
            request: request,
            submitted: true,
            dryRun: false,
            operationID: operation.id
        )
    }

    func generationStatus() throws -> [String: Any] {
        guard let generationService else {
            throw PiMPAudioCommandFacadeError.generationServiceUnavailable
        }

        refreshGenerationOperation()
        return [
            "state": generationStatePayload(generationService.state),
            "currentResult": generationService.currentResult.map(generationResultPayload) ?? NSNull(),
            "currentOperation": currentOperationPayload() ?? NSNull()
        ]
    }

    func editorStatus() async throws -> [String: Any] {
        guard let editorService else {
            throw PiMPAudioCommandFacadeError.editorServiceUnavailable
        }

        let coordinator = editorService.coordinator
        let loadedAudio = editorService.loadedAudio
        let hasLoadedAudio = await coordinator?.hasLoadedAudio ?? false
        let canUndo = await coordinator?.canUndo ?? false
        let canRedo = await coordinator?.canRedo ?? false

        return [
            "ready": coordinator != nil,
            "hasLoadedAudio": hasLoadedAudio,
            "loadedAudio": loadedAudio.map(editorLoadedAudioPayload) ?? NSNull(),
            "lastEdit": editorService.lastEdit.map(editorEditSummaryPayload) ?? NSNull(),
            "canUndo": canUndo,
            "canRedo": canRedo,
            "currentOperation": currentOperationPayload(kindPrefix: "editor.") ?? NSNull(),
            "supportedActions": editorActionsPayload()
        ]
    }

    func loadEditorAudio(arguments: [String: Any]) throws -> [String: Any] {
        guard editorService?.coordinator != nil else {
            throw PiMPAudioCommandFacadeError.editorServiceUnavailable
        }

        let source = try editorAudioLoadSource(from: arguments)
        try validateReadableFile(source.url)

        let originalPrompt = stringValue(arguments["originalPrompt"] ?? arguments["prompt"])?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let resolvedPrompt = originalPrompt?.isEmpty == false ? originalPrompt : source.originalPrompt
        let request = editorAudioLoadRequestPayload(source: source, originalPrompt: resolvedPrompt)
        let dryRun = boolValue(arguments["dryRun"] ?? arguments["validateOnly"]) == true
        if dryRun {
            return [
                "submitted": false,
                "dryRun": true,
                "request": request,
                "operationID": NSNull(),
                "operation": NSNull()
            ]
        }

        let operation = registerOperation(
            kind: "editor.loadAudio",
            title: "Load editor audio",
            request: request,
            progress: editorLoadProgressPayload(stage: "queued", message: "Editor audio load accepted.")
        )

        operationTasks[operation.id] = Task { [weak self] in
            await self?.runEditorAudioLoadOperation(
                id: operation.id,
                source: source,
                originalPrompt: resolvedPrompt
            )
        }

        return [
            "submitted": true,
            "dryRun": false,
            "request": request,
            "operationID": operation.id,
            "operation": operationPayload(operation)
        ]
    }

    func submitEditorInpaint(arguments: [String: Any]) async throws -> [String: Any] {
        let dryRun = boolValue(arguments["dryRun"] ?? arguments["validateOnly"]) == true
        let loadedAudio = editorService?.loadedAudio
        let request = try editorInpaintRequest(arguments, loadedAudio: loadedAudio)
        let requestPayload = editorInpaintRequestPayload(request)

        if dryRun {
            return [
                "submitted": false,
                "dryRun": true,
                "requiresLoadedAudio": true,
                "requiresConfirmation": true,
                "confirmation": confirmationRequirementPayload(toolName: "submit_editor_inpaint"),
                "request": requestPayload,
                "operationID": NSNull(),
                "operation": NSNull()
            ]
        }

        guard boolValue(arguments["confirm"]) == true else {
            throw PiMPAudioCommandFacadeError.confirmationRequired("submit_editor_inpaint")
        }
        guard let editorService, let coordinator = editorService.coordinator else {
            throw PiMPAudioCommandFacadeError.editorServiceUnavailable
        }
        guard await coordinator.hasLoadedAudio else {
            throw PiMPAudioCommandFacadeError.editorAudioNotLoaded
        }
        try validateEditorSelection(request.selection, loadedAudio: editorService.loadedAudio)

        let operation = registerOperation(
            kind: "editor.inpaint",
            title: request.editMode == .repaint ? "Repaint editor audio" : "Inpaint editor audio",
            request: requestPayload,
            progress: editorInpaintProgressPayload(stage: "queued", message: "Editor inpaint accepted.")
        )

        operationTasks[operation.id] = Task { [weak self] in
            await self?.runEditorInpaintOperation(id: operation.id, request: request)
        }

        return [
            "submitted": true,
            "dryRun": false,
            "request": requestPayload,
            "operationID": operation.id,
            "operation": operationPayload(operation)
        ]
    }

    func renderEditorAudio(arguments: [String: Any]) async throws -> [String: Any] {
        let dryRun = boolValue(arguments["dryRun"] ?? arguments["validateOnly"]) == true
        let loadedAudio = editorService?.loadedAudio
        let request = try editorRenderRequest(arguments, loadedAudio: loadedAudio)
        let requestPayload = editorRenderRequestPayload(request)

        if dryRun {
            return [
                "submitted": false,
                "dryRun": true,
                "requiresLoadedAudio": true,
                "request": requestPayload,
                "operationID": NSNull(),
                "operation": NSNull()
            ]
        }

        guard let editorService, let coordinator = editorService.coordinator else {
            throw PiMPAudioCommandFacadeError.editorServiceUnavailable
        }
        guard await coordinator.hasLoadedAudio, editorService.loadedAudio != nil else {
            throw PiMPAudioCommandFacadeError.editorAudioNotLoaded
        }
        if FileManager.default.fileExists(atPath: request.outputURL.path(percentEncoded: false)),
           !request.overwrite {
            throw PiMPAudioCommandFacadeError.editorRenderFileExists(
                request.outputURL.path(percentEncoded: false)
            )
        }

        let operation = registerOperation(
            kind: "editor.renderAudio",
            title: request.selection == nil ? "Render editor audio" : "Render editor selection",
            request: requestPayload,
            progress: editorRenderProgressPayload(stage: "queued", message: "Editor render accepted.")
        )

        operationTasks[operation.id] = Task { [weak self] in
            await self?.runEditorRenderOperation(id: operation.id, request: request)
        }

        return [
            "submitted": true,
            "dryRun": false,
            "request": requestPayload,
            "operationID": operation.id,
            "operation": operationPayload(operation)
        ]
    }

    func applyEditorEffectChain(arguments: [String: Any]) async throws -> [String: Any] {
        let dryRun = boolValue(arguments["dryRun"] ?? arguments["validateOnly"]) == true
        let loadedAudio = editorService?.loadedAudio
        let request = try editorDSPRequest(arguments, loadedAudio: loadedAudio)
        let requestPayload = editorDSPRequestPayload(request)

        if dryRun {
            return [
                "submitted": false,
                "dryRun": true,
                "requiresLoadedAudio": true,
                "requiresConfirmation": true,
                "confirmation": confirmationRequirementPayload(toolName: "apply_editor_effect_chain"),
                "request": requestPayload,
                "operationID": NSNull(),
                "operation": NSNull()
            ]
        }

        guard boolValue(arguments["confirm"]) == true else {
            throw PiMPAudioCommandFacadeError.confirmationRequired("apply_editor_effect_chain")
        }
        guard let editorService, let coordinator = editorService.coordinator else {
            throw PiMPAudioCommandFacadeError.editorServiceUnavailable
        }
        guard await coordinator.hasLoadedAudio, editorService.loadedAudio != nil else {
            throw PiMPAudioCommandFacadeError.editorAudioNotLoaded
        }

        let operation = registerOperation(
            kind: "editor.applyEffectChain",
            title: request.steps.count == 1 ? "Apply editor effect" : "Apply editor effect chain",
            request: requestPayload,
            progress: editorDSPProgressPayload(stage: "queued", message: "Editor DSP chain accepted.")
        )

        operationTasks[operation.id] = Task { [weak self] in
            await self?.runEditorDSPOperation(id: operation.id, request: request)
        }

        return [
            "submitted": true,
            "dryRun": false,
            "request": requestPayload,
            "operationID": operation.id,
            "operation": operationPayload(operation)
        ]
    }

    func listOperations(arguments: [String: Any]) -> [String: Any] {
        refreshGenerationOperation()
        let requestedLimit = doubleValue(arguments["limit"]).map { Int($0) } ?? 20
        let cappedLimit = max(1, min(requestedLimit, maxOperationRecords))
        let operations = operationSnapshot(limit: cappedLimit).map(operationPayload)
        return [
            "count": operations.count,
            "currentOperationID": activeGenerationOperationID.map { $0 as Any } ?? NSNull(),
            "activeOperationIDs": activeOperationIDsPayload(),
            "operations": operations
        ]
    }

    func getOperation(arguments: [String: Any]) throws -> [String: Any] {
        refreshGenerationOperation()
        let id = try operationID(from: arguments)
        guard let record = operationRecords[id] else {
            throw PiMPAudioCommandFacadeError.operationNotFound(id)
        }
        return operationPayload(record)
    }

    func cancelOperation(arguments: [String: Any]) throws -> [String: Any] {
        refreshGenerationOperation()
        let id = try operationID(from: arguments)
        guard var record = operationRecords[id] else {
            throw PiMPAudioCommandFacadeError.operationNotFound(id)
        }

        let reason = stringValue(arguments["reason"])?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let canCancel = isOperationCancelable(record)
        guard canCancel else {
            return [
                "cancelled": false,
                "reason": "operationNotActive",
                "operation": operationPayload(record)
            ]
        }

        if record.kind == "generation", id == activeGenerationOperationID {
            generationService?.cancel()
            activeGenerationOperationID = nil
        }
        operationTasks[id]?.cancel()
        operationTasks[id] = nil

        record.status = .cancelled
        record.updatedAt = Date()
        record.cancellationReason = reason?.isEmpty == false
            ? reason
            : "Cancelled through MCP cancel_operation."
        record.progress = cancellationProgressPayload(for: record)
        operationRecords[id] = record

        return [
            "cancelled": true,
            "operation": operationPayload(record)
        ]
    }

    func releaseMemory(arguments: [String: Any]) async throws -> [String: Any] {
        guard boolValue(arguments["confirm"]) == true else {
            throw PiMPAudioCommandFacadeError.confirmationRequired("release_memory")
        }
        guard let generationService else {
            throw PiMPAudioCommandFacadeError.generationServiceUnavailable
        }

        let summary = await generationService.unloadAllWeights()
        return memoryReleasePayload(summary)
    }

    private func loadAdapters() async -> [LoRAAdapter] {
        await Task.detached(priority: .userInitiated) {
            AdapterRegistry.listAdapters()
        }.value
    }

    private func parseAdapterSelection(
        _ arguments: [String: Any],
        installed: [LoRAAdapter]
    ) throws -> [AdapterSelection] {
        var selection: [AdapterSelection] = []

        if let rawStack = arguments["adapters"] ?? arguments["adapterStack"] ?? arguments["stack"] {
            let items: [[String: Any]]
            if let typed = rawStack as? [[String: Any]] {
                items = typed
            } else if let rawItems = rawStack as? [Any] {
                items = rawItems.compactMap { $0 as? [String: Any] }
            } else {
                throw PiMPAudioCommandFacadeError.invalidAdapterStack
            }

            for item in items {
                let identifier = stringValue(item["id"] ?? item["adapterID"] ?? item["name"])?
                    .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
                guard !identifier.isEmpty else {
                    throw PiMPAudioCommandFacadeError.missingRequiredArgument("id")
                }
                let adapter = try resolveAdapter(identifier, installed: installed)
                let strength = try adapterStrength(item["strength"] ?? item["weight"] ?? item["scale"])
                selection.append(AdapterSelection(adapter: adapter, strength: strength))
            }
        } else {
            let identifier = stringValue(arguments["id"] ?? arguments["adapterID"] ?? arguments["name"])?
                .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            guard !identifier.isEmpty else {
                throw PiMPAudioCommandFacadeError.missingRequiredArgument("id")
            }
            let adapter = try resolveAdapter(identifier, installed: installed)
            let strength = try adapterStrength(arguments["strength"] ?? arguments["weight"] ?? arguments["scale"])
            selection.append(AdapterSelection(adapter: adapter, strength: strength))
        }

        var unique: [AdapterSelection] = []
        var seen = Set<String>()
        for entry in selection where seen.insert(entry.adapter.id).inserted {
            unique.append(entry)
        }

        guard !unique.isEmpty else {
            throw PiMPAudioCommandFacadeError.missingRequiredArgument("id")
        }
        guard unique.count <= AdapterStackManager.maxStackSize else {
            throw PiMPAudioCommandFacadeError.tooManyAdapters(AdapterStackManager.maxStackSize)
        }
        for entry in unique where entry.adapter.baseModelCategory != .aceStepDiT {
            throw PiMPAudioCommandFacadeError.incompatibleAdapter(
                entry.adapter.name,
                entry.adapter.baseModelCategory.rawValue
            )
        }

        return unique
    }

    private func resolveAdapter(_ identifier: String, installed: [LoRAAdapter]) throws -> LoRAAdapter {
        let needle = identifier.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if let exactID = installed.first(where: { $0.id.lowercased() == needle }) {
            return exactID
        }
        if let exactName = installed.first(where: { $0.name.lowercased() == needle }) {
            return exactName
        }
        if let partial = installed.first(where: {
            $0.id.lowercased().contains(needle) || $0.name.lowercased().contains(needle)
        }) {
            return partial
        }
        throw PiMPAudioCommandFacadeError.adapterNotFound(identifier)
    }

    private func adapterStrength(_ value: Any?) throws -> Float {
        guard let value else { return 1.0 }
        guard let parsed = doubleValue(value), parsed >= 0.0, parsed <= 1.0 else {
            throw PiMPAudioCommandFacadeError.invalidStrength
        }
        return Float(parsed)
    }

    private func syncGenerationAdapterSelectionFromDefaultsIfNeeded(_ generationService: GenerationService) {
        guard generationService.adapterStack.isEmpty else { return }

        let defaults = UserDefaults.standard
        let adapterID = defaults.string(forKey: AppUserDefaultsKeys.selectedAdapterID)?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        guard !adapterID.isEmpty else {
            generationService.selectedAdapterID = nil
            generationService.adapterStack = []
            return
        }

        let adapters = AdapterRegistry.listAdapters()
        guard adapters.contains(where: { $0.id == adapterID }) else {
            generationService.selectedAdapterID = nil
            generationService.adapterStack = []
            return
        }

        let rawStrength = doubleValue(defaults.object(forKey: AppUserDefaultsKeys.selectedAdapterStrength)) ?? 1.0
        let strength = Float(min(max(rawStrength, 0.0), 1.0))
        generationService.selectedAdapterID = adapterID
        generationService.adapterStack = [(id: adapterID, weight: strength)]
    }

    private func orchestrationAdapterArguments(from arguments: [String: Any]) -> [String: Any]? {
        if boolValue(arguments["clearAdapter"] ?? arguments["clearAdapters"]) == true {
            return ["clear": true]
        }

        if let adapters = arguments["adapters"] ?? arguments["adapterStack"] ?? arguments["stack"] {
            return ["adapters": adapters]
        }

        guard let identifier = stringValue(
            arguments["adapter"] ?? arguments["adapterID"] ?? arguments["lora"] ?? arguments["loraID"]
        )?.trimmingCharacters(in: .whitespacesAndNewlines),
              !identifier.isEmpty else {
            return nil
        }

        var adapterArguments: [String: Any] = ["id": identifier]
        if let strength = arguments["strength"] ?? arguments["weight"] ?? arguments["scale"] {
            adapterArguments["strength"] = strength
        }
        return adapterArguments
    }

    private func orchestrationDuration(in arguments: [String: Any]) throws -> Double {
        let duration = doubleValue(arguments["duration"] ?? arguments["durationSeconds"]) ?? 30.0
        guard duration > 0 else {
            throw PiMPAudioCommandFacadeError.invalidDuration
        }
        return duration
    }

    private func generationRequest(_ arguments: [String: Any]) throws -> GenerationRequest {
        let prompt = try requiredString("prompt", aliases: ["text"], in: arguments)
        let lyrics = stringValue(arguments["lyrics"]) ?? ""
        let mode = generationMode(from: arguments["mode"] ?? arguments["generationMode"]) ?? .music
        let pipeline = pipelineChoice(from: arguments["pipeline"] ?? arguments["pipelineOverride"]) ?? .auto
        let requestedDuration = doubleValue(arguments["duration"] ?? arguments["durationSeconds"]) ?? 30.0
        guard requestedDuration > 0 else {
            throw PiMPAudioCommandFacadeError.invalidDuration
        }

        let duration: Double
        if pipeline == .stableAudio || (pipeline == .auto && (mode == .soundFX || mode == .ambient)) {
            duration = min(requestedDuration, 47.0)
        } else {
            duration = requestedDuration
        }

        return GenerationRequest(
            text: prompt,
            lyrics: lyrics,
            referenceAudioURL: referenceAudioURL(from: arguments["referenceAudioPath"] ?? arguments["referenceAudioURL"]),
            duration: duration,
            seed: try seedValue(arguments["seed"]),
            method: inferMethod(from: arguments["method"]) ?? .ode,
            guidanceScale: Float(doubleValue(arguments["guidanceScale"] ?? arguments["guidance"]) ?? 7.5),
            guidancePreset: guidancePreset(from: arguments["guidancePreset"] ?? arguments["preset"]) ?? .defaultSelection,
            precision: precision(from: arguments["precision"]) ?? .float32,
            stepsOverride: try stepsValue(arguments["steps"] ?? arguments["stepsOverride"]),
            shiftOverride: Float(doubleValue(arguments["shift"] ?? arguments["shiftOverride"]) ?? 3.0),
            resolvedMode: mode,
            pipelineOverride: pipeline,
            coverStrength: Float(min(max(doubleValue(arguments["coverStrength"]) ?? 0.7, 0.0), 1.0))
        )
    }

    private func referenceAudioURL(from value: Any?) -> URL? {
        guard let path = stringValue(value)?.trimmingCharacters(in: .whitespacesAndNewlines),
              !path.isEmpty else {
            return nil
        }
        if let url = URL(string: path), url.scheme != nil {
            return url
        }
        return URL(fileURLWithPath: NSString(string: path).expandingTildeInPath)
    }

    private func seedValue(_ value: Any?) throws -> Int? {
        guard let value else { return nil }
        if value is NSNull { return nil }
        if let int = value as? Int { return int }
        if let number = value as? NSNumber { return number.intValue }
        if let string = value as? String {
            let trimmed = string.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.isEmpty || trimmed.lowercased() == "random" {
                return nil
            }
            guard let parsed = Int(trimmed) else {
                throw PiMPAudioCommandFacadeError.invalidSeed
            }
            return parsed
        }
        throw PiMPAudioCommandFacadeError.invalidSeed
    }

    private func stepsValue(_ value: Any?) throws -> Int? {
        guard let value else { return nil }
        if value is NSNull { return nil }
        if let int = value as? Int, int > 0 { return int }
        if let number = value as? NSNumber, number.intValue > 0 { return number.intValue }
        if let string = value as? String {
            let trimmed = string.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.isEmpty || trimmed.lowercased() == "auto" {
                return nil
            }
            guard let parsed = Int(trimmed), parsed > 0 else {
                throw PiMPAudioCommandFacadeError.invalidSteps
            }
            return parsed
        }
        throw PiMPAudioCommandFacadeError.invalidSteps
    }

    private func modelResource(id: String) async throws -> [String: Any] {
        let specs = await registry.allSpecs
        guard let spec = specs.first(where: { $0.id == id }) else {
            throw PiMPAudioCommandFacadeError.resourceNotFound(resourceURI(kind: "model", identifier: id))
        }

        let activeID = await registry.activeModelID(for: spec.category)
        return [
            "id": spec.id,
            "displayName": spec.displayName,
            "category": spec.category.rawValue,
            "family": spec.variantFamily ?? NSNull(),
            "huggingFaceRepo": spec.huggingFaceRepo,
            "files": spec.files.map { file in
                [
                    "remotePath": file.remotePath,
                    "localName": file.localName,
                    "sizeBytes": file.sizeBytes,
                    "sha256": file.sha256 ?? NSNull()
                ] as [String: Any]
            },
            "totalSizeBytes": spec.totalSizeBytes,
            "totalSizeFormatted": spec.totalSizeFormatted,
            "requiresAuth": spec.requiresAuth,
            "isRunnable": spec.isRunnable,
            "runtimeNote": spec.runtimeNote ?? NSNull(),
            "status": await registry.status(for: spec.id).mcpDescription,
            "isActive": activeID == spec.id
        ]
    }

    private func adapterResource(id: String) async throws -> [String: Any] {
        guard let adapter = await loadAdapters().first(where: { $0.id == id }) else {
            throw PiMPAudioCommandFacadeError.resourceNotFound(resourceURI(kind: "adapter", identifier: id))
        }

        return [
            "id": adapter.id,
            "name": adapter.name,
            "baseModelID": adapter.baseModelID,
            "baseModelCategory": adapter.baseModelCategory.rawValue,
            "rank": adapter.config.rank,
            "alpha": adapter.config.alpha,
            "dropout": adapter.config.dropout,
            "targetLayers": adapter.config.targetLayers.map(\.rawValue).sorted(),
            "injectionPatterns": adapter.config.injectionPatterns,
            "learningRate": adapter.config.learningRate,
            "maxEpochs": adapter.config.maxEpochs,
            "sampleCount": adapter.sampleCount,
            "finalLoss": adapter.finalLoss.map { Double($0) } ?? NSNull(),
            "createdAt": dateFormatter.string(from: adapter.createdAt),
            "source": adapter.source.rawValue,
            "remoteRepoURL": adapter.remoteRepoURL?.absoluteString ?? NSNull(),
            "hasCheckpoint": AdapterRegistry.hasCheckpoint(adapterID: adapter.id)
        ]
    }

    private func audioResource(identifier: String) throws -> [String: Any] {
        let entry = try resolveHistoryEntry(identifier)
        let audioURL = historyStore.fileURL(for: entry)
        return [
            "id": entry.id.uuidString,
            "fileName": entry.fileName,
            "audioFilePath": audioURL.path(percentEncoded: false),
            "exists": FileManager.default.fileExists(atPath: audioURL.path(percentEncoded: false)),
            "duration": entry.duration,
            "createdAt": dateFormatter.string(from: entry.createdAt),
            "pipeline": entry.pipeline ?? NSNull(),
            "modelID": entry.modelID ?? NSNull(),
            "modelName": entry.modelName ?? NSNull(),
            "traceFileName": entry.traceFileName ?? NSNull(),
            "hasTrace": historyStore.traceURL(for: entry) != nil,
            "isPreferred": entry.isPreferred ?? false,
            "listeningNote": entry.listeningNote ?? NSNull()
        ]
    }

    private func validateResourceIdentifier(_ identifier: String?, expected: String, uri: String) throws {
        guard identifier == expected else {
            throw PiMPAudioCommandFacadeError.invalidResourceURI(uri)
        }
    }

    private func projectResource() async throws -> [String: Any] {
        refreshGenerationOperation()

        let editorPayload: Any
        do {
            editorPayload = try await editorStatus()
        } catch {
            editorPayload = [
                "ready": false,
                "error": error.localizedDescription
            ]
        }

        return [
            "id": "current",
            "kind": "project",
            "name": "PiMP Audio Studio Local Workstation",
            "scope": "local",
            "resourceURI": resourceURI(kind: "project", identifier: "current"),
            "resources": [
                "models": "pimpaudio://models",
                "adapters": "pimpaudio://adapters",
                "training": "pimpaudio://training",
                "trainingDatasets": "pimpaudio://training-datasets",
                "trainingRuns": "pimpaudio://training-runs",
                "history": "pimpaudio://history",
                "takes": "pimpaudio://takes",
                "track": resourceURI(kind: "track", identifier: "current"),
                "clip": resourceURI(kind: "clip", identifier: "current"),
                "region": resourceURI(kind: "region", identifier: "current-selection"),
                "presets": "pimpaudio://presets",
                "workflows": "pimpaudio://workflows",
                "editor": "pimpaudio://editor",
                "operations": "pimpaudio://operations"
            ],
            "counts": [
                "history": historyStore.entries.count,
                "takes": historyStore.entries.count,
                "trainingDatasets": (listTrainingDatasets(arguments: ["limit": 200])["datasets"] as? [[String: Any]])?.count ?? 0,
                "trainingRuns": (listTrainingRuns(arguments: ["limit": 100])["runs"] as? [[String: Any]])?.count ?? 0,
                "presets": loadGenerationPresets().count,
                "workflows": workflowPromptDefinitions.count,
                "operations": operationSnapshot(limit: maxOperationRecords).count
            ],
            "training": trainingStatus(),
            "editor": editorPayload,
            "operations": listOperations(arguments: ["limit": 10])
        ]
    }

    private func trainingStatePayload(_ state: TrainingState) -> [String: Any] {
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

    private func trainingUIStatePayload(_ uiState: TrainingUIState) -> [String: Any] {
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
            "elapsedSeconds": uiState.elapsed,
            "remainingSeconds": uiState.remaining.map { $0 as Any } ?? NSNull(),
            "lossHistoryCount": uiState.lossHistory.count,
            "latestLossHistory": Array(uiState.lossHistory.suffix(25)).map(Double.init)
        ]
    }

    private func trainingCheckpointPayload(_ checkpoint: TrainingCheckpoint) -> [String: Any] {
        [
            "epoch": checkpoint.epoch,
            "globalStep": checkpoint.globalStep,
            "totalSteps": checkpoint.totalSteps,
            "bestLoss": checkpoint.bestLoss.map { Double($0) } ?? NSNull(),
            "lossHistoryCount": checkpoint.lossHistory.count,
            "weightsPath": checkpoint.weightsURL.path(percentEncoded: false),
            "optimizerStatePath": checkpoint.optimizerStateURL.path(percentEncoded: false),
            "baseModelID": checkpoint.baseModelID
        ]
    }

    private func trainingAdapterSummaryPayload(_ adapter: LoRAAdapter) -> [String: Any] {
        [
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
            "directory": AdapterRegistry.adapterDirectory(id: adapter.id).path(percentEncoded: false)
        ]
    }

    private func trainingDatasetSearchRoots() -> [URL] {
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

    private func trainingDatasetPayload(_ summary: DatasetPackageSummary) -> [String: Any] {
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
            "trainingBundlePath": summary.directory
                .appendingPathComponent(TrainingDatasetPackageWriter.fullMixBundleFileName)
                .path(percentEncoded: false),
            "remotePackagePath": summary.directory
                .appendingPathComponent(TrainingDatasetPackageWriter.remotePackageFileName)
                .path(percentEncoded: false)
        ]
    }

    private func resolveMCPTrainingRequest(arguments: [String: Any]) throws -> MCPTrainingRequest {
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
                trainingLayerCount: layerCount
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

    private func mcpTrainingDatasetArguments(_ arguments: [String: Any]) -> [String: Any] {
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

    private func mcpTrainingKind(from value: Any?) throws -> AdapterKind {
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

    private func mcpTrainingRecipe(from value: Any?, kind: AdapterKind) throws -> MCPTrainingRecipe {
        guard let raw = stringValue(value)?.trimmingCharacters(in: .whitespacesAndNewlines),
              !raw.isEmpty else {
            return .local
        }

        switch compactTrainingToken(raw) {
        case "local", "balanced", "balancedlocal":
            return .local
        case "cloudstyle", "cloudstylequality", "style", "stylequality":
            guard kind == .lora else {
                throw PiMPAudioCommandFacadeError.invalidTrainingRecipe(raw)
            }
            return .cloudStyle
        case "cloudvoice", "cloudvoicebalanced", "voice", "voicebalanced":
            guard kind == .lora else {
                throw PiMPAudioCommandFacadeError.invalidTrainingRecipe(raw)
            }
            return .cloudVoice
        case "wide", "expanded", "expandedlokr":
            guard kind == .lokr else {
                throw PiMPAudioCommandFacadeError.invalidTrainingRecipe(raw)
            }
            return .wide
        default:
            throw PiMPAudioCommandFacadeError.invalidTrainingRecipe(raw)
        }
    }

    private func mcpTrainingDefaults(
        recipe: MCPTrainingRecipe,
        kind: AdapterKind,
        recommendation: TrainingPlanRecommendation?
    ) -> MCPTrainingDefaults {
        switch recipe {
        case .local:
            return MCPTrainingDefaults(
                dimension: recommendation?.rank ?? 16,
                epochs: recommendation?.epochs ?? 10,
                layerCount: recommendation?.layerCount ?? 8,
                projectionCoverage: .queryValue,
                alphaScale: 1,
                dropout: 0,
                targetLayers: [.selfAttention, .crossAttention],
                gradientAccumulationSteps: recommendation?.gradientAccumulationSteps ?? 1,
                learningRate: 0.0001
            )
        case .cloudStyle:
            return MCPTrainingDefaults(
                dimension: 96,
                epochs: 150,
                layerCount: 32,
                projectionCoverage: .fullAttention,
                alphaScale: 2,
                dropout: 0.08,
                targetLayers: [.selfAttention, .crossAttention],
                gradientAccumulationSteps: 4,
                learningRate: 0.000075
            )
        case .cloudVoice:
            return MCPTrainingDefaults(
                dimension: 64,
                epochs: 100,
                layerCount: 32,
                projectionCoverage: .fullAttention,
                alphaScale: 2,
                dropout: 0.10,
                targetLayers: [.selfAttention, .crossAttention],
                gradientAccumulationSteps: 4,
                learningRate: 0.0001
            )
        case .wide:
            return MCPTrainingDefaults(
                dimension: kind == .lokr ? 32 : 64,
                epochs: 100,
                layerCount: kind == .lokr ? 16 : 32,
                projectionCoverage: kind == .lokr ? .queryValue : .fullAttention,
                alphaScale: 2,
                dropout: 0.08,
                targetLayers: [.selfAttention, .crossAttention],
                gradientAccumulationSteps: 4,
                learningRate: 0.0001
            )
        }
    }

    private func boundedTrainingInt(
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

    private func boundedTrainingLayerCount(_ value: Any?, defaultValue: Int) throws -> Int {
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

    private func boundedTrainingFloat(
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

    private func mcpTrainingAlpha(
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

    private func mcpTrainingTargetLayers(
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

    private func mcpTrainingInjectionPatterns(
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

    private func mcpTrainingAdapterName(
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

    private func trainingPreflightResponsePayload(
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

    private func trainingLocalPreflightPayload(_ report: TrainingLocalPreflightReport) -> [String: Any] {
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

    private func trainingStateIsActive(_ state: TrainingState) -> Bool {
        switch state {
        case .loadingModels, .preparingDataset, .training, .saving:
            return true
        case .idle, .complete, .failed, .cancelled, .paused:
            return false
        }
    }

    private func mcpTrainingKindTitle(_ kind: AdapterKind) -> String {
        switch kind {
        case .lora:
            return "LoRA"
        case .lokr:
            return "LoKR"
        }
    }

    private func compactTrainingToken(_ raw: String) -> String {
        raw
            .lowercased()
            .replacingOccurrences(of: " ", with: "")
            .replacingOccurrences(of: "-", with: "")
            .replacingOccurrences(of: "_", with: "")
            .replacingOccurrences(of: "/", with: "")
    }

    private func resolveTrainingDatasetDirectory(arguments: [String: Any]) throws -> URL {
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

    private func trainingDatasetDetailPayload(
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

    private func trainingDatasetValidationPayload(_ validation: TrainingDatasetPackageValidationResult) -> [String: Any] {
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

    private func trainingDatasetProfile(for document: DatasetPackageDocument) -> TrainingDatasetProfile {
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

    private func trainingDatasetProfilePayload(_ profile: TrainingDatasetProfile) -> [String: Any] {
        [
            "sampleCount": profile.sampleCount,
            "knownDurationSeconds": profile.knownDuration,
            "knownDurationMinutes": profile.knownDuration / 60,
            "knownDurationSampleCount": profile.knownDurationSampleCount,
            "durationCoverage": profile.durationCoverage,
            "hasUsableDurationCoverage": profile.hasUsableDurationCoverage
        ]
    }

    private func trainingDatasetAnnotationStatsPayload(_ document: DatasetPackageDocument) -> [String: Any] {
        var captionCount = 0
        var lyricsCount = 0
        var durationCount = 0
        var bpmCount = 0
        var keyCount = 0
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
            "captionSources": captionSources,
            "lyricsSources": lyricsSources,
            "transcriptionStatuses": transcriptionStatuses,
            "audioStatuses": audioStatuses
        ]
    }

    private func trainingDatasetClipPayload(_ clip: DatasetClipReview) -> [String: Any] {
        [
            "id": clip.id,
            "file": clip.file,
            "displayName": clip.displayName,
            "caption": compact(clip.caption, limit: 220),
            "lyricsLineCount": lyricLineCount(clip.lyrics),
            "durationSeconds": clip.durationSeconds ?? NSNull(),
            "bpm": clip.bpm.isEmpty ? NSNull() : clip.bpm,
            "key": clip.key.isEmpty ? NSNull() : clip.key,
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

    private func trainingDatasetPathsPayload(_ directory: URL) -> [String: Any] {
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

    private func trainingPlanRecommendationPayload(_ recommendation: TrainingPlanRecommendation) -> [String: Any] {
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

    private func trainingRunPayload(adapter: LoRAAdapter, log: TrainingRunLog?) -> [String: Any] {
        var payload = trainingAdapterSummaryPayload(adapter)
        payload["runLogPath"] = AdapterRegistry.trainingRunLogURL(adapterID: adapter.id).path(percentEncoded: false)
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

    private func resolveTrainingRunAdapter(arguments: [String: Any]) throws -> LoRAAdapter {
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

    private func resolveTrainingRunComparisonAdapters(
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

    private func trainingRunUpdatedAt(_ adapter: LoRAAdapter) -> Date {
        AdapterRegistry.loadTrainingRunLog(adapterID: adapter.id)?.updatedAt ?? adapter.createdAt
    }

    private func trainingRunDetailPayload(
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

    private func trainingRunComparisonPayload(adapter: LoRAAdapter, log: TrainingRunLog?) -> [String: Any] {
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
            "latestUpdateNorm": health["latestUpdateNorm"] ?? NSNull(),
            "zeroUpdateNormCount": health["zeroUpdateNormCount"] ?? NSNull(),
            "progressCount": log?.progress.count ?? 0,
            "hasCheckpoint": AdapterRegistry.hasCheckpoint(adapterID: adapter.id)
        ]
    }

    private func trainingAdapterConfigurationPayload(_ adapter: LoRAAdapter) -> [String: Any] {
        [
            "kind": adapter.kind.rawValue,
            "lora": loraConfigPayload(adapter.config),
            "lokr": adapter.lokrConfig.map(lokrConfigPayload) ?? NSNull(),
            "effectiveInjectionPatterns": adapter.lokrConfig?.injectionPatterns ?? adapter.config.injectionPatterns,
            "trainedPaths": adapter.trainedPaths ?? [],
            "trainedLayerIndices": adapter.trainedLayerIndices ?? []
        ]
    }

    private func loraConfigPayload(_ config: LoRAConfig) -> [String: Any] {
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
            "trainingLayerCount": config.trainingLayerCount ?? NSNull()
        ]
    }

    private func lokrConfigPayload(_ config: LoKRConfig) -> [String: Any] {
        [
            "linearDim": config.linearDim,
            "alpha": Double(config.alpha),
            "scale": Double(config.alpha) / Double(max(1, config.linearDim)),
            "dropout": Double(config.dropout),
            "targetLayers": targetLayerPayload(config.targetLayers),
            "exactInjectionPatterns": config.exactInjectionPatterns ?? NSNull(),
            "injectionPatterns": config.injectionPatterns,
            "learningRate": Double(config.learningRate),
            "maxEpochs": config.maxEpochs,
            "batchSize": config.batchSize,
            "warmupFraction": Double(config.warmupFraction),
            "gradientAccumulationSteps": config.gradientAccumulationSteps,
            "trainingLayerCount": config.trainingLayerCount ?? NSNull()
        ]
    }

    private func targetLayerPayload(_ layers: Set<TargetLayer>) -> [String] {
        layers.map(\.rawValue).sorted()
    }

    private func trainingRunPathsPayload(_ adapter: LoRAAdapter) -> [String: Any] {
        let directory = AdapterRegistry.adapterDirectory(id: adapter.id)
        let logURL = AdapterRegistry.trainingRunLogURL(adapterID: adapter.id)
        let checkpointURL = AdapterRegistry.checkpointURL(adapterID: adapter.id)
        let optimizerURL = AdapterRegistry.optimizerStateURL(adapterID: adapter.id)
        let weightsURL = AdapterRegistry.weightsURL(adapterID: adapter.id)
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

    private func trainingRunHealthPayload(adapter: LoRAAdapter, log: TrainingRunLog?) -> [String: Any] {
        guard let log else {
            return [
                "hasRunLog": false,
                "finalLoss": adapter.finalLoss.map { Double($0) } ?? NSNull(),
                "progressCount": 0
            ]
        }

        let losses = log.progress.map { Double($0.loss) }
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
            "eventCount": log.events.count,
            "firstLoss": firstLoss ?? NSNull(),
            "latestLoss": latestLoss ?? NSNull(),
            "bestLoss": bestLoss ?? NSNull(),
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

    private func trainingRunLogPayload(_ log: TrainingRunLog) -> [String: Any] {
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
            "events": log.events.suffix(10).map(trainingRunEventPayload),
            "finalLoss": log.finalLoss.map { Double($0) } ?? NSNull(),
            "trainedPaths": log.trainedPaths,
            "trainedLayerIndices": log.trainedLayerIndices,
            "weightsTensorCount": log.weightsTensorCount ?? NSNull(),
            "errorMessage": log.errorMessage ?? NSNull()
        ]
    }

    private func trainingRunProgressPayload(_ progress: TrainingRunProgressEntry) -> [String: Any] {
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
            "elapsedSeconds": progress.elapsed,
            "estimatedRemainingSeconds": progress.estimatedRemaining.map { $0 as Any } ?? NSNull()
        ]
    }

    private func trainingRunEventPayload(_ event: TrainingRunEvent) -> [String: Any] {
        [
            "timestamp": dateFormatter.string(from: event.timestamp),
            "status": event.status.rawValue,
            "message": event.message
        ]
    }

    private func currentTrackResource() async -> [String: Any] {
        let loadedAudio = editorService?.loadedAudio
        let hasLoadedAudio = await editorService?.coordinator?.hasLoadedAudio ?? false

        return [
            "id": "current",
            "kind": "track",
            "projectURI": resourceURI(kind: "project", identifier: "current"),
            "clipURI": resourceURI(kind: "clip", identifier: "current"),
            "loaded": hasLoadedAudio,
            "duration": loadedAudio?.duration ?? NSNull(),
            "sampleRate": loadedAudio?.sampleRate ?? NSNull(),
            "channels": loadedAudio?.channels ?? NSNull(),
            "sourceLoadedAudio": loadedAudio.map(editorLoadedAudioPayload) ?? NSNull()
        ]
    }

    private func currentClipResource() async -> [String: Any] {
        let loadedAudio = editorService?.loadedAudio
        let hasLoadedAudio = await editorService?.coordinator?.hasLoadedAudio ?? false
        let historyID = loadedAudio?.historyEntryID

        return [
            "id": "current",
            "kind": "clip",
            "projectURI": resourceURI(kind: "project", identifier: "current"),
            "trackURI": resourceURI(kind: "track", identifier: "current"),
            "loaded": hasLoadedAudio,
            "startTime": loadedAudio == nil ? NSNull() : 0,
            "endTime": loadedAudio?.duration ?? NSNull(),
            "duration": loadedAudio?.duration ?? NSNull(),
            "fileName": loadedAudio?.fileName ?? NSNull(),
            "path": loadedAudio?.path ?? NSNull(),
            "historyResourceURI": historyID.map { resourceURI(kind: "history", identifier: $0) as Any } ?? NSNull(),
            "takeResourceURI": historyID.map { resourceURI(kind: "take", identifier: $0) as Any } ?? NSNull(),
            "audioResourceURI": historyID.map { resourceURI(kind: "audio", identifier: $0) as Any } ?? NSNull(),
            "sourceLoadedAudio": loadedAudio.map(editorLoadedAudioPayload) ?? NSNull()
        ]
    }

    private func currentRegionResource() async -> [String: Any] {
        let loadedAudio = editorService?.loadedAudio
        let hasLoadedAudio = await editorService?.coordinator?.hasLoadedAudio ?? false
        let lastEdit = editorService?.lastEdit
        let selection: Any

        if let startTime = lastEdit?.selectionStart,
           let endTime = lastEdit?.selectionEnd {
            selection = [
                "startTime": startTime,
                "endTime": endTime,
                "duration": endTime - startTime,
                "frequencyBand": editorFrequencyBandPayload(
                    lowCutHz: lastEdit?.lowCutHz,
                    highCutHz: lastEdit?.highCutHz
                )
            ]
        } else {
            selection = NSNull()
        }

        return [
            "id": "current-selection",
            "kind": "region",
            "projectURI": resourceURI(kind: "project", identifier: "current"),
            "trackURI": resourceURI(kind: "track", identifier: "current"),
            "clipURI": resourceURI(kind: "clip", identifier: "current"),
            "loaded": hasLoadedAudio,
            "clipDuration": loadedAudio?.duration ?? NSNull(),
            "selection": selection,
            "lastEdit": lastEdit.map(editorEditSummaryPayload) ?? NSNull(),
            "note": "SwiftUI drag selection is still local view state; this reports the most recent persisted editor edit selection."
        ]
    }

    private func takesResource(limit: Int) -> [String: Any] {
        let cappedLimit = max(1, min(limit, 200))
        let rows = historyStore.entries.prefix(cappedLimit).map(takeSummary)
        return [
            "kind": "takeCollection",
            "count": rows.count,
            "totalCount": historyStore.entries.count,
            "takes": Array(rows)
        ]
    }

    private func takeResource(identifier: String) throws -> [String: Any] {
        let entry = try resolveHistoryEntry(identifier)
        let traceURI: Any = historyStore.traceURL(for: entry) == nil
            ? NSNull()
            : resourceURI(kind: "trace", identifier: entry.id.uuidString)

        return [
            "id": entry.id.uuidString,
            "kind": "take",
            "summary": takeSummary(entry),
            "history": historySummary(entry),
            "audio": try audioResource(identifier: entry.id.uuidString),
            "adapters": (entry.adapters ?? []).map(historyAdapterPayload),
            "resources": [
                "history": resourceURI(kind: "history", identifier: entry.id.uuidString),
                "take": resourceURI(kind: "take", identifier: entry.id.uuidString),
                "audio": resourceURI(kind: "audio", identifier: entry.id.uuidString),
                "trace": traceURI
            ]
        ]
    }

    private func presetsResource() -> [String: Any] {
        let savedPresets = loadGenerationPresets()
        return [
            "kind": "presetCollection",
            "count": savedPresets.count,
            "savedGenerationPresets": savedPresets.map(generationPresetPayload),
            "generationModes": GenerationMode.allCases.map { mode in
                [
                    "id": mode.rawValue,
                    "displayName": mode.displayName
                ]
            },
            "pipelines": PipelineChoice.allCases.map { pipeline in
                [
                    "id": pipeline.rawValue,
                    "displayName": pipeline.displayName
                ]
            },
            "methods": [
                ["id": InferMethod.ode.rawValue],
                ["id": InferMethod.sde.rawValue]
            ],
            "precisions": Precision.allCases.map { ["id": $0.rawValue] },
            "guidancePresets": GuidancePreset.allCases.map(guidancePresetPayload),
            "inpaintPresets": InpaintPreset.allCases.map(inpaintPresetPayload),
            "reverbPresets": ReverbPreset.allCases.map(reverbPresetPayload),
            "dspEffects": DSPEffect.allCases.map(dspEffectPayload)
        ]
    }

    private func presetResource(identifier: String) throws -> [String: Any] {
        let normalized = identifier.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard let preset = loadGenerationPresets().first(where: {
            $0.id.uuidString.lowercased() == normalized || $0.name.lowercased() == normalized
        }) else {
            throw PiMPAudioCommandFacadeError.resourceNotFound(resourceURI(kind: "preset", identifier: identifier))
        }
        return generationPresetPayload(preset)
    }

    private func workflowsResource() -> [String: Any] {
        [
            "kind": "workflowCollection",
            "count": workflowPromptDefinitions.count,
            "workflows": workflowPromptDefinitions.map(workflowPromptPayload)
        ]
    }

    private func workflowResource(identifier: String) throws -> [String: Any] {
        guard let workflow = workflowPromptDefinitions.first(where: { $0.id == identifier }) else {
            throw PiMPAudioCommandFacadeError.resourceNotFound(resourceURI(kind: "workflow", identifier: identifier))
        }
        var payload = workflowPromptPayload(workflow)
        payload["template"] = workflow.template
        return payload
    }

    private func takeSummary(_ entry: HistoryEntry) -> [String: Any] {
        [
            "id": entry.id.uuidString,
            "kind": "take",
            "createdAt": dateFormatter.string(from: entry.createdAt),
            "promptSummary": compact(entry.prompt, limit: 160),
            "duration": entry.duration,
            "seed": entry.seed,
            "pipeline": entry.pipeline ?? NSNull(),
            "modelID": entry.modelID ?? NSNull(),
            "modelName": entry.modelName ?? NSNull(),
            "fileName": entry.fileName,
            "isPreferred": entry.isPreferred ?? false,
            "listeningNote": entry.listeningNote ?? NSNull(),
            "historyResourceURI": resourceURI(kind: "history", identifier: entry.id.uuidString),
            "takeResourceURI": resourceURI(kind: "take", identifier: entry.id.uuidString),
            "audioResourceURI": resourceURI(kind: "audio", identifier: entry.id.uuidString),
            "traceResourceURI": historyStore.traceURL(for: entry) == nil
                ? NSNull()
                : resourceURI(kind: "trace", identifier: entry.id.uuidString)
        ]
    }

    private func historyAdapterPayload(_ adapter: HistoryAdapterSnapshot) -> [String: Any] {
        [
            "id": adapter.id,
            "name": adapter.name ?? NSNull(),
            "weight": adapter.weight,
            "source": adapter.source ?? NSNull(),
            "rank": adapter.rank ?? NSNull(),
            "baseModelID": adapter.baseModelID ?? NSNull()
        ]
    }

    private func loadGenerationPresets() -> [GenerationPreset] {
        let directory = PresetStore.presetsDirectory
        guard let files = try? FileManager.default.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: nil,
            options: .skipsHiddenFiles
        ) else {
            return []
        }

        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return files
            .filter { $0.pathExtension == "json" }
            .compactMap { try? Data(contentsOf: $0) }
            .compactMap { try? decoder.decode(GenerationPreset.self, from: $0) }
            .sorted { $0.createdAt > $1.createdAt }
    }

    private func generationPresetPayload(_ preset: GenerationPreset) -> [String: Any] {
        [
            "id": preset.id.uuidString,
            "kind": "generationPreset",
            "name": preset.name,
            "createdAt": dateFormatter.string(from: preset.createdAt),
            "prompt": preset.prompt,
            "promptSummary": compact(preset.prompt, limit: 160),
            "lyricsPresent": !preset.lyrics.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
            "mode": preset.mode?.rawValue ?? NSNull(),
            "modeDisplayName": preset.mode?.displayName ?? NSNull(),
            "duration": preset.duration,
            "seedText": preset.seedText,
            "guidanceScale": preset.guidanceScale,
            "guidancePreset": guidancePresetPayload(preset.guidancePreset),
            "pipelineOverride": preset.pipelineOverride.rawValue,
            "pipelineDisplayName": preset.pipelineOverride.displayName,
            "method": preset.method.rawValue,
            "precision": preset.precision.rawValue,
            "stepsOverride": preset.stepsOverride,
            "shiftOverride": preset.shiftOverride,
            "adapterID": preset.adapterID ?? NSNull(),
            "adapterStrength": preset.adapterStrength,
            "coverStrength": preset.coverStrength,
            "resourceURI": resourceURI(kind: "preset", identifier: preset.id.uuidString)
        ]
    }

    private func guidancePresetPayload(_ preset: GuidancePreset) -> [String: Any] {
        let interval = preset.cfgInterval
        return [
            "id": preset.rawValue,
            "displayName": preset.displayName,
            "detail": preset.detail,
            "cfgInterval": [
                "start": interval.start,
                "end": interval.end
            ],
            "guidanceScaleOverride": preset.guidanceScaleOverride ?? NSNull()
        ]
    }

    private func inpaintPresetPayload(_ preset: InpaintPreset) -> [String: Any] {
        [
            "id": preset.rawValue,
            "displayName": preset.rawValue.capitalized,
            "strength": preset.strength,
            "guidanceScale": preset.guidanceScale
        ]
    }

    private func reverbPresetPayload(_ preset: ReverbPreset) -> [String: Any] {
        let parameters = preset.parameters
        return [
            "id": preset.rawValue,
            "displayName": preset.displayName,
            "parameters": [
                "decayTime": parameters.decayTime,
                "wetDryMix": parameters.wetDryMix,
                "preDelay": parameters.preDelay,
                "damping": parameters.damping
            ]
        ]
    }

    private func dspEffectPayload(_ effect: DSPEffect) -> [String: Any] {
        [
            "id": effect.rawValue,
            "displayName": effect.displayName,
            "requiresSelection": effect.requiresSelection,
            "requiresParameters": effect.requiresParameters,
            "isTransform": effect.isTransform,
            "operationKind": "editor.applyEffectChain",
            "tool": "apply_editor_effect_chain"
        ]
    }

    private func workflowPromptPayload(_ workflow: WorkflowPromptDefinition) -> [String: Any] {
        [
            "id": workflow.id,
            "kind": "workflowPrompt",
            "title": workflow.title,
            "description": workflow.description,
            "promptName": workflow.promptName,
            "arguments": workflow.arguments.map { argument in
                [
                    "name": argument.name,
                    "description": argument.description,
                    "required": argument.required
                ]
            },
            "resources": workflow.resources,
            "tools": workflow.tools,
            "resourceURI": resourceURI(kind: "workflow", identifier: workflow.id)
        ]
    }

    private func resolveHistoryEntry(_ identifier: String) throws -> HistoryEntry {
        if let entryID = UUID(uuidString: identifier),
           let entry = historyStore.entries.first(where: { $0.id == entryID }) {
            return entry
        }

        if let entry = historyStore.entries.first(where: { entry in
            entry.traceFileName == identifier
                || HistoryEntry.traceFileName(forAudioFileName: entry.fileName) == identifier
                || entry.fileName == identifier
        }) {
            return entry
        }

        throw PiMPAudioCommandFacadeError.historyEntryNotFound(identifier)
    }

    private func resolveTrace(arguments: [String: Any]) throws -> GenerationTrace {
        let id = stringValue(arguments["id"])
            ?? stringValue(arguments["historyEntryID"])
            ?? stringValue(arguments["traceFileName"])
            ?? stringValue(arguments["audioFileName"])

        guard let id, !id.isEmpty else {
            throw PiMPAudioCommandFacadeError.missingTraceIdentifier
        }

        if let entryID = UUID(uuidString: id),
           let entry = historyStore.entries.first(where: { $0.id == entryID }) {
            return try historyStore.loadTrace(for: entry)
        }

        if let entry = historyStore.entries.first(where: { entry in
            entry.traceFileName == id
                || HistoryEntry.traceFileName(forAudioFileName: entry.fileName) == id
                || entry.fileName == id
        }) {
            return try historyStore.loadTrace(for: entry)
        }

        if id.hasSuffix(".trace.json") || id.hasSuffix(".failed.trace.json") {
            return try historyStore.loadTrace(traceFileName: id)
        }

        throw PiMPAudioCommandFacadeError.traceNotFound(id)
    }

    private func historySummary(_ entry: HistoryEntry) -> [String: Any] {
        [
            "id": entry.id.uuidString,
            "createdAt": dateFormatter.string(from: entry.createdAt),
            "prompt": entry.prompt,
            "promptSummary": compact(entry.prompt, limit: 160),
            "duration": entry.duration,
            "seed": entry.seed,
            "method": entry.method,
            "guidanceScale": entry.guidanceScale,
            "precision": entry.precision,
            "fileName": entry.fileName,
            "audioFilePath": historyStore.fileURL(for: entry).path(percentEncoded: false),
            "traceFileName": entry.traceFileName ?? NSNull(),
            "hasTrace": historyStore.traceURL(for: entry) != nil,
            "pipeline": entry.pipeline ?? NSNull(),
            "modelID": entry.modelID ?? NSNull(),
            "modelName": entry.modelName ?? NSNull(),
            "adapterCount": entry.adapters?.count ?? 0,
            "isPreferred": entry.isPreferred ?? false,
            "listeningNote": entry.listeningNote ?? NSNull()
        ]
    }

    private func tracePayload(_ trace: GenerationTrace) throws -> [String: Any] {
        var payload = try encodedJSONObject(trace)
        payload["audioFilePath"] = historyStore.fileURL(forAudioFileName: trace.audioFileName).path(percentEncoded: false)
        return payload
    }

    private func traceSummary(_ trace: GenerationTrace) -> [String: Any] {
        [
            "historyEntryID": trace.historyEntryID.uuidString,
            "createdAt": dateFormatter.string(from: trace.createdAt),
            "pipeline": trace.pipeline,
            "model": trace.primaryModel?.displayName ?? trace.primaryModel?.id ?? "None",
            "audioFileName": trace.audioFileName,
            "audioFilePath": historyStore.fileURL(forAudioFileName: trace.audioFileName).path(percentEncoded: false),
            "traceFileName": trace.traceFileName,
            "seed": trace.request.seed.map(String.init) ?? "Random",
            "guidanceScale": trace.request.guidanceScale,
            "guidancePreset": trace.request.guidancePresetName,
            "duration": trace.request.duration,
            "absMean": trace.audio.absMean,
            "absMax": trace.audio.absMax,
            "isPreferred": trace.isPreferred ?? false,
            "listeningNote": trace.listeningNote ?? NSNull()
        ]
    }

    private func reviewPayload(_ trace: GenerationTrace) -> [String: Any] {
        [
            "historyEntryID": trace.historyEntryID.uuidString,
            "traceFileName": trace.traceFileName,
            "audioFileName": trace.audioFileName,
            "audioFilePath": historyStore.fileURL(forAudioFileName: trace.audioFileName).path(percentEncoded: false),
            "isPreferred": trace.isPreferred ?? false,
            "listeningNote": trace.listeningNote ?? NSNull()
        ]
    }

    private func memoryReleasePayload(_ summary: MemoryReleaseSummary) -> [String: Any] {
        [
            "released": true,
            "before": [
                "activeMemoryBytes": summary.beforeActiveMemory,
                "cacheMemoryBytes": summary.beforeCacheMemory,
                "peakMemoryBytes": summary.beforePeakMemory
            ],
            "after": [
                "activeMemoryBytes": summary.afterActiveMemory,
                "cacheMemoryBytes": summary.afterCacheMemory,
                "peakMemoryBytes": summary.afterPeakMemory
            ],
            "delta": [
                "releasedActiveMemoryBytes": summary.releasedActiveMemory,
                "releasedCacheMemoryBytes": summary.releasedCacheMemory
            ]
        ]
    }

    private func adapterSelectionPayload(_ selection: [AdapterSelection]) -> [String: Any] {
        [
            "applied": true,
            "cleared": selection.isEmpty,
            "count": selection.count,
            "primary": selection.first.map(adapterSelectionEntryPayload) ?? NSNull(),
            "adapters": selection.map(adapterSelectionEntryPayload)
        ]
    }

    private func adapterSelectionEntryPayload(_ entry: AdapterSelection) -> [String: Any] {
        [
            "id": entry.adapter.id,
            "name": entry.adapter.name,
            "strength": entry.strength,
            "baseModelID": entry.adapter.baseModelID,
            "baseModelCategory": entry.adapter.baseModelCategory.rawValue,
            "rank": entry.adapter.config.rank,
            "alpha": entry.adapter.config.alpha
        ]
    }

    private func generationSubmissionPayload(
        request: GenerationRequest,
        submitted: Bool,
        dryRun: Bool,
        operationID: String? = nil
    ) -> [String: Any] {
        [
            "submitted": submitted,
            "dryRun": dryRun,
            "request": generationRequestPayload(request),
            "status": generationService.map { generationStatePayload($0.state) } ?? NSNull(),
            "operationID": operationID.map { $0 as Any } ?? NSNull(),
            "operation": operationID
                .flatMap { operationRecords[$0].map(operationPayload) }
                .map { $0 as Any } ?? NSNull()
        ]
    }

    private func generationRequestPayload(_ request: GenerationRequest) -> [String: Any] {
        [
            "prompt": request.text,
            "lyrics": request.lyrics.isEmpty ? NSNull() : request.lyrics,
            "duration": request.duration,
            "seed": request.seed.map { $0 as Any } ?? NSNull(),
            "method": request.method.rawValue,
            "guidanceScale": request.guidanceScale,
            "effectiveGuidanceScale": request.effectiveGuidanceScale,
            "guidancePreset": request.guidancePreset.rawValue,
            "cfgIntervalStart": request.cfgIntervalStart,
            "cfgIntervalEnd": request.cfgIntervalEnd,
            "precision": request.precision.rawValue,
            "steps": request.stepsOverride.map { $0 as Any } ?? NSNull(),
            "shift": request.shiftOverride.map { $0 as Any } ?? NSNull(),
            "mode": request.resolvedMode.rawValue,
            "pipeline": request.pipelineOverride.rawValue,
            "referenceAudioPath": request.referenceAudioURL?.path(percentEncoded: false) ?? NSNull(),
            "coverStrength": request.coverStrength
        ]
    }

    private func generationResultPayload(_ result: GenerationResult) -> [String: Any] {
        [
            "historyEntryID": result.entry.id.uuidString,
            "fileName": result.entry.fileName,
            "audioFilePath": result.fileURL.path(percentEncoded: false),
            "traceFileName": result.entry.traceFileName ?? NSNull(),
            "prompt": result.entry.prompt,
            "duration": result.entry.duration,
            "seed": result.entry.seed,
            "pipeline": result.entry.pipeline ?? NSNull(),
            "modelID": result.entry.modelID ?? NSNull(),
            "modelName": result.entry.modelName ?? NSNull(),
            "hasWaveformPeaks": result.waveformPeaks != nil
        ]
    }

    private func generationStatePayload(_ state: GenerationState) -> [String: Any] {
        switch state {
        case .idle:
            return ["name": "idle"]
        case .downloadingModel(let progress):
            return ["name": "downloadingModel", "progress": progress]
        case .loadingWeights(let progress):
            return ["name": "loadingWeights", "progress": progress]
        case .conditioning:
            return ["name": "conditioning"]
        case .tokenizingSource:
            return ["name": "tokenizingSource"]
        case .lmAudioCodeConditioning:
            return ["name": "lmAudioCodeConditioning"]
        case .generating(let step, let totalSteps, let eta):
            return ["name": "generating", "step": step, "totalSteps": totalSteps, "eta": eta]
        case .decoding:
            return ["name": "decoding"]
        case .complete:
            return ["name": "complete"]
        case .error(let message):
            return ["name": "error", "message": message]
        case .cancelled:
            return ["name": "cancelled"]
        }
    }

    private func registerGenerationOperation(request: GenerationRequest) -> MCPOperationRecord {
        let record = registerOperation(
            kind: "generation",
            title: "Generate audio",
            request: generationRequestPayload(request),
            progress: [
                "stage": "queued",
                "progress": 0.0,
                "total": 1.0,
                "message": "Generation request accepted."
            ]
        )

        activeGenerationOperationID = record.id
        return record
    }

    private func refreshGenerationOperation() {
        guard let id = activeGenerationOperationID,
              var record = operationRecords[id],
              let generationService else {
            return
        }

        let state = generationService.state
        record.updatedAt = Date()
        record.progress = generationOperationProgressPayload(state)

        switch state {
        case .complete:
            record.status = .succeeded
            record.result = generationService.currentResult.map(generationResultPayload)
            activeGenerationOperationID = nil
        case .error(let message):
            record.status = .failed
            record.error = message
            activeGenerationOperationID = nil
        case .cancelled:
            record.status = .cancelled
            if record.cancellationReason == nil {
                record.cancellationReason = "Generation was cancelled."
            }
            activeGenerationOperationID = nil
        case .idle:
            record.status = .running
            record.progress = [
                "stage": "queued",
                "progress": 0.0,
                "total": 1.0,
                "message": "Generation request is queued."
            ]
        default:
            record.status = .running
        }

        operationRecords[id] = record
    }

    private func generationOperationProgressPayload(_ state: GenerationState) -> [String: Any] {
        switch state {
        case .idle:
            return generationOperationProgressPayload(
                stage: "idle",
                message: "Generation service is idle."
            )
        case .downloadingModel(let progress):
            return generationOperationProgressPayload(
                stage: "downloadingModel",
                message: "Downloading model weights.",
                stageProgress: Double(progress),
                stageTotal: 1.0
            )
        case .loadingWeights(let progress):
            return generationOperationProgressPayload(
                stage: "loadingWeights",
                message: "Loading model weights.",
                stageProgress: Double(progress),
                stageTotal: 1.0
            )
        case .conditioning:
            return generationOperationProgressPayload(
                stage: "conditioning",
                message: "Preparing prompt conditioning."
            )
        case .tokenizingSource:
            return generationOperationProgressPayload(
                stage: "tokenizingSource",
                message: "Tokenizing reference audio."
            )
        case .lmAudioCodeConditioning:
            return generationOperationProgressPayload(
                stage: "lmAudioCodeConditioning",
                message: "Conditioning reference audio codes."
            )
        case .generating(let step, let totalSteps, let eta):
            var payload = generationOperationProgressPayload(
                stage: "generating",
                message: "Generating audio.",
                stageProgress: Double(step),
                stageTotal: Double(totalSteps)
            )
            payload["eta"] = eta
            return payload
        case .decoding:
            return generationOperationProgressPayload(
                stage: "decoding",
                message: "Decoding audio."
            )
        case .complete:
            return generationOperationProgressPayload(
                stage: "complete",
                message: "Generation completed."
            )
        case .error(let message):
            return generationOperationProgressPayload(
                stage: "error",
                message: message
            )
        case .cancelled:
            return generationOperationProgressPayload(
                stage: "cancelled",
                message: "Generation cancelled."
            )
        }
    }

    private func generationOperationProgressPayload(
        stage: String,
        message: String,
        stageProgress: Double? = nil,
        stageTotal: Double? = nil
    ) -> [String: Any] {
        operationProgressPayload(
            stage: stage,
            message: message,
            plan: generationProgressPlan,
            stageProgress: stageProgress,
            stageTotal: stageTotal
        )
    }

    private func editorLoadProgressPayload(stage: String, message: String) -> [String: Any] {
        operationProgressPayload(
            stage: stage,
            message: message,
            plan: editorLoadProgressPlan
        )
    }

    private func editorInpaintProgressPayload(_ progress: InpaintProgress) -> [String: Any] {
        switch progress {
        case .buildingMask:
            return editorInpaintProgressPayload(
                stage: "buildingMask",
                message: "Building editor inpaint mask."
            )
        case .encoding:
            return editorInpaintProgressPayload(
                stage: "encoding",
                message: "Preparing editor latents for regeneration."
            )
        case .diffusion(let diffusionProgress):
            var payload = editorInpaintProgressPayload(
                stage: "diffusion",
                message: "Regenerating selected editor audio.",
                stageProgress: Double(diffusionProgress.currentStep),
                stageTotal: Double(diffusionProgress.totalSteps)
            )
            payload["elapsedTime"] = diffusionProgress.elapsedTime
            return payload
        case .decoding:
            return editorInpaintProgressPayload(
                stage: "decoding",
                message: "Decoding edited audio."
            )
        case .complete:
            return editorInpaintProgressPayload(
                stage: "complete",
                message: "Editor inpaint completed."
            )
        case .failed(let message):
            return editorInpaintProgressPayload(
                stage: "error",
                message: message
            )
        }
    }

    private func editorInpaintProgressPayload(
        stage: String,
        message: String,
        stageProgress: Double? = nil,
        stageTotal: Double? = nil
    ) -> [String: Any] {
        operationProgressPayload(
            stage: stage,
            message: message,
            plan: editorInpaintProgressPlan,
            stageProgress: stageProgress,
            stageTotal: stageTotal
        )
    }

    private func editorRenderProgressPayload(stage: String, message: String) -> [String: Any] {
        operationProgressPayload(
            stage: stage,
            message: message,
            plan: editorRenderProgressPlan
        )
    }

    private func editorDSPProgressPayload(
        stage: String,
        message: String,
        stageProgress: Double? = nil,
        stageTotal: Double? = nil
    ) -> [String: Any] {
        operationProgressPayload(
            stage: stage,
            message: message,
            plan: editorDSPProgressPlan,
            stageProgress: stageProgress,
            stageTotal: stageTotal
        )
    }

    private func cancellationProgressPayload(for record: MCPOperationRecord) -> [String: Any] {
        if record.kind == "generation" {
            return generationOperationProgressPayload(.cancelled)
        }
        if record.kind == "editor.loadAudio" {
            return editorLoadProgressPayload(stage: "cancelled", message: "Editor audio load cancelled.")
        }
        if record.kind == "editor.inpaint" {
            return editorInpaintProgressPayload(stage: "cancelled", message: "Editor inpaint cancelled.")
        }
        if record.kind == "editor.renderAudio" {
            return editorRenderProgressPayload(stage: "cancelled", message: "Editor render cancelled.")
        }
        if record.kind == "editor.applyEffectChain" {
            return editorDSPProgressPayload(stage: "cancelled", message: "Editor DSP chain cancelled.")
        }
        return [
            "stage": "cancelled",
            "progress": 1.0,
            "total": 1.0,
            "message": "Operation cancelled."
        ]
    }

    private func operationProgressPayload(
        stage: String,
        message: String,
        plan: [String: (base: Double, span: Double)],
        stageProgress: Double? = nil,
        stageTotal: Double? = nil
    ) -> [String: Any] {
        let plan = plan[stage] ?? (base: 0.0, span: 0.0)
        let stageRatio: Double
        if let stageProgress, let stageTotal, stageTotal > 0 {
            stageRatio = min(max(stageProgress / stageTotal, 0), 1)
        } else {
            stageRatio = plan.span > 0 ? 0.0 : 1.0
        }

        let progress = min(max(plan.base + (plan.span * stageRatio), 0), 1)
        var payload: [String: Any] = [
            "stage": stage,
            "progress": progress,
            "total": 1.0,
            "message": message
        ]

        if let stageProgress {
            payload["stageProgress"] = stageProgress
        }
        if let stageTotal {
            payload["stageTotal"] = stageTotal
        }
        return payload
    }

    private func runEditorAudioLoadOperation(
        id: String,
        source: EditorAudioLoadSource,
        originalPrompt: String?
    ) async {
        defer { operationTasks[id] = nil }

        do {
            try Task.checkCancellation()
            updateOperationProgress(
                id: id,
                progress: editorLoadProgressPayload(stage: "readingFile", message: "Reading audio file.")
            )
            let buffer = try await readAudioBuffer(from: source.url)

            try Task.checkCancellation()
            updateOperationProgress(
                id: id,
                progress: editorLoadProgressPayload(stage: "preparingAudio", message: "Preparing editor audio buffer.")
            )
            let samples = AudioBufferConverter.toMLXArray(buffer)

            try Task.checkCancellation()
            guard let editorService, let coordinator = editorService.coordinator else {
                throw PiMPAudioCommandFacadeError.editorServiceUnavailable
            }

            updateOperationProgress(
                id: id,
                progress: editorLoadProgressPayload(stage: "encodingLatents", message: "Encoding editor latents.")
            )
            let loadResult = try await coordinator.loadAudio(
                samples: samples,
                sampleRate: buffer.sampleRate,
                precision: .float32
            )

            try Task.checkCancellation()
            let loadedAudio = EditorLoadedAudio(
                source: source.kind,
                historyEntryID: source.historyEntryID,
                fileName: source.fileName,
                path: source.url.path(percentEncoded: false),
                duration: loadResult.duration,
                sampleRate: loadResult.sampleRate,
                channels: buffer.channels,
                frameCount: buffer.frameCount,
                latentFrames: loadResult.latentFrames,
                originalPrompt: originalPrompt,
                loadedAt: Date()
            )
            editorService.recordLoadedAudio(loadedAudio)
            completeOperation(
                id: id,
                progress: editorLoadProgressPayload(stage: "complete", message: "Editor audio loaded."),
                result: [
                    "loadedAudio": editorLoadedAudioPayload(loadedAudio)
                ]
            )
        } catch is CancellationError {
            cancelOperationRecord(
                id: id,
                reason: "Editor audio load was cancelled.",
                progress: editorLoadProgressPayload(stage: "cancelled", message: "Editor audio load cancelled.")
            )
        } catch {
            failOperation(
                id: id,
                error: error.localizedDescription,
                progress: editorLoadProgressPayload(stage: "error", message: error.localizedDescription)
            )
        }
    }

    private func runEditorInpaintOperation(id: String, request: InpaintRequest) async {
        defer { operationTasks[id] = nil }

        do {
            try Task.checkCancellation()
            guard let editorService, let coordinator = editorService.coordinator else {
                throw PiMPAudioCommandFacadeError.editorServiceUnavailable
            }

            var completed = false
            for await progress in await coordinator.edit(request: request) {
                try Task.checkCancellation()
                updateOperationProgress(id: id, progress: editorInpaintProgressPayload(progress))

                switch progress {
                case .complete:
                    completed = true
                    let summary = EditorEditSummary(
                        mode: request.editMode.rawValue,
                        regionPrompt: request.regionPrompt,
                        originalPrompt: request.originalPrompt,
                        pipeline: request.pipeline.rawValue,
                        selectionStart: request.selection.startTime,
                        selectionEnd: request.selection.endTime,
                        lowCutHz: request.selection.frequencyBand?.lowCutHz,
                        highCutHz: request.selection.frequencyBand?.highCutHz,
                        strength: request.strength,
                        guidanceScale: request.guidanceScale,
                        seed: request.seed,
                        completedAt: Date()
                    )
                    editorService.recordCompletedEdit(summary)
                    completeOperation(
                        id: id,
                        progress: editorInpaintProgressPayload(progress),
                        result: [
                            "edit": editorEditSummaryPayload(summary),
                            "canUndo": await coordinator.canUndo,
                            "canRedo": await coordinator.canRedo
                        ]
                    )
                    return
                case .failed(let message):
                    failOperation(
                        id: id,
                        error: message,
                        progress: editorInpaintProgressPayload(progress)
                    )
                    return
                default:
                    break
                }
            }

            if !completed {
                throw PiMPAudioCommandFacadeError.editorInpaintDidNotComplete
            }
        } catch is CancellationError {
            cancelOperationRecord(
                id: id,
                reason: "Editor inpaint was cancelled.",
                progress: editorInpaintProgressPayload(stage: "cancelled", message: "Editor inpaint cancelled.")
            )
        } catch {
            failOperation(
                id: id,
                error: error.localizedDescription,
                progress: editorInpaintProgressPayload(stage: "error", message: error.localizedDescription)
            )
        }
    }

    private func runEditorRenderOperation(id: String, request: EditorRenderRequest) async {
        defer { operationTasks[id] = nil }

        do {
            try Task.checkCancellation()
            guard let editorService, let coordinator = editorService.coordinator else {
                throw PiMPAudioCommandFacadeError.editorServiceUnavailable
            }
            guard let loadedAudio = editorService.loadedAudio else {
                throw PiMPAudioCommandFacadeError.editorAudioNotLoaded
            }

            updateOperationProgress(
                id: id,
                progress: editorRenderProgressPayload(stage: "readingBuffer", message: "Reading editor audio buffer.")
            )
            guard let currentAudio = await coordinator.getCurrentAudio() else {
                throw PiMPAudioCommandFacadeError.editorAudioNotLoaded
            }

            try Task.checkCancellation()
            updateOperationProgress(
                id: id,
                progress: editorRenderProgressPayload(stage: "preparingAudio", message: "Preparing rendered audio.")
            )
            let fullBuffer = AudioBufferConverter.fromMLXArray(
                currentAudio,
                sampleRate: loadedAudio.sampleRate
            )
            let renderedBuffer = try request.selection.map {
                try croppedAudioBuffer(fullBuffer, selection: $0)
            } ?? fullBuffer

            try Task.checkCancellation()
            updateOperationProgress(
                id: id,
                progress: editorRenderProgressPayload(stage: "writingFile", message: "Writing rendered audio file.")
            )
            try FileManager.default.createDirectory(
                at: request.outputURL.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            do {
                try writeAudioBuffer(renderedBuffer, to: request.outputURL)
            } catch {
                throw PiMPAudioCommandFacadeError.editorRenderWriteFailed(
                    request.outputURL.path(percentEncoded: false),
                    error.localizedDescription
                )
            }

            let result = editorRenderResultPayload(
                request: request,
                buffer: renderedBuffer,
                sourceAudio: loadedAudio
            )
            completeOperation(
                id: id,
                progress: editorRenderProgressPayload(stage: "complete", message: "Editor render completed."),
                result: result
            )
        } catch is CancellationError {
            cancelOperationRecord(
                id: id,
                reason: "Editor render was cancelled.",
                progress: editorRenderProgressPayload(stage: "cancelled", message: "Editor render cancelled.")
            )
        } catch {
            failOperation(
                id: id,
                error: error.localizedDescription,
                progress: editorRenderProgressPayload(stage: "error", message: error.localizedDescription)
            )
        }
    }

    private func runEditorDSPOperation(id: String, request: EditorDSPRequest) async {
        defer { operationTasks[id] = nil }

        do {
            try Task.checkCancellation()
            guard let editorService, let coordinator = editorService.coordinator else {
                throw PiMPAudioCommandFacadeError.editorServiceUnavailable
            }
            guard let loadedAudio = editorService.loadedAudio else {
                throw PiMPAudioCommandFacadeError.editorAudioNotLoaded
            }

            updateOperationProgress(
                id: id,
                progress: editorDSPProgressPayload(stage: "readingBuffer", message: "Reading editor audio buffer.")
            )
            guard let currentAudio = await coordinator.getCurrentAudio() else {
                throw PiMPAudioCommandFacadeError.editorAudioNotLoaded
            }

            try Task.checkCancellation()
            var workingBuffer = AudioBufferConverter.fromMLXArray(
                currentAudio,
                sampleRate: loadedAudio.sampleRate
            )
            let beforeBuffer = workingBuffer
            let processor = DSPProcessor()
            var summaries: [EditorDSPEffectSummary] = []

            for (index, step) in request.steps.enumerated() {
                try Task.checkCancellation()
                let operation = processor.makeDescription(
                    effect: step.effect,
                    target: step.target,
                    parameters: step.parameters
                )
                updateOperationProgress(
                    id: id,
                    progress: editorDSPProgressPayload(
                        stage: "applyingEffects",
                        message: "Applying \(step.effect.displayName).",
                        stageProgress: Double(index),
                        stageTotal: Double(request.steps.count)
                    )
                )
                workingBuffer = try processor.apply(
                    effect: step.effect,
                    to: workingBuffer,
                    target: step.target,
                    parameters: step.parameters
                )
                summaries.append(editorDSPEffectSummary(step: step, description: operation.description))
            }

            try Task.checkCancellation()
            updateOperationProgress(
                id: id,
                progress: editorDSPProgressPayload(
                    stage: "applyingEffects",
                    message: "Applied editor DSP chain.",
                    stageProgress: Double(request.steps.count),
                    stageTotal: Double(request.steps.count)
                )
            )

            updateOperationProgress(
                id: id,
                progress: editorDSPProgressPayload(stage: "preparingAudio", message: "Preparing edited audio buffer.")
            )
            let editedSamples = AudioBufferConverter.toMLXArray(workingBuffer)

            try Task.checkCancellation()
            updateOperationProgress(
                id: id,
                progress: editorDSPProgressPayload(stage: "encodingLatents", message: "Encoding edited audio latents.")
            )
            let loadResult = try await coordinator.replaceAudio(
                samples: editedSamples,
                sampleRate: workingBuffer.sampleRate,
                description: request.description,
                precision: .float32
            )

            try Task.checkCancellation()
            let updatedAudio = loadedAudio.updating(
                duration: loadResult.duration,
                sampleRate: loadResult.sampleRate,
                channels: workingBuffer.channels,
                frameCount: workingBuffer.frameCount,
                latentFrames: loadResult.latentFrames
            )
            let summary = EditorEditSummary(
                dspEffects: summaries,
                beforeDuration: beforeBuffer.duration,
                afterDuration: workingBuffer.duration,
                completedAt: Date()
            )
            editorService.recordCompletedEdit(summary, loadedAudio: updatedAudio)

            completeOperation(
                id: id,
                progress: editorDSPProgressPayload(stage: "complete", message: "Editor DSP chain completed."),
                result: editorDSPResultPayload(
                    request: request,
                    sourceAudio: loadedAudio,
                    updatedAudio: updatedAudio,
                    beforeBuffer: beforeBuffer,
                    afterBuffer: workingBuffer,
                    edit: summary
                )
            )
        } catch is CancellationError {
            cancelOperationRecord(
                id: id,
                reason: "Editor DSP chain was cancelled.",
                progress: editorDSPProgressPayload(stage: "cancelled", message: "Editor DSP chain cancelled.")
            )
        } catch {
            failOperation(
                id: id,
                error: error.localizedDescription,
                progress: editorDSPProgressPayload(stage: "error", message: error.localizedDescription)
            )
        }
    }

    private func readAudioBuffer(from url: URL) async throws -> AudioBuffer {
        try await Task.detached(priority: .userInitiated) {
            let audioFile = try AVAudioFile(forReading: url)
            let frameCount = AVAudioFrameCount(audioFile.length)
            guard let pcmBuffer = AVAudioPCMBuffer(
                pcmFormat: audioFile.processingFormat,
                frameCapacity: frameCount
            ) else {
                throw AudioBufferConverter.ConversionError.bufferCreationFailed
            }
            try audioFile.read(into: pcmBuffer)
            return AudioBufferConverter.fromPCMBuffer(pcmBuffer)
        }.value
    }

    private func writeAudioBuffer(_ buffer: AudioBuffer, to url: URL) throws {
        let pcmBuffer = try AudioBufferConverter.toPCMBuffer(buffer)
        let outputFile = try AVAudioFile(forWriting: url, settings: pcmBuffer.format.settings)
        try outputFile.write(from: pcmBuffer)
    }

    private func croppedAudioBuffer(
        _ buffer: AudioBuffer,
        selection: InpaintSelection
    ) throws -> AudioBuffer {
        try validateEditorSelection(selection, loadedAudio: nil)

        let startFrame = max(0, Int((selection.startTime * Double(buffer.sampleRate)).rounded(.down)))
        let endFrame = min(
            buffer.frameCount,
            Int((selection.endTime * Double(buffer.sampleRate)).rounded(.up))
        )
        guard endFrame > startFrame else {
            throw PiMPAudioCommandFacadeError.invalidEditorSelection
        }

        let bytesPerFrame = MemoryLayout<Int16>.size * buffer.channels
        let startByte = startFrame * bytesPerFrame
        let endByte = endFrame * bytesPerFrame
        return AudioBuffer(
            data: Data(buffer.data[startByte..<endByte]),
            sampleRate: buffer.sampleRate,
            channels: buffer.channels
        )
    }

    private func registerOperation(
        kind: String,
        title: String,
        request: [String: Any],
        progress: [String: Any]?
    ) -> MCPOperationRecord {
        let now = Date()
        let id = "op_\(UUID().uuidString.lowercased())"
        let record = MCPOperationRecord(
            id: id,
            kind: kind,
            title: title,
            createdAt: now,
            updatedAt: now,
            status: .running,
            request: request,
            progress: progress,
            result: nil,
            error: nil,
            cancellationReason: nil
        )

        operationRecords[id] = record
        operationOrder.insert(id, at: 0)
        trimOperationRecords()
        return record
    }

    private func updateOperationProgress(id: String, progress: [String: Any]) {
        guard var record = operationRecords[id], !record.status.isTerminal else { return }
        record.updatedAt = Date()
        record.progress = progress
        operationRecords[id] = record
    }

    private func completeOperation(
        id: String,
        progress: [String: Any],
        result: [String: Any]
    ) {
        guard var record = operationRecords[id], !record.status.isTerminal else { return }
        record.updatedAt = Date()
        record.status = .succeeded
        record.progress = progress
        record.result = result
        operationRecords[id] = record
    }

    private func failOperation(id: String, error: String, progress: [String: Any]) {
        guard var record = operationRecords[id], !record.status.isTerminal else { return }
        record.updatedAt = Date()
        record.status = .failed
        record.progress = progress
        record.error = error
        operationRecords[id] = record
    }

    private func cancelOperationRecord(id: String, reason: String, progress: [String: Any]) {
        guard var record = operationRecords[id], !record.status.isTerminal else { return }
        record.updatedAt = Date()
        record.status = .cancelled
        record.progress = progress
        record.cancellationReason = reason
        operationRecords[id] = record
    }

    private func operationSnapshot(limit: Int) -> [MCPOperationRecord] {
        operationOrder
            .prefix(limit)
            .compactMap { operationRecords[$0] }
    }

    private func operationPayload(_ record: MCPOperationRecord) -> [String: Any] {
        [
            "id": record.id,
            "kind": record.kind,
            "title": record.title,
            "status": record.status.rawValue,
            "createdAt": dateFormatter.string(from: record.createdAt),
            "updatedAt": dateFormatter.string(from: record.updatedAt),
            "cancelable": isOperationCancelable(record),
            "request": record.request,
            "progress": record.progress.map { $0 as Any } ?? NSNull(),
            "result": record.result.map { $0 as Any } ?? NSNull(),
            "error": record.error.map { $0 as Any } ?? NSNull(),
            "cancellationReason": record.cancellationReason.map { $0 as Any } ?? NSNull()
        ]
    }

    private func currentOperationPayload() -> Any? {
        if let id = activeGenerationOperationID,
           let record = operationRecords[id] {
            return operationPayload(record)
        }
        guard let latest = operationSnapshot(limit: 1).first else {
            return nil
        }
        return operationPayload(latest)
    }

    private func currentOperationPayload(kindPrefix: String) -> Any? {
        operationOrder
            .compactMap { operationRecords[$0] }
            .first { $0.kind.hasPrefix(kindPrefix) && !$0.status.isTerminal }
            .map(operationPayload)
    }

    private func activeOperationIDsPayload() -> [String: String] {
        var ids: [String: String] = [:]
        if let activeGenerationOperationID {
            ids["generation"] = activeGenerationOperationID
        }
        for id in operationTasks.keys.sorted() {
            guard let record = operationRecords[id], !record.status.isTerminal else { continue }
            ids[record.kind] = id
        }
        return ids
    }

    private func isOperationCancelable(_ record: MCPOperationRecord) -> Bool {
        if record.status.isTerminal {
            return false
        }
        if record.kind == "generation", record.id == activeGenerationOperationID {
            return true
        }
        return operationTasks[record.id] != nil
    }

    private func operationID(from arguments: [String: Any]) throws -> String {
        let id = stringValue(arguments["id"] ?? arguments["operationID"] ?? arguments["operationId"])?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        guard !id.isEmpty else {
            throw PiMPAudioCommandFacadeError.missingRequiredArgument("id")
        }
        return id
    }

    private func trimOperationRecords() {
        guard operationOrder.count > maxOperationRecords else { return }
        let staleIDs = operationOrder.dropFirst(maxOperationRecords)
        operationOrder = Array(operationOrder.prefix(maxOperationRecords))
        for id in staleIDs {
            operationRecords[id] = nil
            operationTasks[id]?.cancel()
            operationTasks[id] = nil
        }
    }

    private func editorAudioLoadRequestPayload(
        source: EditorAudioLoadSource,
        originalPrompt: String?
    ) -> [String: Any] {
        [
            "source": source.kind,
            "historyEntryID": source.historyEntryID.map { $0 as Any } ?? NSNull(),
            "path": source.url.path(percentEncoded: false),
            "fileName": source.fileName,
            "originalPrompt": originalPrompt.map { $0 as Any } ?? NSNull()
        ]
    }

    private func editorLoadedAudioPayload(_ audio: EditorLoadedAudio) -> [String: Any] {
        [
            "source": audio.source,
            "historyEntryID": audio.historyEntryID.map { $0 as Any } ?? NSNull(),
            "fileName": audio.fileName,
            "path": audio.path,
            "duration": audio.duration,
            "sampleRate": audio.sampleRate,
            "channels": audio.channels,
            "frameCount": audio.frameCount,
            "latentFrames": audio.latentFrames,
            "originalPrompt": audio.originalPrompt.map { $0 as Any } ?? NSNull(),
            "loadedAt": dateFormatter.string(from: audio.loadedAt)
        ]
    }

    private func editorEditSummaryPayload(_ edit: EditorEditSummary) -> [String: Any] {
        [
            "kind": edit.kind,
            "mode": edit.mode.map { $0 as Any } ?? NSNull(),
            "regionPrompt": edit.regionPrompt.map { $0 as Any } ?? NSNull(),
            "originalPrompt": edit.originalPrompt.map { $0 as Any } ?? NSNull(),
            "pipeline": edit.pipeline.map { $0 as Any } ?? NSNull(),
            "selection": [
                "startTime": edit.selectionStart.map { $0 as Any } ?? NSNull(),
                "endTime": edit.selectionEnd.map { $0 as Any } ?? NSNull(),
                "frequencyBand": editorFrequencyBandPayload(
                    lowCutHz: edit.lowCutHz,
                    highCutHz: edit.highCutHz
                )
            ],
            "strength": edit.strength.map { $0 as Any } ?? NSNull(),
            "guidanceScale": edit.guidanceScale.map { $0 as Any } ?? NSNull(),
            "seed": edit.seed.map { $0 as Any } ?? NSNull(),
            "dspEffects": edit.dspEffects.map(editorDSPEffectSummaryPayload),
            "beforeDuration": edit.beforeDuration.map { $0 as Any } ?? NSNull(),
            "afterDuration": edit.afterDuration.map { $0 as Any } ?? NSNull(),
            "completedAt": dateFormatter.string(from: edit.completedAt)
        ]
    }

    private func editorInpaintRequestPayload(_ request: InpaintRequest) -> [String: Any] {
        [
            "mode": request.editMode.rawValue,
            "regionPrompt": request.regionPrompt,
            "originalPrompt": request.originalPrompt.map { $0 as Any } ?? NSNull(),
            "pipeline": request.pipeline.rawValue,
            "selection": editorSelectionPayload(request.selection),
            "strength": request.strength,
            "guidanceScale": request.guidanceScale,
            "seed": request.seed.map { $0 as Any } ?? NSNull()
        ]
    }

    private func editorRenderRequestPayload(_ request: EditorRenderRequest) -> [String: Any] {
        [
            "format": request.format.rawValue,
            "outputPath": request.outputURL.path(percentEncoded: false),
            "overwrite": request.overwrite,
            "selection": request.selection.map { editorSelectionPayload($0) as Any } ?? NSNull()
        ]
    }

    private func editorRenderResultPayload(
        request: EditorRenderRequest,
        buffer: AudioBuffer,
        sourceAudio: EditorLoadedAudio
    ) -> [String: Any] {
        [
            "fileName": request.outputURL.lastPathComponent,
            "path": request.outputURL.path(percentEncoded: false),
            "format": request.format.rawValue,
            "duration": buffer.duration,
            "sampleRate": buffer.sampleRate,
            "channels": buffer.channels,
            "frameCount": buffer.frameCount,
            "selection": request.selection.map { editorSelectionPayload($0) as Any } ?? NSNull(),
            "sourceLoadedAudio": editorLoadedAudioPayload(sourceAudio)
        ]
    }

    private func editorDSPRequestPayload(_ request: EditorDSPRequest) -> [String: Any] {
        [
            "effectCount": request.steps.count,
            "description": request.description,
            "effects": request.steps.map(editorDSPStepPayload)
        ]
    }

    private func editorDSPStepPayload(_ step: EditorDSPEffectStep) -> [String: Any] {
        [
            "effect": step.effect.rawValue,
            "displayName": step.effect.displayName,
            "target": editorDSPTargetPayload(step.target),
            "parameters": editorDSPParametersPayload(step.parameters)
        ]
    }

    private func editorDSPResultPayload(
        request: EditorDSPRequest,
        sourceAudio: EditorLoadedAudio,
        updatedAudio: EditorLoadedAudio,
        beforeBuffer: AudioBuffer,
        afterBuffer: AudioBuffer,
        edit: EditorEditSummary
    ) -> [String: Any] {
        [
            "request": editorDSPRequestPayload(request),
            "edit": editorEditSummaryPayload(edit),
            "sourceLoadedAudio": editorLoadedAudioPayload(sourceAudio),
            "updatedLoadedAudio": editorLoadedAudioPayload(updatedAudio),
            "before": editorAudioBufferPayload(beforeBuffer),
            "after": editorAudioBufferPayload(afterBuffer),
            "canUndo": true,
            "canRedo": false
        ]
    }

    private func editorAudioBufferPayload(_ buffer: AudioBuffer) -> [String: Any] {
        [
            "duration": buffer.duration,
            "sampleRate": buffer.sampleRate,
            "channels": buffer.channels,
            "frameCount": buffer.frameCount
        ]
    }

    private func editorDSPEffectSummary(
        step: EditorDSPEffectStep,
        description: String
    ) -> EditorDSPEffectSummary {
        let selection: (start: TimeInterval?, end: TimeInterval?)
        switch step.target {
        case .selection(let start, let end):
            selection = (start, end)
        case .wholeTrack:
            selection = (nil, nil)
        }
        return EditorDSPEffectSummary(
            effect: step.effect.rawValue,
            displayName: step.effect.displayName,
            description: description,
            target: editorDSPTargetName(step.target),
            selectionStart: selection.start,
            selectionEnd: selection.end,
            parameters: editorDSPParameterSummary(step.parameters)
        )
    }

    private func editorDSPEffectSummaryPayload(_ summary: EditorDSPEffectSummary) -> [String: Any] {
        [
            "effect": summary.effect,
            "displayName": summary.displayName,
            "description": summary.description,
            "target": summary.target,
            "selection": [
                "startTime": summary.selectionStart.map { $0 as Any } ?? NSNull(),
                "endTime": summary.selectionEnd.map { $0 as Any } ?? NSNull()
            ],
            "parameters": summary.parameters
        ]
    }

    private func editorDSPTargetPayload(_ target: ApplyTarget) -> [String: Any] {
        switch target {
        case .wholeTrack:
            return [
                "type": "wholeTrack",
                "selection": NSNull()
            ]
        case .selection(let start, let end):
            return [
                "type": "selection",
                "selection": [
                    "startTime": start,
                    "endTime": end
                ]
            ]
        }
    }

    private func editorDSPTargetName(_ target: ApplyTarget) -> String {
        switch target {
        case .wholeTrack:
            return "wholeTrack"
        case .selection:
            return "selection"
        }
    }

    private func editorDSPParametersPayload(_ parameters: DSPParameters) -> Any {
        switch parameters {
        case .none:
            return NSNull()
        case .eq(let params):
            return [
                "bands": params.bands.map { band in
                    [
                        "frequency": band.frequency,
                        "gain": band.gain,
                        "q": band.q
                    ]
                }
            ]
        case .reverb(let params):
            return [
                "decayTime": params.decayTime,
                "wetDryMix": params.wetDryMix,
                "preDelay": params.preDelay,
                "damping": params.damping
            ]
        case .timeStretch(let params):
            return ["rate": params.rate]
        case .pitchShift(let params):
            return [
                "pitch": params.pitch,
                "semitones": params.pitch / 100.0
            ]
        }
    }

    private func editorDSPParameterSummary(_ parameters: DSPParameters) -> [String: Double] {
        switch parameters {
        case .none:
            return [:]
        case .eq(let params):
            var summary: [String: Double] = [:]
            for (index, band) in params.bands.enumerated() {
                summary["band\(index + 1)Frequency"] = Double(band.frequency)
                summary["band\(index + 1)Gain"] = Double(band.gain)
                summary["band\(index + 1)Q"] = Double(band.q)
            }
            return summary
        case .reverb(let params):
            return [
                "decayTime": Double(params.decayTime),
                "wetDryMix": Double(params.wetDryMix),
                "preDelay": Double(params.preDelay),
                "damping": Double(params.damping)
            ]
        case .timeStretch(let params):
            return ["rate": Double(params.rate)]
        case .pitchShift(let params):
            return [
                "pitch": Double(params.pitch),
                "semitones": Double(params.pitch / 100.0)
            ]
        }
    }

    private func editorSelectionPayload(_ selection: InpaintSelection) -> [String: Any] {
        [
            "startTime": selection.startTime,
            "endTime": selection.endTime,
            "frequencyBand": selection.frequencyBand.map { editorFrequencyBandPayload($0) as Any } ?? NSNull()
        ]
    }

    private func editorFrequencyBandPayload(_ band: FrequencyBand) -> Any {
        editorFrequencyBandPayload(lowCutHz: band.lowCutHz, highCutHz: band.highCutHz)
    }

    private func editorFrequencyBandPayload(lowCutHz: Float?, highCutHz: Float?) -> Any {
        guard let lowCutHz, let highCutHz else { return NSNull() }
        return [
            "lowCutHz": lowCutHz,
            "highCutHz": highCutHz
        ]
    }

    private func editorInpaintRequest(
        _ arguments: [String: Any],
        loadedAudio: EditorLoadedAudio?
    ) throws -> InpaintRequest {
        let editMode = try editorEditMode(from: arguments["mode"] ?? arguments["editMode"])
        let regionPrompt = try requiredString("regionPrompt", aliases: ["prompt"], in: arguments)
        let originalPrompt = optionalEditorOriginalPrompt(arguments, loadedAudio: loadedAudio)
        let selection = try editorSelection(from: arguments, editMode: editMode, loadedAudio: loadedAudio)
        let strength = try editorInpaintStrength(from: arguments)
        let guidanceScale = try editorInpaintGuidanceScale(from: arguments)
        let pipeline = try editorPipelineChoice(from: arguments)

        return InpaintRequest(
            editMode: editMode,
            selection: selection,
            regionPrompt: regionPrompt,
            originalPrompt: originalPrompt,
            strength: strength,
            guidanceScale: guidanceScale,
            pipeline: pipeline,
            seed: try seedValue(arguments["seed"])
        )
    }

    private func editorRenderRequest(
        _ arguments: [String: Any],
        loadedAudio: EditorLoadedAudio?
    ) throws -> EditorRenderRequest {
        let format = try editorRenderFormat(from: arguments["format"] ?? arguments["fileFormat"])
        let selection = try editorRenderSelection(from: arguments, loadedAudio: loadedAudio)
        let outputURL = try editorRenderOutputURL(
            arguments,
            loadedAudio: loadedAudio,
            format: format
        )
        return EditorRenderRequest(
            format: format,
            outputURL: outputURL,
            selection: selection,
            overwrite: boolValue(arguments["overwrite"]) == true
        )
    }

    private func editorDSPRequest(
        _ arguments: [String: Any],
        loadedAudio: EditorLoadedAudio?
    ) throws -> EditorDSPRequest {
        let rawSteps = try editorDSPRawSteps(from: arguments)
        let steps = try rawSteps.map { try editorDSPEffectStep($0, topLevel: arguments, loadedAudio: loadedAudio) }
        guard !steps.isEmpty else {
            throw PiMPAudioCommandFacadeError.editorDSPChainEmpty
        }
        let descriptions = steps.map { step in
            DSPProcessor()
                .makeDescription(effect: step.effect, target: step.target, parameters: step.parameters)
                .description
        }
        return EditorDSPRequest(
            steps: steps,
            description: descriptions.joined(separator: " -> ")
        )
    }

    private func editorDSPRawSteps(from arguments: [String: Any]) throws -> [[String: Any]] {
        if let rawEffects = arguments["effects"] ?? arguments["effectChain"] ?? arguments["chain"] {
            if let typed = rawEffects as? [[String: Any]] {
                return typed
            }
            if let rawItems = rawEffects as? [Any] {
                let items = rawItems.compactMap { item -> [String: Any]? in
                    if let object = item as? [String: Any] { return object }
                    if let name = stringValue(item) { return ["effect": name] }
                    return nil
                }
                guard items.count == rawItems.count else {
                    throw PiMPAudioCommandFacadeError.invalidDSPChain
                }
                return items
            }
            throw PiMPAudioCommandFacadeError.invalidDSPChain
        }

        let effect = stringValue(arguments["effect"] ?? arguments["name"])?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        guard !effect.isEmpty else {
            throw PiMPAudioCommandFacadeError.missingRequiredArgument("effect")
        }
        return [arguments]
    }

    private func editorDSPEffectStep(
        _ stepArguments: [String: Any],
        topLevel: [String: Any],
        loadedAudio: EditorLoadedAudio?
    ) throws -> EditorDSPEffectStep {
        let effectName = stringValue(
            stepArguments["effect"] ?? stepArguments["name"] ?? stepArguments["type"]
        )?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        guard !effectName.isEmpty else {
            throw PiMPAudioCommandFacadeError.missingRequiredArgument("effect")
        }

        let effect = try editorDSPEffect(from: effectName)
        let parameters = try editorDSPParameters(
            for: effect,
            stepArguments: stepArguments,
            topLevel: topLevel
        )
        let target = try editorDSPTarget(
            for: effect,
            stepArguments: stepArguments,
            topLevel: topLevel,
            loadedAudio: loadedAudio
        )
        return EditorDSPEffectStep(effect: effect, target: target, parameters: parameters)
    }

    private func editorDSPEffect(from value: String) throws -> DSPEffect {
        let normalized = value
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
            .replacingOccurrences(of: "_", with: "")
            .replacingOccurrences(of: "-", with: "")
            .replacingOccurrences(of: " ", with: "")
        switch normalized {
        case "trim", "trimselection", "trimtoselection":
            return .trimToSelection
        case "delete", "deleteselection", "cut", "cutselection":
            return .deleteSelection
        case "fadein":
            return .fadeIn
        case "fadeout":
            return .fadeOut
        case "normalize", "normalise":
            return .normalize
        case "eq", "equalizer", "equaliser", "parametriceq", "parametricequalizer":
            return .eq
        case "reverb":
            return .reverb
        case "timestretch", "stretch":
            return .timeStretch
        case "pitchshift", "pitch", "transpose":
            return .pitchShift
        default:
            throw PiMPAudioCommandFacadeError.invalidDSPEffect(value)
        }
    }

    private func editorDSPTarget(
        for effect: DSPEffect,
        stepArguments: [String: Any],
        topLevel: [String: Any],
        loadedAudio: EditorLoadedAudio?
    ) throws -> ApplyTarget {
        let targetValue = stepArguments["target"] ?? topLevel["target"]
        let explicitTarget = stringValue(targetValue)?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
            .replacingOccurrences(of: "_", with: "")
            .replacingOccurrences(of: "-", with: "")
            .replacingOccurrences(of: " ", with: "")

        let selection = try editorDSPSelection(
            stepArguments: stepArguments,
            topLevel: topLevel,
            loadedAudio: loadedAudio
        )

        if let explicitTarget, !explicitTarget.isEmpty {
            switch explicitTarget {
            case "whole", "track", "wholetrack", "full", "fulltrack":
                if effect.requiresSelection {
                    throw PiMPAudioCommandFacadeError.invalidDSPTarget("Effect '\(effect.rawValue)' requires a selection target.")
                }
                return .wholeTrack
            case "selection", "selected", "region", "range":
                guard let selection else {
                    throw PiMPAudioCommandFacadeError.invalidEditorSelection
                }
                return .selection(start: selection.startTime, end: selection.endTime)
            default:
                throw PiMPAudioCommandFacadeError.invalidDSPTarget("Unsupported target '\(explicitTarget)'.")
            }
        }

        if let selection {
            return .selection(start: selection.startTime, end: selection.endTime)
        }
        if effect.requiresSelection {
            throw PiMPAudioCommandFacadeError.invalidEditorSelection
        }
        return .wholeTrack
    }

    private func editorDSPSelection(
        stepArguments: [String: Any],
        topLevel: [String: Any],
        loadedAudio: EditorLoadedAudio?
    ) throws -> InpaintSelection? {
        let stepSelection = stepArguments["selection"] as? [String: Any]
        let topSelection = topLevel["selection"] as? [String: Any]

        let startValue = editorSelectionValue(
            ["selectionStart", "startTime", "start"],
            arguments: stepArguments,
            selectionArguments: stepSelection
        ) ?? editorSelectionValue(
            ["selectionStart", "startTime", "start"],
            arguments: topLevel,
            selectionArguments: topSelection
        )
        let endValue = editorSelectionValue(
            ["selectionEnd", "endTime", "end"],
            arguments: stepArguments,
            selectionArguments: stepSelection
        ) ?? editorSelectionValue(
            ["selectionEnd", "endTime", "end"],
            arguments: topLevel,
            selectionArguments: topSelection
        )

        guard startValue != nil || endValue != nil else { return nil }
        guard let startTime = doubleValue(startValue),
              let endTime = doubleValue(endValue) else {
            throw PiMPAudioCommandFacadeError.invalidEditorSelection
        }
        let selection = InpaintSelection(startTime: startTime, endTime: endTime)
        try validateEditorSelection(selection, loadedAudio: loadedAudio)
        return selection
    }

    private func editorDSPParameters(
        for effect: DSPEffect,
        stepArguments: [String: Any],
        topLevel: [String: Any]
    ) throws -> DSPParameters {
        let parameterArguments = stepArguments["parameters"] as? [String: Any]
            ?? stepArguments["params"] as? [String: Any]
            ?? [:]

        switch effect {
        case .eq:
            return .eq(try editorEQParameters(parameterArguments, fallback: topLevel))
        case .reverb:
            return .reverb(try editorReverbParameters(parameterArguments, fallback: topLevel))
        case .timeStretch:
            return .timeStretch(try editorTimeStretchParameters(parameterArguments, fallback: topLevel))
        case .pitchShift:
            return .pitchShift(try editorPitchShiftParameters(parameterArguments, fallback: topLevel))
        case .trimToSelection, .deleteSelection, .fadeIn, .fadeOut, .normalize:
            return .none
        }
    }

    private func editorEQParameters(
        _ parameters: [String: Any],
        fallback: [String: Any]
    ) throws -> EQParameters {
        let rawBands = parameters["bands"] ?? fallback["bands"]
        guard let rawBands else {
            return .default
        }
        let bandObjects: [[String: Any]]
        if let typed = rawBands as? [[String: Any]] {
            bandObjects = typed
        } else if let rawItems = rawBands as? [Any] {
            bandObjects = rawItems.compactMap { $0 as? [String: Any] }
            guard bandObjects.count == rawItems.count else {
                throw PiMPAudioCommandFacadeError.invalidDSPParameters("EQ bands must be objects.")
            }
        } else {
            throw PiMPAudioCommandFacadeError.invalidDSPParameters("EQ bands must be an array.")
        }

        let bands = try bandObjects.map { band -> EQBand in
            guard let frequency = doubleValue(band["frequency"] ?? band["freq"] ?? band["hz"]),
                  frequency > 0,
                  frequency <= 24_000 else {
                throw PiMPAudioCommandFacadeError.invalidDSPParameters("EQ band frequency must be between 0 and 24000 Hz.")
            }
            guard let gain = doubleValue(band["gain"] ?? band["gainDB"] ?? band["db"]),
                  gain >= -48,
                  gain <= 48 else {
                throw PiMPAudioCommandFacadeError.invalidDSPParameters("EQ band gain must be between -48 and 48 dB.")
            }
            let q = doubleValue(band["q"] ?? band["quality"]) ?? 1.0
            guard q > 0, q <= 20 else {
                throw PiMPAudioCommandFacadeError.invalidDSPParameters("EQ band q must be > 0 and <= 20.")
            }
            return EQBand(frequency: Float(frequency), gain: Float(gain), q: Float(q))
        }
        guard !bands.isEmpty else {
            throw PiMPAudioCommandFacadeError.invalidDSPParameters("EQ must include at least one band.")
        }
        return EQParameters(bands: bands)
    }

    private func editorReverbParameters(
        _ parameters: [String: Any],
        fallback: [String: Any]
    ) throws -> ReverbParameters {
        let presetValue = stringValue(parameters["preset"] ?? fallback["preset"])?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        let presetParams: ReverbParameters
        if let presetValue, !presetValue.isEmpty {
            guard let preset = ReverbPreset(rawValue: presetValue) else {
                throw PiMPAudioCommandFacadeError.invalidDSPParameters("Unsupported reverb preset '\(presetValue)'.")
            }
            presetParams = preset.parameters
        } else {
            presetParams = .default
        }

        let decayTime = doubleValue(parameters["decayTime"] ?? fallback["decayTime"])
            .map(Float.init) ?? presetParams.decayTime
        let wetDryMix = doubleValue(parameters["wetDryMix"] ?? parameters["mix"] ?? fallback["wetDryMix"] ?? fallback["mix"])
            .map(Float.init) ?? presetParams.wetDryMix
        let preDelay = doubleValue(parameters["preDelay"] ?? fallback["preDelay"])
            .map(Float.init) ?? presetParams.preDelay
        let damping = doubleValue(parameters["damping"] ?? fallback["damping"])
            .map(Float.init) ?? presetParams.damping

        guard decayTime >= 0.1, decayTime <= 20,
              wetDryMix >= 0, wetDryMix <= 1,
              preDelay >= 0, preDelay <= 500,
              damping >= 0, damping <= 1 else {
            throw PiMPAudioCommandFacadeError.invalidDSPParameters(
                "Reverb requires decayTime 0.1...20, wetDryMix 0...1, preDelay 0...500, and damping 0...1."
            )
        }
        return ReverbParameters(
            decayTime: decayTime,
            wetDryMix: wetDryMix,
            preDelay: preDelay,
            damping: damping
        )
    }

    private func editorTimeStretchParameters(
        _ parameters: [String: Any],
        fallback: [String: Any]
    ) throws -> TimeStretchParameters {
        let rate = doubleValue(parameters["rate"] ?? parameters["ratio"] ?? fallback["rate"] ?? fallback["ratio"]) ?? 1.0
        guard rate >= 0.25, rate <= 4.0 else {
            throw PiMPAudioCommandFacadeError.invalidDSPParameters("Time-stretch rate must be between 0.25 and 4.0.")
        }
        return TimeStretchParameters(rate: Float(rate))
    }

    private func editorPitchShiftParameters(
        _ parameters: [String: Any],
        fallback: [String: Any]
    ) throws -> PitchShiftParameters {
        let centsValue = doubleValue(parameters["pitch"] ?? parameters["cents"] ?? fallback["pitch"] ?? fallback["cents"])
        let semitoneValue = doubleValue(parameters["semitones"] ?? parameters["semitone"] ?? fallback["semitones"] ?? fallback["semitone"])
        let pitch = centsValue ?? semitoneValue.map { $0 * 100.0 } ?? 0
        guard pitch >= -2_400, pitch <= 2_400 else {
            throw PiMPAudioCommandFacadeError.invalidDSPParameters("Pitch shift must be between -2400 and 2400 cents.")
        }
        return PitchShiftParameters(pitch: Float(pitch))
    }

    private func editorRenderSelection(
        from arguments: [String: Any],
        loadedAudio: EditorLoadedAudio?
    ) throws -> InpaintSelection? {
        let selectionArguments = arguments["selection"] as? [String: Any]
        let startValue = editorSelectionValue(
            ["selectionStart", "startTime", "start"],
            arguments: arguments,
            selectionArguments: selectionArguments
        )
        let endValue = editorSelectionValue(
            ["selectionEnd", "endTime", "end"],
            arguments: arguments,
            selectionArguments: selectionArguments
        )
        guard startValue != nil || endValue != nil else { return nil }
        guard let startTime = doubleValue(startValue),
              let endTime = doubleValue(endValue) else {
            throw PiMPAudioCommandFacadeError.invalidEditorSelection
        }
        let selection = InpaintSelection(startTime: startTime, endTime: endTime)
        try validateEditorSelection(selection, loadedAudio: loadedAudio)
        return selection
    }

    private func editorRenderOutputURL(
        _ arguments: [String: Any],
        loadedAudio: EditorLoadedAudio?,
        format: EditorRenderFormat
    ) throws -> URL {
        if let rawPath = stringValue(arguments["outputPath"] ?? arguments["path"])?
            .trimmingCharacters(in: .whitespacesAndNewlines),
           !rawPath.isEmpty {
            return urlWithRenderExtension(localFileURL(from: rawPath), format: format)
        }

        let directory: URL
        if let rawDirectory = stringValue(arguments["outputDirectory"] ?? arguments["directory"])?
            .trimmingCharacters(in: .whitespacesAndNewlines),
           !rawDirectory.isEmpty {
            directory = localFileURL(from: rawDirectory)
        } else if let sourcePath = loadedAudio?.path {
            directory = URL(fileURLWithPath: sourcePath)
                .deletingLastPathComponent()
                .appending(path: "MCPExports")
        } else {
            directory = FileManager.default.temporaryDirectory.appending(path: "PiMPAudioMCPExports")
        }

        let fileName = stringValue(arguments["fileName"])?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let resolvedName = fileName?.isEmpty == false
            ? fileName!
            : defaultEditorRenderFileName(loadedAudio: loadedAudio, format: format)
        return urlWithRenderExtension(directory.appending(path: resolvedName), format: format)
    }

    private func defaultEditorRenderFileName(
        loadedAudio: EditorLoadedAudio?,
        format: EditorRenderFormat
    ) -> String {
        let sourceBase = loadedAudio
            .map { ($0.fileName as NSString).deletingPathExtension }
            .flatMap { $0.isEmpty ? nil : $0 } ?? "editor-audio"
        let timestamp = renderFileDateFormatter.string(from: Date())
        return "\(sanitizedFileStem(sourceBase))-mcp-render-\(timestamp).\(format.fileExtension)"
    }

    private func sanitizedFileStem(_ value: String) -> String {
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "-_ "))
        let sanitized = value.unicodeScalars.map { scalar in
            allowed.contains(scalar) ? Character(scalar) : "-"
        }
        let stem = String(sanitized).trimmingCharacters(in: .whitespacesAndNewlines)
        return stem.isEmpty ? "editor-audio" : stem
    }

    private func urlWithRenderExtension(_ url: URL, format: EditorRenderFormat) -> URL {
        let ext = url.pathExtension.lowercased()
        guard ext != format.fileExtension else { return url }
        guard ext.isEmpty else { return url.deletingPathExtension().appendingPathExtension(format.fileExtension) }
        return url.appendingPathExtension(format.fileExtension)
    }

    private func editorRenderFormat(from value: Any?) throws -> EditorRenderFormat {
        guard let rawValue = stringValue(value)?.trimmingCharacters(in: .whitespacesAndNewlines),
              !rawValue.isEmpty else {
            return .wav
        }
        guard let format = EditorRenderFormat(rawValue: rawValue.lowercased()) else {
            throw PiMPAudioCommandFacadeError.invalidRenderFormat(rawValue)
        }
        return format
    }

    private func optionalEditorOriginalPrompt(
        _ arguments: [String: Any],
        loadedAudio: EditorLoadedAudio?
    ) -> String? {
        let prompt = stringValue(arguments["originalPrompt"] ?? arguments["sourcePrompt"])?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if let prompt, !prompt.isEmpty {
            return prompt
        }
        return loadedAudio?.originalPrompt
    }

    private func editorEditMode(from value: Any?) throws -> EditMode {
        guard let rawValue = stringValue(value)?.trimmingCharacters(in: .whitespacesAndNewlines),
              !rawValue.isEmpty else {
            return .inpaint
        }
        if let mode = EditMode(rawValue: rawValue.lowercased()) {
            return mode
        }
        throw PiMPAudioCommandFacadeError.invalidEditMode(rawValue)
    }

    private func editorSelection(
        from arguments: [String: Any],
        editMode: EditMode,
        loadedAudio: EditorLoadedAudio?
    ) throws -> InpaintSelection {
        let selectionArguments = arguments["selection"] as? [String: Any]
        let startValue = editorSelectionValue(
            ["selectionStart", "startTime", "start"],
            arguments: arguments,
            selectionArguments: selectionArguments
        )
        let endValue = editorSelectionValue(
            ["selectionEnd", "endTime", "end"],
            arguments: arguments,
            selectionArguments: selectionArguments
        )

        let startTime: Double
        let endTime: Double
        if startValue != nil || endValue != nil {
            guard let parsedStart = doubleValue(startValue),
                  let parsedEnd = doubleValue(endValue) else {
                throw PiMPAudioCommandFacadeError.invalidEditorSelection
            }
            startTime = parsedStart
            endTime = parsedEnd
        } else if editMode == .repaint, let duration = loadedAudio?.duration {
            startTime = 0
            endTime = duration
        } else {
            throw PiMPAudioCommandFacadeError.missingRequiredArgument("selectionStart")
        }

        let selection = InpaintSelection(
            startTime: startTime,
            endTime: endTime,
            frequencyBand: try editorFrequencyBand(
                from: arguments,
                selectionArguments: selectionArguments
            )
        )
        try validateEditorSelection(selection, loadedAudio: loadedAudio)
        return selection
    }

    private func editorSelectionValue(
        _ keys: [String],
        arguments: [String: Any],
        selectionArguments: [String: Any]?
    ) -> Any? {
        for key in keys {
            if let value = arguments[key], !(value is NSNull) {
                return value
            }
            if let value = selectionArguments?[key], !(value is NSNull) {
                return value
            }
        }
        return nil
    }

    private func editorFrequencyBand(
        from arguments: [String: Any],
        selectionArguments: [String: Any]?
    ) throws -> FrequencyBand? {
        let bandArguments = arguments["frequencyBand"] as? [String: Any]
            ?? selectionArguments?["frequencyBand"] as? [String: Any]
        let lowValue = editorFrequencyValue(
            ["lowCutHz", "lowHz", "low"],
            arguments: arguments,
            bandArguments: bandArguments
        )
        let highValue = editorFrequencyValue(
            ["highCutHz", "highHz", "high"],
            arguments: arguments,
            bandArguments: bandArguments
        )

        guard lowValue != nil || highValue != nil else { return nil }
        guard let low = doubleValue(lowValue),
              let high = doubleValue(highValue),
              low >= 0,
              high > low,
              high <= 24_000 else {
            throw PiMPAudioCommandFacadeError.invalidFrequencyBand
        }
        return FrequencyBand(lowCutHz: Float(low), highCutHz: Float(high))
    }

    private func editorFrequencyValue(
        _ keys: [String],
        arguments: [String: Any],
        bandArguments: [String: Any]?
    ) -> Any? {
        for key in keys {
            if let value = arguments[key], !(value is NSNull) {
                return value
            }
            if let value = bandArguments?[key], !(value is NSNull) {
                return value
            }
        }
        return nil
    }

    private func validateEditorSelection(
        _ selection: InpaintSelection,
        loadedAudio: EditorLoadedAudio?
    ) throws {
        guard selection.startTime >= 0, selection.endTime > selection.startTime else {
            throw PiMPAudioCommandFacadeError.invalidEditorSelection
        }
        if let duration = loadedAudio?.duration,
           selection.endTime > duration + 0.001 {
            throw PiMPAudioCommandFacadeError.invalidEditorSelection
        }
    }

    private func editorInpaintStrength(from arguments: [String: Any]) throws -> Float {
        let preset = try editorInpaintPreset(from: arguments)
        let defaultStrength = preset?.strength ?? 0.6
        let rawValue = arguments["strength"] ?? arguments["denoiseStrength"] ?? arguments["editStrength"]
        guard let rawValue, !(rawValue is NSNull) else { return defaultStrength }
        guard let value = doubleValue(rawValue), value >= 0.0, value <= 1.0 else {
            throw PiMPAudioCommandFacadeError.invalidEditorStrength
        }
        return Float(value)
    }

    private func editorInpaintGuidanceScale(from arguments: [String: Any]) throws -> Float {
        let preset = try editorInpaintPreset(from: arguments)
        let defaultGuidance = preset?.guidanceScale ?? 7.0
        let rawValue = arguments["guidanceScale"] ?? arguments["guidance"]
        guard let rawValue, !(rawValue is NSNull) else { return defaultGuidance }
        guard let value = doubleValue(rawValue), value > 0 else {
            throw PiMPAudioCommandFacadeError.invalidGuidanceScale
        }
        return Float(value)
    }

    private func editorInpaintPreset(from arguments: [String: Any]) throws -> InpaintPreset? {
        guard let rawValue = stringValue(arguments["preset"])?
            .trimmingCharacters(in: .whitespacesAndNewlines),
              !rawValue.isEmpty else {
            return nil
        }
        guard let preset = InpaintPreset(rawValue: rawValue.lowercased()) else {
            throw PiMPAudioCommandFacadeError.invalidInpaintPreset(rawValue)
        }
        return preset
    }

    private func editorPipelineChoice(from arguments: [String: Any]) throws -> PipelineChoice {
        let rawValue = arguments["pipeline"] ?? arguments["pipelineOverride"]
        guard let rawValue, !(rawValue is NSNull) else {
            return .auto
        }
        guard let pipeline = pipelineChoice(from: rawValue) else {
            throw PiMPAudioCommandFacadeError.invalidPipeline(stringValue(rawValue) ?? "\(rawValue)")
        }
        return pipeline
    }

    private func editorAudioLoadSource(from arguments: [String: Any]) throws -> EditorAudioLoadSource {
        let id = stringValue(
            arguments["historyID"]
                ?? arguments["historyId"]
                ?? arguments["audioID"]
                ?? arguments["audioId"]
                ?? arguments["id"]
        )?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        if !id.isEmpty {
            let entry = try resolveHistoryEntry(id)
            return EditorAudioLoadSource(
                kind: "history",
                url: historyStore.fileURL(for: entry),
                fileName: entry.fileName,
                historyEntryID: entry.id.uuidString,
                originalPrompt: entry.prompt
            )
        }

        let rawPath = try requiredString("path", aliases: ["audioPath", "filePath"], in: arguments)
        let url = localFileURL(from: rawPath)
        return EditorAudioLoadSource(
            kind: "path",
            url: url,
            fileName: url.lastPathComponent,
            historyEntryID: nil,
            originalPrompt: nil
        )
    }

    private func editorActionsPayload() -> [[String: Any]] {
        [
            [
                "id": "loadAudio",
                "tool": "load_editor_audio",
                "operationKind": "editor.loadAudio",
                "status": "implemented",
                "description": "Load a local audio file into the editor coordinator and encode editable latents."
            ],
            [
                "id": "inpaint",
                "tool": "submit_editor_inpaint",
                "operationKind": "editor.inpaint",
                "status": "implemented",
                "description": "Regenerate a selected region with explicit confirmation and pollable progress."
            ],
            [
                "id": "applyEffectChain",
                "tool": "apply_editor_effect_chain",
                "operationKind": "editor.applyEffectChain",
                "status": "implemented",
                "description": "Apply confirmed bounded DSP effect chains to the loaded editor buffer."
            ],
            [
                "id": "renderAudio",
                "tool": "render_editor_audio",
                "operationKind": "editor.renderAudio",
                "status": "implemented",
                "description": "Bounce a selected region or full editor buffer to a lossless output file."
            ]
        ]
    }

    private func localFileURL(from rawPath: String) -> URL {
        let expanded = (rawPath as NSString).expandingTildeInPath
        return URL(fileURLWithPath: expanded)
    }

    private func validateReadableFile(_ url: URL) throws {
        let path = url.path(percentEncoded: false)
        var isDirectory = ObjCBool(false)
        let exists = FileManager.default.fileExists(atPath: path, isDirectory: &isDirectory)
        guard exists, !isDirectory.boolValue else {
            throw PiMPAudioCommandFacadeError.editorAudioFileNotFound(path)
        }
    }

    private func stateName(_ state: GenerationState) -> String {
        switch state {
        case .idle: return "idle"
        case .downloadingModel: return "downloadingModel"
        case .loadingWeights: return "loadingWeights"
        case .conditioning: return "conditioning"
        case .tokenizingSource: return "tokenizingSource"
        case .lmAudioCodeConditioning: return "lmAudioCodeConditioning"
        case .generating: return "generating"
        case .decoding: return "decoding"
        case .complete: return "complete"
        case .error: return "error"
        case .cancelled: return "cancelled"
        }
    }

    private func persistAdapterSelection(_ selection: [AdapterSelection]) {
        let defaults = UserDefaults.standard
        if let primary = selection.first {
            defaults.set(primary.adapter.id, forKey: AppUserDefaultsKeys.selectedAdapterID)
            defaults.set(Double(primary.strength), forKey: AppUserDefaultsKeys.selectedAdapterStrength)
        } else {
            defaults.removeObject(forKey: AppUserDefaultsKeys.selectedAdapterID)
            defaults.set(1.0, forKey: AppUserDefaultsKeys.selectedAdapterStrength)
        }
    }

    private func notifyAdapterSelectionChanged(_ selection: [AdapterSelection]) {
        NotificationCenter.default.post(
            name: .mcpAdapterSelectionDidChange,
            object: nil,
            userInfo: [
                "adapters": selection.map(adapterSelectionEntryPayload)
            ]
        )
    }

    private func rowPayload(_ row: TraceComparisonRow) -> [String: Any] {
        [
            "label": row.label,
            "left": row.left,
            "right": row.right,
            "isDifferent": row.isDifferent
        ]
    }

    private func encodedJSONObject<T: Encodable>(_ value: T) throws -> [String: Any] {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        let data = try encoder.encode(value)
        guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw PiMPAudioCommandFacadeError.encodingFailed
        }
        return object
    }

    private func stringValue(_ value: Any?) -> String? {
        switch value {
        case is NSNull:
            return nil
        case let value as String:
            return value
        case let value as UUID:
            return value.uuidString
        case let value as CustomStringConvertible:
            return value.description
        default:
            return nil
        }
    }

    private func boolValue(_ value: Any?) -> Bool? {
        switch value {
        case let value as Bool:
            return value
        case let value as NSNumber:
            return value.boolValue
        case let value as String:
            switch value.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
            case "true", "yes", "1":
                return true
            case "false", "no", "0":
                return false
            default:
                return nil
            }
        default:
            return nil
        }
    }

    private func integerValue(_ value: Any?) -> Int? {
        switch value {
        case let value as Int:
            return value
        case let value as NSNumber:
            return value.intValue
        case let value as String:
            return Int(value.trimmingCharacters(in: .whitespacesAndNewlines))
        default:
            return nil
        }
    }

    private func hasArgument(_ key: String, in arguments: [String: Any]) -> Bool {
        arguments.keys.contains(key)
    }

    private func doubleValue(_ value: Any?) -> Double? {
        switch value {
        case let value as Double:
            return value
        case let value as Float:
            return Double(value)
        case let value as Int:
            return Double(value)
        case let value as NSNumber:
            return value.doubleValue
        case let value as String:
            return Double(value.trimmingCharacters(in: .whitespacesAndNewlines))
        default:
            return nil
        }
    }

    private func stringArrayValue(_ value: Any?) -> [String]? {
        switch value {
        case is NSNull:
            return nil
        case let values as [String]:
            return values
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { !$0.isEmpty }
        case let values as [Any]:
            return values
                .compactMap(stringValue)
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { !$0.isEmpty }
        case let raw as String:
            return raw
                .split { character in
                    character == "," || character == "\n" || character == ";"
                }
                .map { String($0).trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { !$0.isEmpty }
        default:
            return nil
        }
    }

    private func requiredString(_ key: String, in arguments: [String: Any]) throws -> String {
        let value = stringValue(arguments[key])?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        guard !value.isEmpty else {
            throw PiMPAudioCommandFacadeError.missingRequiredArgument(key)
        }
        return value
    }

    private func requiredString(
        _ key: String,
        aliases: [String],
        in arguments: [String: Any]
    ) throws -> String {
        for candidate in [key] + aliases {
            let value = stringValue(arguments[candidate])?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            if !value.isEmpty {
                return value
            }
        }
        throw PiMPAudioCommandFacadeError.missingRequiredArgument(key)
    }

    private func requiredDuration(in arguments: [String: Any]) throws -> Double {
        guard let duration = doubleValue(arguments["duration"] ?? arguments["durationSeconds"]),
              duration > 0 else {
            throw PiMPAudioCommandFacadeError.invalidDuration
        }
        return duration
    }

    private func generationMode(from value: Any?) -> GenerationMode? {
        guard let rawValue = stringValue(value) else { return nil }
        if let mode = GenerationMode(rawValue: rawValue) {
            return mode
        }
        return GenerationMode(fromClassification: rawValue)
    }

    private func pipelineChoice(from value: Any?) -> PipelineChoice? {
        guard let rawValue = stringValue(value)?.trimmingCharacters(in: .whitespacesAndNewlines) else {
            return nil
        }
        if let pipeline = PipelineChoice(rawValue: rawValue) {
            return pipeline
        }
        switch rawValue.lowercased() {
        case "music", "acestep", "ace-step", "ace_step":
            return .aceStep
        case "sfx", "soundfx", "sound_fx", "sound fx", "stableaudio", "stable-audio", "stable_audio":
            return .stableAudio
        default:
            return nil
        }
    }

    private func inferMethod(from value: Any?) -> InferMethod? {
        guard let rawValue = stringValue(value)?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() else {
            return nil
        }
        return InferMethod(rawValue: rawValue)
    }

    private func precision(from value: Any?) -> Precision? {
        guard let rawValue = stringValue(value)?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() else {
            return nil
        }
        return Precision(rawValue: rawValue)
    }

    private func guidancePreset(from value: Any?) -> GuidancePreset? {
        guard let rawValue = stringValue(value)?.trimmingCharacters(in: .whitespacesAndNewlines) else {
            return nil
        }
        if let preset = GuidancePreset(rawValue: rawValue) {
            return preset
        }
        let compact = rawValue
            .lowercased()
            .replacingOccurrences(of: " ", with: "")
            .replacingOccurrences(of: "-", with: "")
        switch compact {
        case "constant":
            return .constant
        case "lateemphasis", "late":
            return .lateEmphasis
        case "midtolate", "midlate":
            return .midToLate
        case "softlate", "soft":
            return .softLate
        default:
            return nil
        }
    }

    private func normalizedText(_ text: String) -> String {
        text.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func lyricLineCount(_ lyrics: String) -> Int {
        lyrics
            .components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty && !$0.hasPrefix("[") }
            .count
    }

    private func progressFraction(current: Int, total: Int) -> Double {
        guard total > 0 else { return 0 }
        return min(1, max(0, Double(current) / Double(total)))
    }

    private func diagnosticsStatusPayload() -> [String: Any] {
        [
            "enabled": AudioDiagnostics.isEnabled,
            "directory": (AudioDiagnostics.configuredLogDirectoryURL ?? AppStorageLocations.diagnosticsDirectory)
                .path(percentEncoded: false),
            "activeLogPath": AudioDiagnostics.activeLogFileURL?.path(percentEncoded: false) ?? NSNull(),
            "latestLogPath": latestDiagnosticLogURL()?.path(percentEncoded: false) ?? NSNull()
        ]
    }

    private func resolveDiagnosticLogURL(arguments: [String: Any]) throws -> URL {
        let diagnosticsDirectory = AudioDiagnostics.configuredLogDirectoryURL
            ?? AppStorageLocations.diagnosticsDirectory

        if let path = stringValue(arguments["path"] ?? arguments["logPath"])?
            .trimmingCharacters(in: .whitespacesAndNewlines),
           !path.isEmpty {
            let url = URL(fileURLWithPath: path)
            guard isInDiagnosticsDirectory(url) else {
                throw PiMPAudioCommandFacadeError.resourceNotFound(path)
            }
            guard FileManager.default.fileExists(atPath: url.path(percentEncoded: false)) else {
                throw PiMPAudioCommandFacadeError.resourceNotFound(path)
            }
            return url
        }

        if let fileName = stringValue(arguments["fileName"] ?? arguments["name"])?
            .trimmingCharacters(in: .whitespacesAndNewlines),
           !fileName.isEmpty {
            let url = diagnosticsDirectory.appendingPathComponent(fileName)
            guard FileManager.default.fileExists(atPath: url.path(percentEncoded: false)) else {
                throw PiMPAudioCommandFacadeError.resourceNotFound(fileName)
            }
            return url
        }

        if let activeURL = AudioDiagnostics.activeLogFileURL,
           FileManager.default.fileExists(atPath: activeURL.path(percentEncoded: false)) {
            return activeURL
        }

        if let latestURL = latestDiagnosticLogURL() {
            return latestURL
        }

        throw PiMPAudioCommandFacadeError.resourceNotFound("latest diagnostics log")
    }

    private func latestDiagnosticLogURL() -> URL? {
        let diagnosticsDirectory = AudioDiagnostics.configuredLogDirectoryURL
            ?? AppStorageLocations.diagnosticsDirectory
        guard let contents = try? FileManager.default.contentsOfDirectory(
            at: diagnosticsDirectory,
            includingPropertiesForKeys: [.contentModificationDateKey, .isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else {
            return nil
        }

        return contents
            .filter { url in
                let values = try? url.resourceValues(forKeys: [.isRegularFileKey])
                return values?.isRegularFile == true
                    && url.lastPathComponent.hasPrefix("PiMPAudioStudio-AUDIODIAG-")
                    && url.pathExtension == "log"
            }
            .sorted { modificationDate($0) > modificationDate($1) }
            .first
    }

    private func tailLines(
        from url: URL,
        limit: Int,
        containing filter: String?
    ) throws -> [String] {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }

        let fileSize = try handle.seekToEnd()
        let maxBytes: UInt64 = 1_048_576
        let readSize = min(fileSize, maxBytes)
        try handle.seek(toOffset: fileSize - readSize)
        let data = handle.readDataToEndOfFile()

        var text = String(decoding: data, as: UTF8.self)
        if readSize < fileSize, let firstNewline = text.firstIndex(of: "\n") {
            text = String(text[text.index(after: firstNewline)...])
        }

        let allLines = text
            .split(separator: "\n", omittingEmptySubsequences: false)
            .map(String.init)
        let filteredLines: [String]
        if let filter, !filter.isEmpty {
            filteredLines = allLines.filter {
                $0.localizedCaseInsensitiveContains(filter)
            }
        } else {
            filteredLines = allLines
        }
        return Array(filteredLines.suffix(limit))
    }

    private func isInDiagnosticsDirectory(_ url: URL) -> Bool {
        let diagnosticsDirectory = (AudioDiagnostics.configuredLogDirectoryURL
            ?? AppStorageLocations.diagnosticsDirectory)
            .standardizedFileURL
            .path(percentEncoded: false)
        let candidate = url.standardizedFileURL.path(percentEncoded: false)
        return candidate == diagnosticsDirectory
            || candidate.hasPrefix(diagnosticsDirectory + "/")
    }

    private func modificationDate(_ url: URL) -> Date {
        (try? url.resourceValues(forKeys: [.contentModificationDateKey]))?
            .contentModificationDate ?? .distantPast
    }

    private func increment(_ dictionary: inout [String: Int], key: String) {
        dictionary[key, default: 0] += 1
    }

    private func compact(_ text: String, limit: Int) -> String {
        let oneLine = text
            .replacingOccurrences(of: "\n", with: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard oneLine.count > limit else { return oneLine }
        return String(oneLine.prefix(max(0, limit - 3))) + "..."
    }

    private func formatGiB(_ bytes: UInt64) -> String {
        String(format: "%.1f GiB", Double(bytes) / 1_073_741_824)
    }

    private func resource(
        uri: String,
        name: String,
        title: String,
        description: String
    ) -> [String: Any] {
        [
            "uri": uri,
            "name": name,
            "title": title,
            "description": description,
            "mimeType": "application/json"
        ]
    }

    private func resourceURI(kind: String, identifier: String? = nil) -> String {
        guard let identifier else { return "\(AppRuntimeIdentity.mcpScheme)://\(kind)" }
        let encoded = identifier.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? identifier
        return "\(AppRuntimeIdentity.mcpScheme)://\(kind)/\(encoded)"
    }

    private func confirmationRequirementPayload(toolName: String) -> [String: Any] {
        [
            "required": true,
            "confirmArgument": "confirm",
            "confirmValue": true,
            "supportedModes": [
                "confirmArgument",
                "elicitation/form"
            ],
            "fallback": "Clients without MCP elicitation support must call \(toolName) with confirm: true after explicit user approval."
        ]
    }
}

private enum MCPTrainingRecipe: String {
    case local
    case cloudStyle
    case cloudVoice
    case wide
}

private enum MCPTrainingProjectionCoverage {
    case queryValue
    case fullAttention
}

private struct MCPTrainingDefaults {
    let dimension: Int
    let epochs: Int
    let layerCount: Int
    let projectionCoverage: MCPTrainingProjectionCoverage
    let alphaScale: Float
    let dropout: Float
    let targetLayers: Set<TargetLayer>
    let gradientAccumulationSteps: Int
    let learningRate: Float
}

private struct MCPTrainingRequest {
    let kind: AdapterKind
    let recipe: MCPTrainingRecipe
    let dataset: DatasetPackageDocument
    let importedFiles: [TrainingImportedFile]
    let adapterName: String
    let config: LoRAConfig
    let lokrConfig: LoKRConfig?
}

private enum PiMPAudioCommandFacadeError: LocalizedError {
    case missingTraceIdentifier
    case traceNotFound(String)
    case historyEntryNotFound(String)
    case invalidResourceURI(String)
    case resourceNotFound(String)
    case adapterNotFound(String)
    case incompatibleAdapter(String, String)
    case invalidAdapterStack
    case invalidBoolean(String)
    case invalidStrength
    case invalidSeed
    case invalidSteps
    case tooManyAdapters(Int)
    case missingRequiredArgument(String)
    case invalidDuration
    case outputDirectoryUnavailable
    case generationAlreadyActive(String)
    case confirmationRequired(String)
    case generationServiceUnavailable
    case editorServiceUnavailable
    case editorAudioNotLoaded
    case editorAudioFileNotFound(String)
    case invalidEditMode(String)
    case invalidEditorSelection
    case invalidFrequencyBand
    case invalidEditorStrength
    case invalidGuidanceScale
    case invalidInpaintPreset(String)
    case invalidPipeline(String)
    case editorInpaintDidNotComplete
    case invalidRenderFormat(String)
    case editorRenderFileExists(String)
    case editorRenderWriteFailed(String, String)
    case editorDSPChainEmpty
    case invalidDSPChain
    case invalidDSPEffect(String)
    case invalidDSPTarget(String)
    case invalidDSPParameters(String)
    case operationNotFound(String)
    case trainingServiceUnavailable
    case trainingAlreadyActive(String)
    case invalidTrainingKind(String)
    case invalidTrainingRecipe(String)
    case invalidTrainingConfiguration(String)
    case encodingFailed

    var errorDescription: String? {
        switch self {
        case .missingTraceIdentifier:
            return "Provide a history id, trace filename, or audio filename."
        case .traceNotFound(let identifier):
            return "No generation trace matched '\(identifier)'."
        case .historyEntryNotFound(let identifier):
            return "No history entry matched '\(identifier)'."
        case .invalidResourceURI(let uri):
            return "Invalid PiMP Audio Studio resource URI: \(uri)."
        case .resourceNotFound(let uri):
            return "No PiMP Audio Studio resource matched '\(uri)'."
        case .adapterNotFound(let identifier):
            return "No LoRA adapter matched '\(identifier)'."
        case .incompatibleAdapter(let name, let category):
            return "'\(name)' is a \(category) adapter and cannot be applied to ACE-Step generation."
        case .invalidAdapterStack:
            return "'adapters' must be an array of adapter objects."
        case .invalidBoolean(let key):
            return "'\(key)' must be true or false."
        case .invalidStrength:
            return "Adapter strength must be a number from 0.0 to 1.0."
        case .invalidSeed:
            return "Seed must be an integer or 'random'."
        case .invalidSteps:
            return "Steps must be a positive integer or 'auto'."
        case .tooManyAdapters(let maxCount):
            return "Adapter stack can contain at most \(maxCount) adapters."
        case .missingRequiredArgument(let key):
            return "Missing required argument '\(key)'."
        case .invalidDuration:
            return "Provide a positive duration in seconds."
        case .outputDirectoryUnavailable:
            return "No output directory is selected."
        case .generationAlreadyActive(let state):
            return "Generation is already active (state: \(state))."
        case .confirmationRequired(let action):
            return "Refusing to run '\(action)' without confirm: true."
        case .generationServiceUnavailable:
            return "Generation service is not ready yet."
        case .editorServiceUnavailable:
            return "Editor service is not ready yet."
        case .editorAudioNotLoaded:
            return "Load audio into the editor before submitting an edit."
        case .editorAudioFileNotFound(let path):
            return "No readable editor audio file exists at '\(path)'."
        case .invalidEditMode(let value):
            return "Unsupported editor edit mode '\(value)'. Use 'inpaint' or 'repaint'."
        case .invalidEditorSelection:
            return "Editor selection must have startTime >= 0, endTime > startTime, and fit within the loaded audio duration when audio is loaded."
        case .invalidFrequencyBand:
            return "Frequency band must provide lowCutHz >= 0, highCutHz > lowCutHz, and highCutHz <= 24000."
        case .invalidEditorStrength:
            return "Editor edit strength must be a number from 0.0 to 1.0."
        case .invalidGuidanceScale:
            return "Guidance scale must be a positive number."
        case .invalidInpaintPreset(let value):
            return "Unsupported inpaint preset '\(value)'. Use 'subtle', 'moderate', or 'full'."
        case .invalidPipeline(let value):
            return "Unsupported pipeline '\(value)'. Use 'auto', 'aceStep', or 'stableAudio'."
        case .editorInpaintDidNotComplete:
            return "Editor inpaint ended before reporting completion."
        case .invalidRenderFormat(let value):
            return "Unsupported editor render format '\(value)'. Use 'wav' or 'aiff'."
        case .editorRenderFileExists(let path):
            return "Editor render output already exists at '\(path)'. Pass overwrite: true or choose another path."
        case .editorRenderWriteFailed(let path, let reason):
            return "Could not write editor render output to '\(path)': \(reason)"
        case .editorDSPChainEmpty:
            return "Editor DSP effect chain must include at least one effect."
        case .invalidDSPChain:
            return "'effects' must be an array of effect names or effect objects."
        case .invalidDSPEffect(let value):
            return "Unsupported editor DSP effect '\(value)'."
        case .invalidDSPTarget(let reason):
            return reason
        case .invalidDSPParameters(let reason):
            return reason
        case .operationNotFound(let id):
            return "No MCP operation matched '\(id)'."
        case .trainingServiceUnavailable:
            return "Training service is not ready yet."
        case .trainingAlreadyActive(let status):
            return "Training is already active (state: \(status)). Stop it before starting another MCP training run."
        case .invalidTrainingKind(let value):
            return "Unsupported training kind '\(value)'. Use 'lora' or 'lokr'."
        case .invalidTrainingRecipe(let value):
            return "Unsupported training recipe '\(value)'. Use local, cloudStyle/cloudVoice for LoRA, or wide for LoKR."
        case .invalidTrainingConfiguration(let reason):
            return reason
        case .encodingFailed:
            return "Could not encode the trace as JSON."
        }
    }
}

private struct AdapterSelection {
    let adapter: LoRAAdapter
    let strength: Float
}

private struct EditorAudioLoadSource {
    let kind: String
    let url: URL
    let fileName: String
    let historyEntryID: String?
    let originalPrompt: String?
}

private struct EditorRenderRequest {
    let format: EditorRenderFormat
    let outputURL: URL
    let selection: InpaintSelection?
    let overwrite: Bool
}

private struct EditorDSPRequest {
    let steps: [EditorDSPEffectStep]
    let description: String
}

private struct EditorDSPEffectStep {
    let effect: DSPEffect
    let target: ApplyTarget
    let parameters: DSPParameters
}

private enum EditorRenderFormat: String {
    case wav
    case aiff

    var fileExtension: String {
        switch self {
        case .wav: return "wav"
        case .aiff: return "aiff"
        }
    }
}

private enum MCPOperationStatus: String {
    case running
    case succeeded
    case failed
    case cancelled

    var isTerminal: Bool {
        switch self {
        case .running:
            return false
        case .succeeded, .failed, .cancelled:
            return true
        }
    }
}

private struct MCPOperationRecord {
    let id: String
    let kind: String
    let title: String
    let createdAt: Date
    var updatedAt: Date
    var status: MCPOperationStatus
    var request: [String: Any]
    var progress: [String: Any]?
    var result: [String: Any]?
    var error: String?
    var cancellationReason: String?
}

private struct WorkflowPromptArgument {
    let name: String
    let description: String
    let required: Bool
}

private struct WorkflowPromptDefinition {
    let id: String
    let title: String
    let description: String
    let promptName: String
    let arguments: [WorkflowPromptArgument]
    let resources: [String]
    let tools: [String]
    let template: String
}

private let workflowPromptDefinitions: [WorkflowPromptDefinition] = [
    WorkflowPromptDefinition(
        id: "improve_chorus",
        title: "Improve Chorus",
        description: "Plan and safely execute a chorus-focused rewrite or polish pass.",
        promptName: "improve_chorus",
        arguments: [
            WorkflowPromptArgument(
                name: "takeID",
                description: "Optional take/history id to load before editing.",
                required: false
            ),
            WorkflowPromptArgument(
                name: "regionStart",
                description: "Optional chorus region start time in seconds.",
                required: false
            ),
            WorkflowPromptArgument(
                name: "regionEnd",
                description: "Optional chorus region end time in seconds.",
                required: false
            ),
            WorkflowPromptArgument(
                name: "direction",
                description: "Creative direction for the improved chorus.",
                required: false
            )
        ],
        resources: [
            "pimpaudio://takes",
            "pimpaudio://track/current",
            "pimpaudio://clip/current",
            "pimpaudio://region/current-selection",
            "pimpaudio://presets"
        ],
        tools: [
            "load_editor_audio",
            "submit_editor_inpaint",
            "apply_editor_effect_chain",
            "render_editor_audio",
            "get_operation"
        ],
        template: """
        Improve the chorus while preserving the best material around it.
        Inspect current takes, the loaded clip, the current/last region, and presets first.
        If a takeID is provided, load it with load_editor_audio and poll the operation before editing.
        Validate edits with dryRun before any mutation; require confirm: true or MCP elicitation approval for inpaint or DSP changes.
        Use direction as the creative target, or infer a conservative chorus polish if direction is blank.
        """
    ),
    WorkflowPromptDefinition(
        id: "create_mix_variants",
        title: "Create Mix Variants",
        description: "Prepare alternate rendered mix treatments from the current take or editor buffer.",
        promptName: "create_mix_variants",
        arguments: [
            WorkflowPromptArgument(
                name: "takeID",
                description: "Optional take/history id to load before creating variants.",
                required: false
            ),
            WorkflowPromptArgument(
                name: "variantCount",
                description: "Requested number of variants. Defaults to three.",
                required: false
            ),
            WorkflowPromptArgument(
                name: "direction",
                description: "Mix direction such as brighter, wider, tighter, or more ambient.",
                required: false
            )
        ],
        resources: [
            "pimpaudio://takes",
            "pimpaudio://track/current",
            "pimpaudio://clip/current",
            "pimpaudio://presets"
        ],
        tools: [
            "load_editor_audio",
            "apply_editor_effect_chain",
            "render_editor_audio",
            "mark_preferred_take",
            "get_operation"
        ],
        template: """
        Create mix variants from the loaded editor audio or the supplied takeID.
        Prefer reversible, bounded DSP chains and validate each chain with dryRun before applying.
        Render each variant with render_editor_audio and keep filenames descriptive.
        Do not mark a preferred take unless the user explicitly asks.
        """
    ),
    WorkflowPromptDefinition(
        id: "clean_up_region",
        title: "Clean Up Region",
        description: "Clean or replace a bounded region while preserving surrounding audio.",
        promptName: "clean_up_region",
        arguments: [
            WorkflowPromptArgument(
                name: "takeID",
                description: "Optional take/history id to load before editing.",
                required: false
            ),
            WorkflowPromptArgument(
                name: "regionStart",
                description: "Region start time in seconds.",
                required: true
            ),
            WorkflowPromptArgument(
                name: "regionEnd",
                description: "Region end time in seconds.",
                required: true
            ),
            WorkflowPromptArgument(
                name: "issue",
                description: "Problem to solve, such as click, weak vocal, or muddy transient.",
                required: false
            )
        ],
        resources: [
            "pimpaudio://track/current",
            "pimpaudio://clip/current",
            "pimpaudio://region/current-selection",
            "pimpaudio://presets"
        ],
        tools: [
            "load_editor_audio",
            "submit_editor_inpaint",
            "apply_editor_effect_chain",
            "render_editor_audio",
            "get_operation"
        ],
        template: """
        Clean the specified region only.
        Use regionStart and regionEnd as the selection boundaries, and keep every mutation bounded to that range unless the user says otherwise.
        Prefer DSP for mechanical cleanup and submit_editor_inpaint for musical replacement.
        Always run dryRun first, then require confirm: true or MCP elicitation approval before applying an editor mutation.
        """
    ),
    WorkflowPromptDefinition(
        id: "prepare_stems_for_review",
        title: "Prepare Stems for Review",
        description: "Render review-ready audio files from the current workstation state.",
        promptName: "prepare_stems_for_review",
        arguments: [
            WorkflowPromptArgument(
                name: "takeID",
                description: "Optional take/history id to load before rendering.",
                required: false
            ),
            WorkflowPromptArgument(
                name: "outputDirectory",
                description: "Optional directory for rendered review exports.",
                required: false
            )
        ],
        resources: [
            "pimpaudio://takes",
            "pimpaudio://track/current",
            "pimpaudio://clip/current",
            "pimpaudio://operations"
        ],
        tools: [
            "load_editor_audio",
            "render_editor_audio",
            "mark_preferred_take",
            "get_operation"
        ],
        template: """
        Prepare review-ready renders from the current clip or the supplied takeID.
        Use render_editor_audio with lossless output and a clear file name.
        Poll operations until renders complete and summarize paths for review.
        Do not delete or overwrite existing files unless overwrite is explicitly requested.
        """
    ),
    WorkflowPromptDefinition(
        id: "production_ready_prompt",
        title: "Production Ready Prompt",
        description: "Turn a rough music or SFX idea into an PiMP Audio Studio generation request.",
        promptName: "production_ready_prompt",
        arguments: [
            WorkflowPromptArgument(
                name: "prompt",
                description: "Rough prompt or creative brief to develop.",
                required: true
            ),
            WorkflowPromptArgument(
                name: "duration",
                description: "Optional target duration in seconds.",
                required: false
            )
        ],
        resources: [
            "pimpaudio://models",
            "pimpaudio://adapters",
            "pimpaudio://presets",
            "pimpaudio://operations"
        ],
        tools: [
            "enhance_prompt",
            "write_lyrics",
            "apply_adapter",
            "generate_song",
            "generation_status"
        ],
        template: """
        Convert the supplied prompt into a production-ready PiMP Audio Studio generation request.
        Inspect models, adapters, and presets first so the request fits the current workstation.
        Use enhance_prompt and write_lyrics when useful, then validate the final generate_song request with dryRun before submitting.
        """
    )
]

private struct PiMPAudioResourceURI {
    let kind: String
    let identifier: String?

    init?(_ rawValue: String) {
        guard let components = URLComponents(string: rawValue),
              components.scheme == AppRuntimeIdentity.mcpScheme ||
                components.scheme == AppRuntimeIdentity.legacyMCPScheme,
              let host = components.host,
              !host.isEmpty else {
            return nil
        }

        let path = components.percentEncodedPath
            .split(separator: "/", omittingEmptySubsequences: true)
            .map(String.init)

        self.kind = host
        if let first = path.first {
            self.identifier = first.removingPercentEncoding ?? first
        } else {
            self.identifier = nil
        }
    }
}

private extension ModelStatus {
    var mcpDescription: [String: Any] {
        switch self {
        case .notDownloaded:
            return ["state": "notDownloaded"]
        case .downloading(let progress):
            return ["state": "downloading", "progress": progress]
        case .verifying:
            return ["state": "verifying"]
        case .ready:
            return ["state": "ready"]
        case .failed(let message):
            return ["state": "failed", "message": message]
        }
    }
}
