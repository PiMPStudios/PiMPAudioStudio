import Foundation
import AVFoundation
import PiMPAudioCore

extension PiMPAudioCommandFacade {
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
                    "weightDecompose": adapter.lokrConfig.map { $0.weightDecompose as Any } ?? NSNull(),
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


    func modelResource(id: String) async throws -> [String: Any] {
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

    func adapterResource(id: String) async throws -> [String: Any] {
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

    func audioResource(identifier: String) throws -> [String: Any] {
        let entry = try resolveHistoryEntry(identifier)
        let audioURL = try historyStore.fileURL(for: entry)
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

    func validateResourceIdentifier(_ identifier: String?, expected: String, uri: String) throws {
        guard identifier == expected else {
            throw PiMPAudioCommandFacadeError.invalidResourceURI(uri)
        }
    }

    func projectResource() async throws -> [String: Any] {
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

    func currentTrackResource() async -> [String: Any] {
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

    func currentClipResource() async -> [String: Any] {
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

    func currentRegionResource() async -> [String: Any] {
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

    func takesResource(limit: Int) -> [String: Any] {
        let cappedLimit = max(1, min(limit, 200))
        let rows = historyStore.entries.prefix(cappedLimit).map(takeSummary)
        return [
            "kind": "takeCollection",
            "count": rows.count,
            "totalCount": historyStore.entries.count,
            "takes": Array(rows)
        ]
    }

    func takeResource(identifier: String) throws -> [String: Any] {
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

    func presetsResource() -> [String: Any] {
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

    func presetResource(identifier: String) throws -> [String: Any] {
        let normalized = identifier.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard let preset = loadGenerationPresets().first(where: {
            $0.id.uuidString.lowercased() == normalized || $0.name.lowercased() == normalized
        }) else {
            throw PiMPAudioCommandFacadeError.resourceNotFound(resourceURI(kind: "preset", identifier: identifier))
        }
        return generationPresetPayload(preset)
    }

    func workflowsResource() -> [String: Any] {
        [
            "kind": "workflowCollection",
            "count": workflowPromptDefinitions.count,
            "workflows": workflowPromptDefinitions.map(workflowPromptPayload)
        ]
    }

    func workflowResource(identifier: String) throws -> [String: Any] {
        guard let workflow = workflowPromptDefinitions.first(where: { $0.id == identifier }) else {
            throw PiMPAudioCommandFacadeError.resourceNotFound(resourceURI(kind: "workflow", identifier: identifier))
        }
        var payload = workflowPromptPayload(workflow)
        payload["template"] = workflow.template
        return payload
    }

    func takeSummary(_ entry: HistoryEntry) -> [String: Any] {
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

    func historyAdapterPayload(_ adapter: HistoryAdapterSnapshot) -> [String: Any] {
        [
            "id": adapter.id,
            "name": adapter.name ?? NSNull(),
            "weight": adapter.weight,
            "source": adapter.source ?? NSNull(),
            "rank": adapter.rank ?? NSNull(),
            "baseModelID": adapter.baseModelID ?? NSNull()
        ]
    }

    func loadGenerationPresets() -> [GenerationPreset] {
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

    func generationPresetPayload(_ preset: GenerationPreset) -> [String: Any] {
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

    func guidancePresetPayload(_ preset: GuidancePreset) -> [String: Any] {
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

    func inpaintPresetPayload(_ preset: InpaintPreset) -> [String: Any] {
        [
            "id": preset.rawValue,
            "displayName": preset.rawValue.capitalized,
            "strength": preset.strength,
            "guidanceScale": preset.guidanceScale
        ]
    }

    func reverbPresetPayload(_ preset: ReverbPreset) -> [String: Any] {
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

    func dspEffectPayload(_ effect: DSPEffect) -> [String: Any] {
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

    func workflowPromptPayload(_ workflow: WorkflowPromptDefinition) -> [String: Any] {
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

    func resolveHistoryEntry(_ identifier: String) throws -> HistoryEntry {
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

    func resolveTrace(arguments: [String: Any]) throws -> GenerationTrace {
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

    func historySummary(_ entry: HistoryEntry) -> [String: Any] {
        let audioPath: Any = (try? historyStore.fileURL(for: entry))?.path(percentEncoded: false) ?? NSNull()
        return [
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
            "audioFilePath": audioPath,
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

    func tracePayload(_ trace: GenerationTrace) throws -> [String: Any] {
        var payload = try encodedJSONObject(trace)
        payload["adapters"] = trace.adapters.map(historyAdapterPayload)
        payload["audioFilePath"] = try historyStore
            .fileURL(forAudioFileName: trace.audioFileName)
            .path(percentEncoded: false)
        return payload
    }

    func traceSummary(_ trace: GenerationTrace) -> [String: Any] {
        let audioPath: Any = (try? historyStore.fileURL(forAudioFileName: trace.audioFileName))?
            .path(percentEncoded: false) ?? NSNull()
        return [
            "historyEntryID": trace.historyEntryID.uuidString,
            "createdAt": dateFormatter.string(from: trace.createdAt),
            "pipeline": trace.pipeline,
            "model": trace.primaryModel?.displayName ?? trace.primaryModel?.id ?? "None",
            "audioFileName": trace.audioFileName,
            "audioFilePath": audioPath,
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

    func reviewPayload(_ trace: GenerationTrace) -> [String: Any] {
        let audioPath: Any = (try? historyStore.fileURL(forAudioFileName: trace.audioFileName))?
            .path(percentEncoded: false) ?? NSNull()
        return [
            "historyEntryID": trace.historyEntryID.uuidString,
            "traceFileName": trace.traceFileName,
            "audioFileName": trace.audioFileName,
            "audioFilePath": audioPath,
            "isPreferred": trace.isPreferred ?? false,
            "listeningNote": trace.listeningNote ?? NSNull()
        ]
    }


    func rowPayload(_ row: TraceComparisonRow) -> [String: Any] {
        [
            "label": row.label,
            "left": row.left,
            "right": row.right,
            "isDifferent": row.isDifferent
        ]
    }

    func encodedJSONObject<T: Encodable>(_ value: T) throws -> [String: Any] {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        let data = try encoder.encode(value)
        guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw PiMPAudioCommandFacadeError.encodingFailed
        }
        return object
    }


}
