import Foundation
import PiMPAudioCore

// MARK: - Tier B / C MCP surface expansions

extension PiMPAudioCommandFacade {
    private var mcpGenericProgressPlan: [String: (base: Double, span: Double)] {
        [
            "queued": (0.00, 0.05),
            "building": (0.05, 0.90),
            "enriching": (0.05, 0.90),
            "downloading": (0.05, 0.90),
            "generating": (0.05, 0.90),
            "complete": (1.00, 0.00),
            "error": (1.00, 0.00),
            "cancelled": (1.00, 0.00)
        ]
    }

    private func mcpProgress(
        stage: String,
        message: String,
        stageProgress: Double? = nil,
        stageTotal: Double? = nil
    ) -> [String: Any] {
        operationProgressPayload(
            stage: stage,
            message: message,
            plan: mcpGenericProgressPlan,
            stageProgress: stageProgress,
            stageTotal: stageTotal
        )
    }

    // MARK: Dataset build / enrich / review

    func buildTrainingDataset(arguments: [String: Any]) async throws -> [String: Any] {
        let dryRun = boolValue(arguments["dryRun"] ?? arguments["validateOnly"]) ?? true
        let paths = stringArrayValue(arguments["paths"] ?? arguments["files"] ?? arguments["sourceFiles"]) ?? []
        let directories = stringArrayValue(arguments["directories"] ?? arguments["sourceDirectories"]) ?? []
        guard !paths.isEmpty || !directories.isEmpty else {
            throw PiMPAudioCommandFacadeError.missingRequiredArgument("paths")
        }

        let modeRaw = stringValue(arguments["buildMode"] ?? arguments["mode"])?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased() ?? LocalDatasetBuildMode.wholeTracks.rawValue
        let buildMode: LocalDatasetBuildMode
        switch modeRaw {
        case "whole_tracks", "wholetracks", "whole-track", "whole":
            buildMode = .wholeTracks
        case "clip_extraction", "clips", "extract", "extract_clips":
            buildMode = .clipExtraction
        default:
            throw PiMPAudioCommandFacadeError.invalidTrainingConfiguration(
                "Unsupported buildMode '\(modeRaw)'. Use whole_tracks or clip_extraction."
            )
        }

        var config = LocalFilesDatasetConfig(buildMode: buildMode)
        if let clipLength = doubleValue(arguments["clipLength"] ?? arguments["clipLengthSeconds"]) {
            config.clipLength = clipLength
        }
        if let clipsPerTrack = integerValue(arguments["clipsPerTrack"]) {
            config.clipsPerTrack = clipsPerTrack
        }
        if let maxTotal = integerValue(arguments["maxTotalClips"] ?? arguments["maxClips"]) {
            config.maxTotalClips = maxTotal
        }
        if let sampling = stringValue(arguments["samplingMode"])?.lowercased() {
            switch sampling {
            case "sequential": config.samplingMode = .sequential
            case "random", "random_seeded", "randomseeded": config.samplingMode = .randomSeeded
            default: config.samplingMode = .evenlySpread
            }
        }
        if let seed = integerValue(arguments["samplingSeed"] ?? arguments["seed"]) {
            config.samplingSeed = seed
        }

        let datasetName = stringValue(arguments["datasetName"] ?? arguments["name"])?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let styleTemplate = stringValue(arguments["styleTemplate"] ?? arguments["style"]) ?? ""
        let outputBase: URL
        if let raw = stringValue(arguments["outputDirectory"] ?? arguments["outputBase"])?
            .trimmingCharacters(in: .whitespacesAndNewlines), !raw.isEmpty {
            outputBase = URL(fileURLWithPath: NSString(string: raw).expandingTildeInPath, isDirectory: true)
        } else {
            outputBase = historyStore.storageDirectory.deletingLastPathComponent()
        }

        let sourceFiles = paths.map { URL(fileURLWithPath: NSString(string: $0).expandingTildeInPath) }
        let sourceDirectories = directories.map {
            URL(fileURLWithPath: NSString(string: $0).expandingTildeInPath, isDirectory: true)
        }

        var payload: [String: Any] = [
            "dryRun": dryRun,
            "buildMode": buildMode.rawValue,
            "sourceFileCount": sourceFiles.count,
            "sourceDirectoryCount": sourceDirectories.count,
            "datasetName": datasetName.isEmpty ? NSNull() : datasetName,
            "outputBase": outputBase.path(percentEncoded: false),
            "clipLength": config.clipLength,
            "clipsPerTrack": config.clipsPerTrack,
            "maxTotalClips": config.maxTotalClips,
            "samplingMode": config.samplingMode.rawValue,
            "submitted": false
        ]

        guard !dryRun else {
            payload["confirmation"] = confirmationRequirementPayload(toolName: "build_training_dataset")
            payload["note"] = "Dry run only. Call with dryRun: false and confirm: true to build the package."
            return payload
        }
        guard boolValue(arguments["confirm"]) == true else {
            throw PiMPAudioCommandFacadeError.confirmationRequired("build_training_dataset")
        }

        let operation = registerOperation(
            kind: "dataset.build",
            title: "Build training dataset",
            request: payload,
            progress: mcpProgress(stage: "queued", message: "Queued dataset build")
        )
        let operationID = operation.id
        operationTasks[operationID] = Task { @MainActor in
            do {
                updateOperationProgress(
                    id: operationID,
                    progress: mcpProgress(stage: "building", message: "Building dataset package")
                )
                let result = try await LocalFilesDatasetBuilder.run(
                    sourceFiles: sourceFiles,
                    sourceDirectories: sourceDirectories,
                    config: config,
                    outputBase: outputBase,
                    datasetName: datasetName,
                    styleTemplate: styleTemplate,
                    onPhase: { _ in }
                )
                completeOperation(
                    id: operationID,
                    progress: mcpProgress(stage: "complete", message: "Dataset package ready"),
                    result: [
                        "outputDirectory": result.outputDirectory.path(percentEncoded: false),
                        "clipCount": result.clipCount
                    ]
                )
            } catch is CancellationError {
                cancelOperationRecord(
                    id: operationID,
                    reason: "Cancelled",
                    progress: mcpProgress(stage: "cancelled", message: "Dataset build cancelled")
                )
            } catch {
                failOperation(
                    id: operationID,
                    error: error.localizedDescription,
                    progress: mcpProgress(stage: "error", message: error.localizedDescription)
                )
            }
        }

        payload["submitted"] = true
        payload["operationID"] = operationID
        payload["note"] = "Dataset build submitted. Poll get_operation with this operationID."
        return payload
    }

    func enrichTrainingDataset(arguments: [String: Any]) async throws -> [String: Any] {
        let dryRun = boolValue(arguments["dryRun"] ?? arguments["validateOnly"]) ?? true
        let packageDirectory = try resolveTrainingDatasetDirectory(arguments: arguments)
        let backendRaw = stringValue(arguments["backend"] ?? arguments["annotationBackend"])?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased() ?? "automatic"
        let createStems = boolValue(arguments["createStems"] ?? arguments["stems"]) ?? false
        let force = boolValue(arguments["force"]) ?? false

        var options = LocalDatasetEnrichmentOptions()
        switch backendRaw {
        case "automatic", "auto":
            options.annotationBackend = .automatic
        case "lightweight", "native", "light":
            options.annotationBackend = .lightweight
        case "ace", "native_ace", "native-ace", "advanced":
            options.annotationBackend = .nativeACE
        default:
            throw PiMPAudioCommandFacadeError.invalidTrainingConfiguration(
                "Unsupported backend '\(backendRaw)'. Use automatic, lightweight, or native_ace."
            )
        }
        options.createStems = createStems

        var payload: [String: Any] = [
            "dryRun": dryRun,
            "packagePath": packageDirectory.path(percentEncoded: false),
            "backend": backendRaw,
            "createStems": createStems,
            "force": force,
            "submitted": false
        ]

        guard !dryRun else {
            payload["confirmation"] = confirmationRequirementPayload(toolName: "enrich_training_dataset")
            payload["note"] = "Dry run only. Call with dryRun: false and confirm: true to enrich."
            return payload
        }
        guard boolValue(arguments["confirm"]) == true else {
            throw PiMPAudioCommandFacadeError.confirmationRequired("enrich_training_dataset")
        }

        let operation = registerOperation(
            kind: "dataset.enrich",
            title: "Enrich training dataset",
            request: payload,
            progress: mcpProgress(stage: "queued", message: "Queued dataset enrichment")
        )
        let operationID = operation.id
        operationTasks[operationID] = Task { @MainActor in
            do {
                updateOperationProgress(
                    id: operationID,
                    progress: mcpProgress(stage: "enriching", message: "Running dataset enrichment")
                )
                _ = try await LocalDatasetEnrichmentService.run(
                    datasetDirectory: packageDirectory,
                    force: force,
                    options: options,
                    onStep: { _ in }
                )
                completeOperation(
                    id: operationID,
                    progress: mcpProgress(stage: "complete", message: "Enrichment complete"),
                    result: ["packagePath": packageDirectory.path(percentEncoded: false)]
                )
            } catch is CancellationError {
                cancelOperationRecord(
                    id: operationID,
                    reason: "Cancelled",
                    progress: mcpProgress(stage: "cancelled", message: "Enrichment cancelled")
                )
            } catch {
                failOperation(
                    id: operationID,
                    error: error.localizedDescription,
                    progress: mcpProgress(stage: "error", message: error.localizedDescription)
                )
            }
        }

        payload["submitted"] = true
        payload["operationID"] = operationID
        return payload
    }

    func updateTrainingDatasetReview(arguments: [String: Any]) throws -> [String: Any] {
        guard boolValue(arguments["confirm"]) == true else {
            throw PiMPAudioCommandFacadeError.confirmationRequired("update_training_dataset_review")
        }
        let packageDirectory = try resolveTrainingDatasetDirectory(arguments: arguments)
        var document = try TrainingDatasetPackageEditor.loadPackage(at: packageDirectory)

        var updated = 0
        if let clipID = stringValue(arguments["clipID"] ?? arguments["assetID"] ?? arguments["id"])?
            .trimmingCharacters(in: .whitespacesAndNewlines),
           !clipID.isEmpty,
           let index = document.clips.firstIndex(where: {
               $0.id == clipID || $0.assetID == clipID || $0.file == clipID
           }) {
            if let caption = stringValue(arguments["caption"] ?? arguments["prompt"]) {
                document.clips[index].caption = caption
                document.clips[index].captionSource = "manual"
                updated += 1
            }
            if let lyrics = stringValue(arguments["lyrics"]) {
                document.clips[index].lyrics = lyrics
                document.clips[index].lyricsSource = "manual"
                updated += 1
            }
            if let include = boolValue(arguments["includeInTraining"] ?? arguments["include"]) {
                document.clips[index].includeInTraining = include
                updated += 1
            }
            if let approved = boolValue(arguments["approved"] ?? arguments["isApproved"]) {
                document.clips[index].isApproved = approved
                if approved { document.clips[index].reviewStatus = "approved" }
                updated += 1
            }
            if let notes = stringValue(arguments["notes"]) {
                document.clips[index].notes = notes
                updated += 1
            }
        } else if boolValue(arguments["approveAllIncluded"]) == true {
            for index in document.clips.indices where document.clips[index].includeInTraining {
                document.clips[index].isApproved = true
                document.clips[index].reviewStatus = "approved"
                updated += 1
            }
        } else {
            throw PiMPAudioCommandFacadeError.missingRequiredArgument("clipID")
        }

        guard updated > 0 else {
            throw PiMPAudioCommandFacadeError.invalidTrainingConfiguration(
                "No review fields were updated. Provide caption, lyrics, includeInTraining, approved, notes, or approveAllIncluded."
            )
        }

        document.hasUnsavedChanges = true
        try TrainingDatasetPackageEditor.save(document)
        return [
            "submitted": true,
            "updatedFieldCount": updated,
            "packagePath": packageDirectory.path(percentEncoded: false),
            "blockerCount": document.blockerCount,
            "isTrainingReady": document.isTrainingReady,
            "includedCount": document.includedCount,
            "approvedCount": document.approvedCount
        ]
    }

    // MARK: Adapters

    func importAdapter(arguments: [String: Any]) async throws -> [String: Any] {
        guard boolValue(arguments["confirm"]) == true else {
            throw PiMPAudioCommandFacadeError.confirmationRequired("import_adapter")
        }
        let path = try requiredString("path", aliases: ["directory", "folder", "adapterPath"], in: arguments)
        let directory = URL(fileURLWithPath: NSString(string: path).expandingTildeInPath, isDirectory: true)
        let preferredName = stringValue(arguments["name"] ?? arguments["adapterName"])
        let importer = CommunityService()
        let adapter = try await importer.importLocalAdapter(directoryURL: directory, preferredName: preferredName)
        return [
            "submitted": true,
            "adapter": [
                "id": adapter.id,
                "name": adapter.name,
                "kind": adapter.kind.rawValue,
                "baseModelID": adapter.baseModelID,
                "rank": adapter.config.rank
            ]
        ]
    }

    func deleteAdapter(arguments: [String: Any]) throws -> [String: Any] {
        guard boolValue(arguments["confirm"]) == true else {
            throw PiMPAudioCommandFacadeError.confirmationRequired("delete_adapter")
        }
        let identifier = try requiredString("id", aliases: ["adapterID", "adapterId", "name"], in: arguments)
        let adapters = AdapterRegistry.listAdapters()
        guard let adapter = adapters.first(where: {
            $0.id == identifier
                || $0.name.caseInsensitiveCompare(identifier) == .orderedSame
                || $0.name.localizedCaseInsensitiveContains(identifier)
        }) else {
            throw PiMPAudioCommandFacadeError.adapterNotFound(identifier)
        }
        try AdapterRegistry.delete(adapterID: adapter.id)
        return [
            "submitted": true,
            "deletedAdapterID": adapter.id,
            "deletedAdapterName": adapter.name
        ]
    }

    func mergeAdapters(arguments: [String: Any]) throws -> [String: Any] {
        guard boolValue(arguments["confirm"]) == true else {
            throw PiMPAudioCommandFacadeError.confirmationRequired("merge_adapters")
        }
        let idA = try requiredString("adapterA", aliases: ["a", "left"], in: arguments)
        let idB = try requiredString("adapterB", aliases: ["b", "right"], in: arguments)
        let alpha = Float(doubleValue(arguments["alpha"] ?? arguments["weight"]) ?? 0.5)
        let name = stringValue(arguments["name"] ?? arguments["adapterName"])?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let adapters = AdapterRegistry.listAdapters()
        func resolve(_ identifier: String) throws -> LoRAAdapter {
            if let match = adapters.first(where: {
                $0.id == identifier || $0.name.caseInsensitiveCompare(identifier) == .orderedSame
            }) {
                return match
            }
            throw PiMPAudioCommandFacadeError.adapterNotFound(identifier)
        }
        let a = try resolve(idA)
        let b = try resolve(idB)
        let mergedName = (name?.isEmpty == false) ? name! : "\(a.name)+\(b.name)"
        let merged = try AdapterRegistry.merge(adapterA: a.id, adapterB: b.id, alpha: alpha, name: mergedName)
        return [
            "submitted": true,
            "adapter": [
                "id": merged.id,
                "name": merged.name,
                "kind": merged.kind.rawValue,
                "baseModelID": merged.baseModelID,
                "rank": merged.config.rank
            ],
            "alpha": alpha
        ]
    }

    // MARK: Models

    func downloadModel(arguments: [String: Any]) async throws -> [String: Any] {
        let dryRun = boolValue(arguments["dryRun"] ?? arguments["validateOnly"]) ?? true
        let modelID = try requiredString("id", aliases: ["modelID", "modelId", "model"], in: arguments)
        let specs = await registry.allSpecs
        guard let spec = specs.first(where: {
            $0.id == modelID || $0.displayName.caseInsensitiveCompare(modelID) == .orderedSame
        }) else {
            throw PiMPAudioCommandFacadeError.resourceNotFound("model:\(modelID)")
        }
        let alreadyReady = await registry.status(for: spec.id) == .ready

        var payload: [String: Any] = [
            "dryRun": dryRun,
            "modelID": spec.id,
            "displayName": spec.displayName,
            "category": spec.category.rawValue,
            "requiresAuth": spec.requiresAuth,
            "alreadyReady": alreadyReady,
            "submitted": false
        ]

        guard !dryRun else {
            payload["confirmation"] = confirmationRequirementPayload(toolName: "download_model")
            return payload
        }
        guard boolValue(arguments["confirm"]) == true else {
            throw PiMPAudioCommandFacadeError.confirmationRequired("download_model")
        }

        let operation = registerOperation(
            kind: "model.download",
            title: "Download \(spec.displayName)",
            request: payload,
            progress: mcpProgress(stage: "queued", message: "Queued model download")
        )
        let operationID = operation.id
        operationTasks[operationID] = Task { @MainActor in
            do {
                try await registry.downloadModels([spec]) { _, progress in
                    Task { @MainActor in
                        self.updateOperationProgress(
                            id: operationID,
                            progress: self.mcpProgress(
                                stage: "downloading",
                                message: "Downloading \(spec.displayName)",
                                stageProgress: progress,
                                stageTotal: 1
                            )
                        )
                    }
                }
                completeOperation(
                    id: operationID,
                    progress: mcpProgress(stage: "complete", message: "Download complete"),
                    result: ["modelID": spec.id, "status": "ready"]
                )
            } catch is CancellationError {
                cancelOperationRecord(
                    id: operationID,
                    reason: "Cancelled",
                    progress: mcpProgress(stage: "cancelled", message: "Download cancelled")
                )
            } catch {
                failOperation(
                    id: operationID,
                    error: error.localizedDescription,
                    progress: mcpProgress(stage: "error", message: error.localizedDescription)
                )
            }
        }

        payload["submitted"] = true
        payload["operationID"] = operationID
        return payload
    }

    func setActiveModel(arguments: [String: Any]) async throws -> [String: Any] {
        guard boolValue(arguments["confirm"]) == true else {
            throw PiMPAudioCommandFacadeError.confirmationRequired("set_active_model")
        }
        let modelID = try requiredString("id", aliases: ["modelID", "modelId", "model"], in: arguments)
        let specs = await registry.allSpecs
        guard let spec = specs.first(where: {
            $0.id == modelID || $0.displayName.caseInsensitiveCompare(modelID) == .orderedSame
        }) else {
            throw PiMPAudioCommandFacadeError.resourceNotFound("model:\(modelID)")
        }
        try await registry.setActiveModel(id: spec.id, for: spec.category)
        return [
            "submitted": true,
            "modelID": spec.id,
            "displayName": spec.displayName,
            "category": spec.category.rawValue
        ]
    }

    // MARK: History / presets / diagnostics / generation cancel / audition

    func deleteHistoryEntries(arguments: [String: Any]) throws -> [String: Any] {
        guard boolValue(arguments["confirm"]) == true else {
            throw PiMPAudioCommandFacadeError.confirmationRequired("delete_history_entries")
        }
        if boolValue(arguments["deleteAll"]) == true {
            let count = historyStore.entries.count
            historyStore.deleteAll()
            return ["submitted": true, "deletedCount": count, "deleteAll": true]
        }

        let ids = stringArrayValue(arguments["ids"] ?? arguments["historyIDs"] ?? arguments["entryIDs"]) ?? []
        guard !ids.isEmpty else {
            throw PiMPAudioCommandFacadeError.missingRequiredArgument("ids")
        }
        var matched = Set<HistoryEntry.ID>()
        for raw in ids {
            if let uuid = UUID(uuidString: raw),
               historyStore.entries.contains(where: { $0.id == uuid }) {
                matched.insert(uuid)
                continue
            }
            if let entry = historyStore.entries.first(where: {
                $0.fileName == raw || $0.id.uuidString == raw
            }) {
                matched.insert(entry.id)
            }
        }
        guard !matched.isEmpty else {
            throw PiMPAudioCommandFacadeError.historyEntryNotFound(ids.joined(separator: ","))
        }
        historyStore.delete(ids: matched)
        return [
            "submitted": true,
            "deletedCount": matched.count,
            "deletedIDs": matched.map(\.uuidString)
        ]
    }

    func saveGenerationPreset(arguments: [String: Any]) throws -> [String: Any] {
        guard boolValue(arguments["confirm"]) == true else {
            throw PiMPAudioCommandFacadeError.confirmationRequired("save_generation_preset")
        }
        let name = try requiredString("name", aliases: ["presetName", "title"], in: arguments)
        let prompt = stringValue(arguments["prompt"] ?? arguments["text"]) ?? ""
        let lyrics = stringValue(arguments["lyrics"]) ?? ""
        let duration = doubleValue(arguments["duration"]) ?? 30
        let seedText = stringValue(arguments["seed"] ?? arguments["seedText"]) ?? ""
        let guidanceScale = doubleValue(arguments["guidanceScale"] ?? arguments["guidance"]) ?? 7.5
        let guidancePresetRaw = stringValue(arguments["guidancePreset"]) ?? GuidancePreset.defaultSelection.rawValue
        let guidancePreset = GuidancePreset(rawValue: guidancePresetRaw) ?? .softLate
        let mode = generationMode(from: arguments["mode"])
        let pipeline = pipelineChoice(from: arguments["pipeline"] ?? arguments["pipelineOverride"]) ?? .auto

        let preset = GenerationPreset(
            name: name,
            prompt: prompt,
            lyrics: lyrics,
            mode: mode,
            duration: duration,
            seedText: seedText,
            guidanceScale: guidanceScale,
            guidancePreset: guidancePreset,
            pipelineOverride: pipeline,
            method: .ode,
            precision: .float16,
            stepsOverride: stringValue(arguments["steps"]) ?? "",
            shiftOverride: doubleValue(arguments["shift"]) ?? 3.0,
            adapterID: stringValue(arguments["adapterID"] ?? arguments["adapter"]),
            adapterStrength: doubleValue(arguments["adapterStrength"] ?? arguments["adapterGain"]) ?? 1.0,
            coverStrength: doubleValue(arguments["coverStrength"]) ?? 0.7
        )
        let store = PresetStore()
        store.save(preset)
        return [
            "submitted": true,
            "preset": [
                "id": preset.id.uuidString,
                "name": preset.name,
                "duration": preset.duration
            ]
        ]
    }

    func deleteGenerationPreset(arguments: [String: Any]) throws -> [String: Any] {
        guard boolValue(arguments["confirm"]) == true else {
            throw PiMPAudioCommandFacadeError.confirmationRequired("delete_generation_preset")
        }
        let identifier = try requiredString("id", aliases: ["presetID", "name"], in: arguments)
        let store = PresetStore()
        guard let preset = store.presets.first(where: {
            $0.id.uuidString.caseInsensitiveCompare(identifier) == .orderedSame
                || $0.name.caseInsensitiveCompare(identifier) == .orderedSame
        }) else {
            throw PiMPAudioCommandFacadeError.resourceNotFound("preset:\(identifier)")
        }
        store.delete(preset)
        return [
            "submitted": true,
            "deletedPresetID": preset.id.uuidString,
            "deletedPresetName": preset.name
        ]
    }

    func setVerboseDiagnostics(arguments: [String: Any]) throws -> [String: Any] {
        let enabled = boolValue(arguments["enabled"] ?? arguments["verbose"] ?? arguments["value"])
        guard let enabled else {
            throw PiMPAudioCommandFacadeError.invalidBoolean("enabled")
        }
        UserDefaults.standard.set(enabled, forKey: AppUserDefaultsKeys.audioDebugVerbose)
        AudioDiagnostics.refreshFileLoggingState()
        return [
            "submitted": true,
            "verboseAudioDiagnostics": enabled,
            "diagnosticsDirectory": AppStorageLocations.diagnosticsDirectory.path(percentEncoded: false)
        ]
    }

    func tailDiagnosticsLog(arguments: [String: Any]) throws -> [String: Any] {
        var patched = arguments
        if patched["contains"] == nil, patched["filter"] == nil {
            patched["contains"] = ""
        }
        return try tailTrainingLog(arguments: patched)
    }

    func cancelGeneration(arguments: [String: Any]) throws -> [String: Any] {
        guard let generationService else {
            throw PiMPAudioCommandFacadeError.generationServiceUnavailable
        }
        generationService.cancel()
        if let operationID = activeGenerationOperationID {
            _ = try? cancelOperation(arguments: [
                "id": operationID,
                "reason": stringValue(arguments["reason"]) ?? "Cancelled via cancel_generation"
            ])
        }
        return [
            "submitted": true,
            "cancelled": true,
            "generation": try generationStatus()
        ]
    }

    func runAdapterAudition(arguments: [String: Any]) async throws -> [String: Any] {
        guard boolValue(arguments["confirm"]) == true else {
            throw PiMPAudioCommandFacadeError.confirmationRequired("run_adapter_audition")
        }
        guard let generationService else {
            throw PiMPAudioCommandFacadeError.generationServiceUnavailable
        }

        let identifier = try requiredString(
            "id",
            aliases: ["adapterID", "adapterId", "adapter", "name"],
            in: arguments
        )
        let adapters = AdapterRegistry.listAdapters()
        guard let adapter = adapters.first(where: {
            $0.id == identifier
                || $0.name.caseInsensitiveCompare(identifier) == .orderedSame
                || $0.name.localizedCaseInsensitiveContains(identifier)
        }) else {
            throw PiMPAudioCommandFacadeError.adapterNotFound(identifier)
        }

        let seed = integerValue(arguments["seed"]) ?? 42
        let duration = doubleValue(arguments["duration"]) ?? 30
        let includeBase = boolValue(arguments["includeBase"]) ?? true
        let strengths = (stringArrayValue(arguments["strengths"] ?? arguments["gains"]) ?? ["1.0", "1.6", "2.0"])
            .compactMap { Float($0) }
        let variants = AdapterAuditionVariant.variants(
            for: adapter,
            strengths: strengths,
            includeBase: includeBase
        )
        guard !variants.isEmpty else {
            throw PiMPAudioCommandFacadeError.invalidTrainingConfiguration(
                "Add at least one adapter gain or include Base."
            )
        }

        let prompt = stringValue(arguments["prompt"] ?? arguments["text"])?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let lyrics = stringValue(arguments["lyrics"]) ?? ""
        let resolvedPrompt = (prompt?.isEmpty == false) ? prompt! : "instrumental showcase for \(adapter.name)"

        let operation = registerOperation(
            kind: "adapter.audition",
            title: "Adapter audition \(adapter.name)",
            request: [
                "adapterID": adapter.id,
                "seed": seed,
                "duration": duration,
                "variantCount": variants.count
            ],
            progress: mcpProgress(stage: "queued", message: "Queued adapter audition")
        )
        let operationID = operation.id
        operationTasks[operationID] = Task { @MainActor in
            var completed: [[String: Any]] = []
            do {
                for (offset, variant) in variants.enumerated() {
                    try Task.checkCancellation()
                    updateOperationProgress(
                        id: operationID,
                        progress: mcpProgress(
                            stage: "generating",
                            message: "Rendering \(variant.label)",
                            stageProgress: Double(offset),
                            stageTotal: Double(variants.count)
                        )
                    )
                    let request = GenerationRequest(
                        text: resolvedPrompt,
                        lyrics: lyrics,
                        duration: duration,
                        seed: seed,
                        method: .ode,
                        guidanceScale: 7.5,
                        guidancePreset: .constant,
                        precision: .float32,
                        stepsOverride: nil,
                        shiftOverride: 3.0,
                        resolvedMode: .music,
                        pipelineOverride: .aceStep,
                        coverStrength: 0.7
                    )
                    let result = try await generationService.generateAndWait(
                        request: request,
                        adapterStack: variant.adapterStack,
                        selectedAdapterID: variant.adapterID
                    )
                    completed.append([
                        "label": variant.label,
                        "audioPath": result.fileURL.path(percentEncoded: false),
                        "historyEntryID": result.entry.id.uuidString,
                        "traceFileName": result.entry.traceFileName ?? NSNull()
                    ])
                }
                completeOperation(
                    id: operationID,
                    progress: mcpProgress(stage: "complete", message: "Audition complete"),
                    result: [
                        "adapterID": adapter.id,
                        "adapterName": adapter.name,
                        "takes": completed
                    ]
                )
            } catch is CancellationError {
                cancelOperationRecord(
                    id: operationID,
                    reason: "Cancelled",
                    progress: mcpProgress(stage: "cancelled", message: "Audition cancelled")
                )
            } catch {
                failOperation(
                    id: operationID,
                    error: error.localizedDescription,
                    progress: mcpProgress(stage: "error", message: error.localizedDescription)
                )
            }
        }

        return [
            "submitted": true,
            "operationID": operationID,
            "adapterID": adapter.id,
            "adapterName": adapter.name,
            "variantCount": variants.count,
            "note": "Poll get_operation for audition progress and take paths."
        ]
    }
}
