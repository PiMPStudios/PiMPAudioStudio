import Foundation
import AVFoundation
import PiMPAudioCore

extension PiMPAudioCommandFacade {
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

        let adapterPlan = orchestrationAdapterPlan(from: arguments)
        let adapterPayload: [String: Any]?
        if adapterPlan.hasAction {
            if dryRun {
                adapterPayload = [
                    "skipped": true,
                    "reason": "dryRun",
                    "wouldClear": adapterPlan.shouldClear,
                    "wouldApply": adapterPlan.applyArguments != nil
                ]
            } else {
                var clearPayload: [String: Any]?
                if adapterPlan.shouldClear {
                    clearPayload = try await applyAdapter(arguments: ["clear": true])
                }

                if let applyArguments = adapterPlan.applyArguments {
                    var appliedPayload = try await applyAdapter(arguments: applyArguments)
                    if adapterPlan.shouldClear {
                        appliedPayload["clearedBeforeApply"] = true
                    }
                    adapterPayload = appliedPayload
                } else {
                    adapterPayload = clearPayload
                }
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


    func loadAdapters() async -> [LoRAAdapter] {
        await Task.detached(priority: .userInitiated) {
            AdapterRegistry.listAdapters()
        }.value
    }

    func parseAdapterSelection(
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

    func resolveAdapter(_ identifier: String, installed: [LoRAAdapter]) throws -> LoRAAdapter {
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

    func adapterStrength(_ value: Any?) throws -> Float {
        guard let value else { return 1.0 }
        guard let parsed = doubleValue(value),
              parsed >= Double(AdapterStrengthLimits.minimum),
              parsed <= Double(AdapterStrengthLimits.maximum) else {
            throw PiMPAudioCommandFacadeError.invalidStrength
        }
        return Float(parsed)
    }

    func syncGenerationAdapterSelectionFromDefaultsIfNeeded(_ generationService: GenerationService) {
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
        let strength = AdapterStrengthLimits.clamped(Float(rawStrength))
        generationService.selectedAdapterID = adapterID
        generationService.adapterStack = [(id: adapterID, weight: strength)]
    }

    func orchestrationAdapterPlan(from arguments: [String: Any]) -> OrchestrationAdapterPlan {
        OrchestrationAdapterPlan(
            shouldClear: boolValue(arguments["clearAdapter"] ?? arguments["clearAdapters"]) == true,
            applyArguments: orchestrationAdapterArguments(from: arguments)
        )
    }

    func orchestrationAdapterArguments(from arguments: [String: Any]) -> [String: Any]? {
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

    func orchestrationDuration(in arguments: [String: Any]) throws -> Double {
        let duration = doubleValue(arguments["duration"] ?? arguments["durationSeconds"]) ?? 30.0
        guard duration > 0 else {
            throw PiMPAudioCommandFacadeError.invalidDuration
        }
        return duration
    }

    func generationRequest(_ arguments: [String: Any]) throws -> GenerationRequest {
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

    func referenceAudioURL(from value: Any?) -> URL? {
        guard let path = stringValue(value)?.trimmingCharacters(in: .whitespacesAndNewlines),
              !path.isEmpty else {
            return nil
        }
        if let url = URL(string: path), url.scheme != nil {
            return url
        }
        return URL(fileURLWithPath: NSString(string: path).expandingTildeInPath)
    }

    func seedValue(_ value: Any?) throws -> Int? {
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

    func stepsValue(_ value: Any?) throws -> Int? {
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


    func adapterSelectionPayload(_ selection: [AdapterSelection]) -> [String: Any] {
        [
            "applied": true,
            "cleared": selection.isEmpty,
            "count": selection.count,
            "primary": selection.first.map(adapterSelectionEntryPayload) ?? NSNull(),
            "adapters": selection.map(adapterSelectionEntryPayload)
        ]
    }

    func adapterSelectionEntryPayload(_ entry: AdapterSelection) -> [String: Any] {
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

    func generationSubmissionPayload(
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

    func generationRequestPayload(_ request: GenerationRequest) -> [String: Any] {
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

    func generationResultPayload(_ result: GenerationResult) -> [String: Any] {
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

    func generationStatePayload(_ state: GenerationState) -> [String: Any] {
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

    func stateName(_ state: GenerationState) -> String {
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

    func persistAdapterSelection(_ selection: [AdapterSelection]) {
        let defaults = UserDefaults.standard
        if let primary = selection.first {
            defaults.set(primary.adapter.id, forKey: AppUserDefaultsKeys.selectedAdapterID)
            defaults.set(Double(primary.strength), forKey: AppUserDefaultsKeys.selectedAdapterStrength)
        } else {
            defaults.removeObject(forKey: AppUserDefaultsKeys.selectedAdapterID)
            defaults.set(1.0, forKey: AppUserDefaultsKeys.selectedAdapterStrength)
        }
    }

    func notifyAdapterSelectionChanged(_ selection: [AdapterSelection]) {
        NotificationCenter.default.post(
            name: .mcpAdapterSelectionDidChange,
            object: nil,
            userInfo: [
                "adapters": selection.map(adapterSelectionEntryPayload)
            ]
        )
    }


}
