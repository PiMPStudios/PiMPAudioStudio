import Foundation
import AVFoundation
import PiMPAudioCore

extension PiMPAudioCommandFacade {
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
        try reserveEditorOperation(id: operation.id)

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
        try reserveEditorOperation(id: operation.id)

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
        try reserveEditorOperation(id: operation.id)

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
        try reserveEditorOperation(id: operation.id)

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


    func editorLoadProgressPayload(stage: String, message: String) -> [String: Any] {
        operationProgressPayload(
            stage: stage,
            message: message,
            plan: editorLoadProgressPlan
        )
    }

    func editorInpaintProgressPayload(_ progress: InpaintProgress) -> [String: Any] {
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

    func editorInpaintProgressPayload(
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

    func editorRenderProgressPayload(stage: String, message: String) -> [String: Any] {
        operationProgressPayload(
            stage: stage,
            message: message,
            plan: editorRenderProgressPlan
        )
    }

    func editorDSPProgressPayload(
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


    func editorAudioLoadRequestPayload(
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

    func editorLoadedAudioPayload(_ audio: EditorLoadedAudio) -> [String: Any] {
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

    func editorEditSummaryPayload(_ edit: EditorEditSummary) -> [String: Any] {
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

    func editorInpaintRequestPayload(_ request: InpaintRequest) -> [String: Any] {
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

    func editorRenderRequestPayload(_ request: EditorRenderRequest) -> [String: Any] {
        [
            "format": request.format.rawValue,
            "outputPath": request.outputURL.path(percentEncoded: false),
            "overwrite": request.overwrite,
            "selection": request.selection.map { editorSelectionPayload($0) as Any } ?? NSNull()
        ]
    }

    func editorRenderResultPayload(
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

    func editorDSPRequestPayload(_ request: EditorDSPRequest) -> [String: Any] {
        [
            "effectCount": request.steps.count,
            "description": request.description,
            "effects": request.steps.map(editorDSPStepPayload)
        ]
    }

    func editorDSPStepPayload(_ step: EditorDSPEffectStep) -> [String: Any] {
        [
            "effect": step.effect.rawValue,
            "displayName": step.effect.displayName,
            "target": editorDSPTargetPayload(step.target),
            "parameters": editorDSPParametersPayload(step.parameters)
        ]
    }

    func editorDSPResultPayload(
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

    func editorAudioBufferPayload(_ buffer: AudioBuffer) -> [String: Any] {
        [
            "duration": buffer.duration,
            "sampleRate": buffer.sampleRate,
            "channels": buffer.channels,
            "frameCount": buffer.frameCount
        ]
    }

    func editorDSPEffectSummary(
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

    func editorDSPEffectSummaryPayload(_ summary: EditorDSPEffectSummary) -> [String: Any] {
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

    func editorDSPTargetPayload(_ target: ApplyTarget) -> [String: Any] {
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

    func editorDSPTargetName(_ target: ApplyTarget) -> String {
        switch target {
        case .wholeTrack:
            return "wholeTrack"
        case .selection:
            return "selection"
        }
    }

    func editorDSPParametersPayload(_ parameters: DSPParameters) -> Any {
        switch parameters {
        case .none:
            return NSNull()
        case .normalize(let params):
            return ["targetPeakDBFS": params.targetPeakDBFS]
        case .trimSilence(let params):
            return [
                "thresholdDBFS": params.thresholdDBFS,
                "maxLeadingSeconds": params.maxLeadingSeconds,
                "maxTrailingSeconds": params.maxTrailingSeconds,
                "edgeFadeMilliseconds": params.edgeFadeMilliseconds
            ]
        case .highPass(let params):
            return ["cutoffHz": params.cutoffHz]
        case .compress(let params):
            return [
                "thresholdDBFS": params.thresholdDBFS,
                "ratio": params.ratio,
                "attackMilliseconds": params.attackMilliseconds,
                "releaseMilliseconds": params.releaseMilliseconds,
                "kneeDB": params.kneeDB,
                "makeupGainDB": params.makeupGainDB
            ]
        case .limit(let params):
            return [
                "ceilingDBFS": params.ceilingDBFS,
                "releaseMilliseconds": params.releaseMilliseconds
            ]
        case .loudness(let params):
            return [
                "targetLUFS": params.targetLUFS,
                "ceilingDBFS": params.ceilingDBFS,
                "maxBoostDB": params.maxBoostDB
            ]
        case .softGate(let params):
            return [
                "thresholdDBFS": params.thresholdDBFS,
                "ratio": params.ratio,
                "attackMilliseconds": params.attackMilliseconds,
                "releaseMilliseconds": params.releaseMilliseconds,
                "rangeDB": params.rangeDB
            ]
        case .polishForKeep(let recipe):
            return [
                "target": recipe.target.rawValue,
                "tone": recipe.tone.rawValue,
                "glue": recipe.glue.rawValue,
                "space": recipe.space.rawValue,
                "trimSilenceAndFades": recipe.trimSilenceAndFades
            ]
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

    func editorDSPParameterSummary(_ parameters: DSPParameters) -> [String: Double] {
        switch parameters {
        case .none:
            return [:]
        case .normalize(let params):
            return ["targetPeakDBFS": Double(params.targetPeakDBFS)]
        case .trimSilence(let params):
            return [
                "thresholdDBFS": Double(params.thresholdDBFS),
                "maxLeadingSeconds": Double(params.maxLeadingSeconds),
                "maxTrailingSeconds": Double(params.maxTrailingSeconds),
                "edgeFadeMilliseconds": Double(params.edgeFadeMilliseconds)
            ]
        case .highPass(let params):
            return ["cutoffHz": Double(params.cutoffHz)]
        case .compress(let params):
            return [
                "thresholdDBFS": Double(params.thresholdDBFS),
                "ratio": Double(params.ratio),
                "attackMilliseconds": Double(params.attackMilliseconds),
                "releaseMilliseconds": Double(params.releaseMilliseconds),
                "kneeDB": Double(params.kneeDB),
                "makeupGainDB": Double(params.makeupGainDB)
            ]
        case .limit(let params):
            return [
                "ceilingDBFS": Double(params.ceilingDBFS),
                "releaseMilliseconds": Double(params.releaseMilliseconds)
            ]
        case .loudness(let params):
            return [
                "targetLUFS": Double(params.targetLUFS),
                "ceilingDBFS": Double(params.ceilingDBFS),
                "maxBoostDB": Double(params.maxBoostDB)
            ]
        case .softGate(let params):
            return [
                "thresholdDBFS": Double(params.thresholdDBFS),
                "ratio": Double(params.ratio),
                "attackMilliseconds": Double(params.attackMilliseconds),
                "releaseMilliseconds": Double(params.releaseMilliseconds),
                "rangeDB": Double(params.rangeDB)
            ]
        case .polishForKeep(let recipe):
            return [
                "trimSilenceAndFades": recipe.trimSilenceAndFades ? 1 : 0,
                "targetStreaming": recipe.target == .streaming ? 1 : 0,
                "targetLouder": recipe.target == .louder ? 1 : 0,
                "targetGentle": recipe.target == .gentle ? 1 : 0
            ]
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

    func editorSelectionPayload(_ selection: InpaintSelection) -> [String: Any] {
        [
            "startTime": selection.startTime,
            "endTime": selection.endTime,
            "frequencyBand": selection.frequencyBand.map { editorFrequencyBandPayload($0) as Any } ?? NSNull()
        ]
    }

    func editorFrequencyBandPayload(_ band: FrequencyBand) -> Any {
        editorFrequencyBandPayload(lowCutHz: band.lowCutHz, highCutHz: band.highCutHz)
    }

    func editorFrequencyBandPayload(lowCutHz: Float?, highCutHz: Float?) -> Any {
        guard let lowCutHz, let highCutHz else { return NSNull() }
        return [
            "lowCutHz": lowCutHz,
            "highCutHz": highCutHz
        ]
    }

    func editorInpaintRequest(
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

    func editorRenderRequest(
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

    func editorDSPRequest(
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

    func editorDSPRawSteps(from arguments: [String: Any]) throws -> [[String: Any]] {
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

    func editorDSPEffectStep(
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

    func editorDSPEffect(from value: String) throws -> DSPEffect {
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
        case "trimsilence", "autosilence", "silencetrim":
            return .trimSilence
        case "highpass", "hpf", "highpassfilter":
            return .highPass
        case "compress", "compressor", "glue":
            return .compress
        case "limit", "limiter", "brickwall":
            return .limit
        case "loudness", "loudnessstage", "lufs", "aimloudness":
            return .loudness
        case "softgate", "gate", "expander", "noisegate":
            return .softGate
        case "polish", "polishforkeep", "keep", "master":
            return .polishForKeep
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

    func editorDSPTarget(
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

    func editorDSPSelection(
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

    func editorDSPParameters(
        for effect: DSPEffect,
        stepArguments: [String: Any],
        topLevel: [String: Any]
    ) throws -> DSPParameters {
        let parameterArguments = stepArguments["parameters"] as? [String: Any]
            ?? stepArguments["params"] as? [String: Any]
            ?? [:]

        switch effect {
        case .normalize:
            return .normalize(try editorNormalizeParameters(parameterArguments, fallback: topLevel))
        case .trimSilence:
            return .trimSilence(try editorTrimSilenceParameters(parameterArguments, fallback: topLevel))
        case .highPass:
            return .highPass(try editorHighPassParameters(parameterArguments, fallback: topLevel))
        case .compress:
            return .compress(try editorCompressorParameters(parameterArguments, fallback: topLevel))
        case .limit:
            return .limit(try editorLimiterParameters(parameterArguments, fallback: topLevel))
        case .loudness:
            return .loudness(try editorLoudnessParameters(parameterArguments, fallback: topLevel))
        case .softGate:
            return .softGate(try editorSoftGateParameters(parameterArguments, fallback: topLevel))
        case .polishForKeep:
            return .polishForKeep(try editorPolishForKeepParameters(parameterArguments, fallback: topLevel))
        case .eq:
            return .eq(try editorEQParameters(parameterArguments, fallback: topLevel))
        case .reverb:
            return .reverb(try editorReverbParameters(parameterArguments, fallback: topLevel))
        case .timeStretch:
            return .timeStretch(try editorTimeStretchParameters(parameterArguments, fallback: topLevel))
        case .pitchShift:
            return .pitchShift(try editorPitchShiftParameters(parameterArguments, fallback: topLevel))
        case .trimToSelection, .deleteSelection, .fadeIn, .fadeOut:
            return .none
        }
    }

    func editorCompressorParameters(
        _ parameters: [String: Any],
        fallback: [String: Any]
    ) throws -> CompressorParameters {
        var result = CompressorParameters.default
        if let preset = stringValue(parameters["preset"] ?? fallback["preset"])?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased() {
            switch preset {
            case "light", "glue": result = .light
            case "medium", "med": result = .medium
            default: break
            }
        }
        if let v = doubleValue(parameters["thresholdDBFS"] ?? parameters["threshold"] ?? fallback["thresholdDBFS"]) {
            result.thresholdDBFS = Float(v)
        }
        if let v = doubleValue(parameters["ratio"] ?? fallback["ratio"]) {
            result.ratio = Float(max(1.001, v))
        }
        if let v = doubleValue(parameters["attackMilliseconds"] ?? parameters["attack"] ?? fallback["attackMilliseconds"]) {
            result.attackMilliseconds = Float(max(0.1, v))
        }
        if let v = doubleValue(parameters["releaseMilliseconds"] ?? parameters["release"] ?? fallback["releaseMilliseconds"]) {
            result.releaseMilliseconds = Float(max(1, v))
        }
        if let v = doubleValue(parameters["kneeDB"] ?? parameters["knee"] ?? fallback["kneeDB"]) {
            result.kneeDB = Float(max(0, v))
        }
        if let v = doubleValue(parameters["makeupGainDB"] ?? parameters["makeup"] ?? fallback["makeupGainDB"]) {
            result.makeupGainDB = Float(v)
        }
        return result
    }

    func editorLimiterParameters(
        _ parameters: [String: Any],
        fallback: [String: Any]
    ) throws -> LimiterParameters {
        var result = LimiterParameters.default
        if let preset = stringValue(parameters["preset"] ?? fallback["preset"])?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased() {
            switch preset {
            case "streaming", "stream": result = .streaming
            case "louder", "loud", "club": result = .louder
            default: break
            }
        }
        if let v = doubleValue(parameters["ceilingDBFS"] ?? parameters["ceiling"] ?? fallback["ceilingDBFS"]) {
            let ceiling = Float(v)
            guard LimiterParameters.ceilingRange.contains(ceiling) else {
                throw PiMPAudioCommandFacadeError.invalidDSPParameters(
                    "Limiter ceilingDBFS must be between \(LimiterParameters.ceilingRange.lowerBound) and \(LimiterParameters.ceilingRange.upperBound)."
                )
            }
            result.ceilingDBFS = ceiling
        }
        if let v = doubleValue(parameters["releaseMilliseconds"] ?? parameters["release"] ?? fallback["releaseMilliseconds"]) {
            result.releaseMilliseconds = Float(max(1, v))
        }
        return result
    }

    func editorSoftGateParameters(
        _ parameters: [String: Any],
        fallback: [String: Any]
    ) throws -> SoftGateParameters {
        var result = SoftGateParameters.default
        if let preset = stringValue(parameters["preset"] ?? fallback["preset"])?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased() {
            switch preset {
            case "gentle", "default", "safe": result = .gentle
            default: break
            }
        }
        if let v = doubleValue(parameters["thresholdDBFS"] ?? parameters["threshold"] ?? fallback["thresholdDBFS"]) {
            let threshold = Float(v)
            guard SoftGateParameters.thresholdRange.contains(threshold) else {
                throw PiMPAudioCommandFacadeError.invalidDSPParameters(
                    "Soft gate thresholdDBFS must be between \(SoftGateParameters.thresholdRange.lowerBound) and \(SoftGateParameters.thresholdRange.upperBound)."
                )
            }
            result.thresholdDBFS = threshold
        }
        if let v = doubleValue(parameters["ratio"] ?? fallback["ratio"]) {
            result.ratio = Float(max(1.001, v))
        }
        if let v = doubleValue(parameters["attackMilliseconds"] ?? parameters["attack"] ?? fallback["attackMilliseconds"]) {
            result.attackMilliseconds = Float(max(0.1, v))
        }
        if let v = doubleValue(parameters["releaseMilliseconds"] ?? parameters["release"] ?? fallback["releaseMilliseconds"]) {
            result.releaseMilliseconds = Float(max(1, v))
        }
        if let v = doubleValue(parameters["rangeDB"] ?? parameters["range"] ?? fallback["rangeDB"]) {
            result.rangeDB = Float(max(0, v))
        }
        return result
    }

    func editorPolishForKeepParameters(
        _ parameters: [String: Any],
        fallback: [String: Any]
    ) throws -> PolishForKeepRecipe {
        var recipe = PolishForKeepRecipe.default

        if let targetRaw = stringValue(parameters["target"] ?? fallback["target"])?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased() {
            switch targetRaw {
            case "streaming", "stream": recipe.target = .streaming
            case "louder", "loud", "club": recipe.target = .louder
            case "gentle", "soft", "headroom": recipe.target = .gentle
            default:
                throw PiMPAudioCommandFacadeError.invalidDSPParameters(
                    "Polish target must be streaming, louder, or gentle."
                )
            }
        }

        if let toneRaw = stringValue(parameters["tone"] ?? fallback["tone"])?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased() {
            switch toneRaw {
            case "neutral": recipe.tone = .neutral
            case "brighter", "bright": recipe.tone = .brighter
            case "darker", "dark": recipe.tone = .darker
            default:
                throw PiMPAudioCommandFacadeError.invalidDSPParameters(
                    "Polish tone must be neutral, brighter, or darker."
                )
            }
        }

        if let glueRaw = stringValue(parameters["glue"] ?? fallback["glue"])?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased() {
            switch glueRaw {
            case "off", "none", "0": recipe.glue = .off
            case "light", "soft": recipe.glue = .light
            case "medium", "med": recipe.glue = .medium
            default:
                throw PiMPAudioCommandFacadeError.invalidDSPParameters(
                    "Polish glue must be off, light, or medium."
                )
            }
        }

        if let spaceRaw = stringValue(parameters["space"] ?? fallback["space"])?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased() {
            switch spaceRaw {
            case "off", "none", "0": recipe.space = .off
            case "room": recipe.space = .room
            case "hall": recipe.space = .hall
            default:
                throw PiMPAudioCommandFacadeError.invalidDSPParameters(
                    "Polish space must be off, room, or hall."
                )
            }
        }

        if let trim = boolValue(
            parameters["trimSilence"]
                ?? parameters["trimSilenceAndFades"]
                ?? parameters["trim"]
                ?? fallback["trimSilence"]
                ?? fallback["trimSilenceAndFades"]
                ?? fallback["trim"]
        ) {
            recipe.trimSilenceAndFades = trim
        }

        return recipe
    }

    func editorLoudnessParameters(
        _ parameters: [String: Any],
        fallback: [String: Any]
    ) throws -> LoudnessParameters {
        var result = LoudnessParameters.default
        if let preset = stringValue(parameters["preset"] ?? fallback["preset"])?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased() {
            switch preset {
            case "streaming", "stream": result = .streaming
            case "louder", "loud", "club": result = .louder
            default: break
            }
        }
        if let v = doubleValue(parameters["targetLUFS"] ?? parameters["target"] ?? fallback["targetLUFS"]) {
            let target = Float(v)
            guard LoudnessParameters.targetRange.contains(target) else {
                throw PiMPAudioCommandFacadeError.invalidDSPParameters(
                    "Loudness targetLUFS must be between \(LoudnessParameters.targetRange.lowerBound) and \(LoudnessParameters.targetRange.upperBound)."
                )
            }
            result.targetLUFS = target
        }
        if let v = doubleValue(parameters["ceilingDBFS"] ?? parameters["ceiling"] ?? fallback["ceilingDBFS"]) {
            let ceiling = Float(v)
            guard LimiterParameters.ceilingRange.contains(ceiling) else {
                throw PiMPAudioCommandFacadeError.invalidDSPParameters(
                    "Loudness ceilingDBFS must be between \(LimiterParameters.ceilingRange.lowerBound) and \(LimiterParameters.ceilingRange.upperBound)."
                )
            }
            result.ceilingDBFS = ceiling
        }
        if let v = doubleValue(parameters["maxBoostDB"] ?? parameters["maxBoost"] ?? fallback["maxBoostDB"]) {
            result.maxBoostDB = Float(max(0, v))
        }
        return result
    }

    func editorNormalizeParameters(
        _ parameters: [String: Any],
        fallback: [String: Any]
    ) throws -> NormalizeParameters {
        let raw = parameters["targetPeakDBFS"]
            ?? parameters["targetPeak"]
            ?? parameters["peakDB"]
            ?? fallback["targetPeakDBFS"]
            ?? fallback["targetPeak"]
        guard let raw else { return .default }
        guard let value = doubleValue(raw) else {
            throw PiMPAudioCommandFacadeError.invalidDSPParameters("Normalize targetPeakDBFS must be a number.")
        }
        let peak = Float(value)
        guard NormalizeParameters.allowedRange.contains(peak) else {
            throw PiMPAudioCommandFacadeError.invalidDSPParameters(
                "Normalize targetPeakDBFS must be between \(NormalizeParameters.allowedRange.lowerBound) and \(NormalizeParameters.allowedRange.upperBound)."
            )
        }
        return NormalizeParameters(targetPeakDBFS: peak)
    }

    func editorTrimSilenceParameters(
        _ parameters: [String: Any],
        fallback: [String: Any]
    ) throws -> TrimSilenceParameters {
        var result = TrimSilenceParameters.default
        if let threshold = doubleValue(
            parameters["thresholdDBFS"] ?? parameters["threshold"] ?? fallback["thresholdDBFS"]
        ) {
            result.thresholdDBFS = Float(threshold)
        }
        if let maxLead = doubleValue(
            parameters["maxLeadingSeconds"] ?? parameters["maxLead"] ?? fallback["maxLeadingSeconds"]
        ) {
            result.maxLeadingSeconds = Float(max(0, maxLead))
        }
        if let maxTrail = doubleValue(
            parameters["maxTrailingSeconds"] ?? parameters["maxTrail"] ?? fallback["maxTrailingSeconds"]
        ) {
            result.maxTrailingSeconds = Float(max(0, maxTrail))
        }
        if let fadeMs = doubleValue(
            parameters["edgeFadeMilliseconds"] ?? parameters["fadeMs"] ?? fallback["edgeFadeMilliseconds"]
        ) {
            result.edgeFadeMilliseconds = Float(max(0, fadeMs))
        }
        return result
    }

    func editorHighPassParameters(
        _ parameters: [String: Any],
        fallback: [String: Any]
    ) throws -> HighPassParameters {
        let raw = parameters["cutoffHz"]
            ?? parameters["cutoff"]
            ?? parameters["frequency"]
            ?? fallback["cutoffHz"]
            ?? fallback["cutoff"]
        guard let raw else { return .default }
        guard let value = doubleValue(raw) else {
            throw PiMPAudioCommandFacadeError.invalidDSPParameters("High-pass cutoffHz must be a number.")
        }
        let cutoff = Float(value)
        guard HighPassParameters.allowedRange.contains(cutoff) else {
            throw PiMPAudioCommandFacadeError.invalidDSPParameters(
                "High-pass cutoffHz must be between \(HighPassParameters.allowedRange.lowerBound) and \(HighPassParameters.allowedRange.upperBound)."
            )
        }
        return HighPassParameters(cutoffHz: cutoff)
    }

    func editorEQParameters(
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

    func editorReverbParameters(
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

    func editorTimeStretchParameters(
        _ parameters: [String: Any],
        fallback: [String: Any]
    ) throws -> TimeStretchParameters {
        let rate = doubleValue(parameters["rate"] ?? parameters["ratio"] ?? fallback["rate"] ?? fallback["ratio"]) ?? 1.0
        guard rate >= 0.25, rate <= 4.0 else {
            throw PiMPAudioCommandFacadeError.invalidDSPParameters("Time-stretch rate must be between 0.25 and 4.0.")
        }
        return TimeStretchParameters(rate: Float(rate))
    }

    func editorPitchShiftParameters(
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

    func editorRenderSelection(
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

    func editorRenderOutputURL(
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

    func defaultEditorRenderFileName(
        loadedAudio: EditorLoadedAudio?,
        format: EditorRenderFormat
    ) -> String {
        let sourceBase = loadedAudio
            .map { ($0.fileName as NSString).deletingPathExtension }
            .flatMap { $0.isEmpty ? nil : $0 } ?? "editor-audio"
        let timestamp = renderFileDateFormatter.string(from: Date())
        return "\(sanitizedFileStem(sourceBase))-mcp-render-\(timestamp).\(format.fileExtension)"
    }

    func sanitizedFileStem(_ value: String) -> String {
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "-_ "))
        let sanitized = value.unicodeScalars.map { scalar in
            allowed.contains(scalar) ? Character(scalar) : "-"
        }
        let stem = String(sanitized).trimmingCharacters(in: .whitespacesAndNewlines)
        return stem.isEmpty ? "editor-audio" : stem
    }

    func urlWithRenderExtension(_ url: URL, format: EditorRenderFormat) -> URL {
        let ext = url.pathExtension.lowercased()
        guard ext != format.fileExtension else { return url }
        guard ext.isEmpty else { return url.deletingPathExtension().appendingPathExtension(format.fileExtension) }
        return url.appendingPathExtension(format.fileExtension)
    }

    func editorRenderFormat(from value: Any?) throws -> EditorRenderFormat {
        guard let rawValue = stringValue(value)?.trimmingCharacters(in: .whitespacesAndNewlines),
              !rawValue.isEmpty else {
            return .wav
        }
        guard let format = EditorRenderFormat(rawValue: rawValue.lowercased()) else {
            throw PiMPAudioCommandFacadeError.invalidRenderFormat(rawValue)
        }
        return format
    }

    func optionalEditorOriginalPrompt(
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

    func editorEditMode(from value: Any?) throws -> EditMode {
        guard let rawValue = stringValue(value)?.trimmingCharacters(in: .whitespacesAndNewlines),
              !rawValue.isEmpty else {
            return .inpaint
        }
        if let mode = EditMode(rawValue: rawValue.lowercased()) {
            return mode
        }
        throw PiMPAudioCommandFacadeError.invalidEditMode(rawValue)
    }

    func editorSelection(
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

    func editorSelectionValue(
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

    func editorFrequencyBand(
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
              let high = doubleValue(highValue) else {
            throw PiMPAudioCommandFacadeError.invalidFrequencyBand
        }
        do {
            return try FrequencyBand(lowCutHz: Float(low), highCutHz: Float(high))
        } catch is FrequencyBandError {
            throw PiMPAudioCommandFacadeError.invalidFrequencyBand
        }
    }

    func editorFrequencyValue(
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

    func validateEditorSelection(
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

    func editorInpaintStrength(from arguments: [String: Any]) throws -> Float {
        let preset = try editorInpaintPreset(from: arguments)
        let defaultStrength = preset?.strength ?? 0.6
        let rawValue = arguments["strength"] ?? arguments["denoiseStrength"] ?? arguments["editStrength"]
        guard let rawValue, !(rawValue is NSNull) else { return defaultStrength }
        guard let value = doubleValue(rawValue), value >= 0.0, value <= 1.0 else {
            throw PiMPAudioCommandFacadeError.invalidEditorStrength
        }
        return Float(value)
    }

    func editorInpaintGuidanceScale(from arguments: [String: Any]) throws -> Float {
        let preset = try editorInpaintPreset(from: arguments)
        let defaultGuidance = preset?.guidanceScale ?? 7.0
        let rawValue = arguments["guidanceScale"] ?? arguments["guidance"]
        guard let rawValue, !(rawValue is NSNull) else { return defaultGuidance }
        guard let value = doubleValue(rawValue), value > 0 else {
            throw PiMPAudioCommandFacadeError.invalidGuidanceScale
        }
        return Float(value)
    }

    func editorInpaintPreset(from arguments: [String: Any]) throws -> InpaintPreset? {
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

    func editorPipelineChoice(from arguments: [String: Any]) throws -> PipelineChoice {
        let rawValue = arguments["pipeline"] ?? arguments["pipelineOverride"]
        guard let rawValue, !(rawValue is NSNull) else {
            return .auto
        }
        guard let pipeline = pipelineChoice(from: rawValue) else {
            throw PiMPAudioCommandFacadeError.invalidPipeline(stringValue(rawValue) ?? "\(rawValue)")
        }
        return pipeline
    }

    func editorAudioLoadSource(from arguments: [String: Any]) throws -> EditorAudioLoadSource {
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
                url: try historyStore.fileURL(for: entry),
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

    func editorActionsPayload() -> [[String: Any]] {
        [
            [
                "id": "loadAudio",
                "tool": "load_editor_audio",
                "operationKind": "editor.loadAudio",
                "status": "implemented",
                "description": "Load a local audio file into the editor coordinator. VAE latents are encoded lazily on the first AI edit."
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

    func localFileURL(from rawPath: String) -> URL {
        let expanded = (rawPath as NSString).expandingTildeInPath
        return URL(fileURLWithPath: expanded)
    }

    func validateReadableFile(_ url: URL) throws {
        let path = url.path(percentEncoded: false)
        var isDirectory = ObjCBool(false)
        let exists = FileManager.default.fileExists(atPath: path, isDirectory: &isDirectory)
        guard exists, !isDirectory.boolValue else {
            throw PiMPAudioCommandFacadeError.editorAudioFileNotFound(path)
        }
    }


}
