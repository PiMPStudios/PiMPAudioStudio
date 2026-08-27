import AVFoundation
import Foundation
import PiMPAudioCore

extension PiMPAudioCommandFacade {
    func listOperations(arguments: [String: Any]) -> [String: Any] {
        refreshGenerationOperation()
        let requestedLimit = doubleValue(arguments["limit"]).map { Int($0) } ?? 20
        let cappedLimit = max(1, min(requestedLimit, maxOperationRecords))
        let operations = operationSnapshot(limit: cappedLimit).map(operationPayload)
        return [
            "count": operations.count,
            "currentOperationID": activeGenerationOperationID.map { $0 as Any } ?? NSNull(),
            "activeOperationIDs": activeOperationIDsPayload(),
            "operations": operations,
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
        guard boolValue(arguments["confirm"]) == true else {
            throw PiMPAudioCommandFacadeError.confirmationRequired("cancel_operation")
        }
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
                "operation": operationPayload(record),
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
            "operation": operationPayload(record),
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

    func memoryReleasePayload(_ summary: MemoryReleaseSummary) -> [String: Any] {
        [
            "released": true,
            "before": [
                "activeMemoryBytes": summary.beforeActiveMemory,
                "cacheMemoryBytes": summary.beforeCacheMemory,
                "peakMemoryBytes": summary.beforePeakMemory,
            ],
            "after": [
                "activeMemoryBytes": summary.afterActiveMemory,
                "cacheMemoryBytes": summary.afterCacheMemory,
                "peakMemoryBytes": summary.afterPeakMemory,
            ],
            "delta": [
                "releasedActiveMemoryBytes": summary.releasedActiveMemory,
                "releasedCacheMemoryBytes": summary.releasedCacheMemory,
            ],
        ]
    }

    func registerGenerationOperation(request: GenerationRequest) -> MCPOperationRecord {
        let record = registerOperation(
            kind: "generation",
            title: "Generate audio",
            request: generationRequestPayload(request),
            progress: [
                "stage": "queued",
                "progress": 0.0,
                "total": 1.0,
                "message": "Generation request accepted.",
            ]
        )

        activeGenerationOperationID = record.id
        return record
    }

    func refreshGenerationOperation() {
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
        case let .error(message):
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
                "message": "Generation request is queued.",
            ]
        default:
            record.status = .running
        }

        operationRecords[id] = record
    }

    func generationOperationProgressPayload(_ state: GenerationState) -> [String: Any] {
        switch state {
        case .idle:
            return generationOperationProgressPayload(
                stage: "idle",
                message: "Generation service is idle."
            )
        case let .downloadingModel(progress):
            return generationOperationProgressPayload(
                stage: "downloadingModel",
                message: "Downloading model weights.",
                stageProgress: Double(progress),
                stageTotal: 1.0
            )
        case let .loadingWeights(progress):
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
        case let .generating(step, totalSteps, eta):
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
        case let .error(message):
            return generationOperationProgressPayload(
                stage: "error",
                message: message
            )
        case .cancelled:
            return generationOperationProgressPayload(
                stage: "cancelled",
                message: "Generation cancelled."
            )
        case .releasing:
            return generationOperationProgressPayload(
                stage: "releasing",
                message: "Releasing generation models from memory."
            )
        }
    }

    func generationOperationProgressPayload(
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

    func cancellationProgressPayload(for record: MCPOperationRecord) -> [String: Any] {
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
            "message": "Operation cancelled.",
        ]
    }

    func operationProgressPayload(
        stage: String,
        message: String,
        plan: [String: (base: Double, span: Double)],
        stageProgress: Double? = nil,
        stageTotal: Double? = nil
    ) -> [String: Any] {
        let plan = plan[stage] ?? (base: 0.0, span: 0.0)
        let stageRatio: Double = if let stageProgress, let stageTotal, stageTotal > 0 {
            min(max(stageProgress / stageTotal, 0), 1)
        } else {
            plan.span > 0 ? 0.0 : 1.0
        }

        let progress = min(max(plan.base + (plan.span * stageRatio), 0), 1)
        var payload: [String: Any] = [
            "stage": stage,
            "progress": progress,
            "total": 1.0,
            "message": message,
        ]

        if let stageProgress {
            payload["stageProgress"] = stageProgress
        }
        if let stageTotal {
            payload["stageTotal"] = stageTotal
        }
        return payload
    }

    func runEditorAudioLoadOperation(
        id: String,
        source: EditorAudioLoadSource,
        originalPrompt: String?
    ) async {
        defer {
            operationTasks[id] = nil
            releaseEditorOperation(id: id)
        }

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
                progress: editorLoadProgressPayload(
                    stage: "loadingAudio",
                    message: "Loading editor audio (VAE encode deferred until AI edit)."
                )
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
                    "loadedAudio": editorLoadedAudioPayload(loadedAudio),
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

    func runEditorInpaintOperation(id: String, request: InpaintRequest) async {
        defer {
            operationTasks[id] = nil
            releaseEditorOperation(id: id)
        }

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
                    await completeOperation(
                        id: id,
                        progress: editorInpaintProgressPayload(progress),
                        result: [
                            "edit": editorEditSummaryPayload(summary),
                            "canUndo": coordinator.canUndo,
                            "canRedo": coordinator.canRedo,
                        ]
                    )
                    return
                case let .failed(message):
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

    func runEditorRenderOperation(id: String, request: EditorRenderRequest) async {
        defer {
            operationTasks[id] = nil
            releaseEditorOperation(id: id)
        }

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

    func runEditorDSPOperation(id: String, request: EditorDSPRequest) async {
        defer {
            operationTasks[id] = nil
            releaseEditorOperation(id: id)
        }

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
                progress: editorDSPProgressPayload(
                    stage: "updatingAudio",
                    message: "Updating editor audio (VAE encode deferred until AI edit)."
                )
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

    func readAudioBuffer(from url: URL) async throws -> AudioBuffer {
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

    func writeAudioBuffer(_ buffer: AudioBuffer, to url: URL) throws {
        let pcmBuffer = try AudioBufferConverter.toPCMBuffer(buffer)
        let outputFile = try AVAudioFile(forWriting: url, settings: pcmBuffer.format.settings)
        try outputFile.write(from: pcmBuffer)
    }

    func croppedAudioBuffer(
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
            data: Data(buffer.data[startByte ..< endByte]),
            sampleRate: buffer.sampleRate,
            channels: buffer.channels
        )
    }

    func registerOperation(
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

    func reserveEditorOperation(id: String) throws {
        guard activeEditorOperationID == nil else {
            throw PiMPAudioCommandFacadeError.editorOperationAlreadyActive
        }
        activeEditorOperationID = id
    }

    func releaseEditorOperation(id: String) {
        guard activeEditorOperationID == id else { return }
        activeEditorOperationID = nil
    }

    func updateOperationProgress(id: String, progress: [String: Any]) {
        guard var record = operationRecords[id], !record.status.isTerminal else { return }
        record.updatedAt = Date()
        record.progress = progress
        operationRecords[id] = record
    }

    func completeOperation(
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

    func failOperation(id: String, error: String, progress: [String: Any]) {
        guard var record = operationRecords[id], !record.status.isTerminal else { return }
        record.updatedAt = Date()
        record.status = .failed
        record.progress = progress
        record.error = error
        operationRecords[id] = record
    }

    func cancelOperationRecord(id: String, reason: String, progress: [String: Any]) {
        guard var record = operationRecords[id], !record.status.isTerminal else { return }
        record.updatedAt = Date()
        record.status = .cancelled
        record.progress = progress
        record.cancellationReason = reason
        operationRecords[id] = record
    }

    func operationSnapshot(limit: Int) -> [MCPOperationRecord] {
        operationOrder
            .prefix(limit)
            .compactMap { operationRecords[$0] }
    }

    func operationPayload(_ record: MCPOperationRecord) -> [String: Any] {
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
            "cancellationReason": record.cancellationReason.map { $0 as Any } ?? NSNull(),
        ]
    }

    func currentOperationPayload() -> Any? {
        if let id = activeGenerationOperationID,
           let record = operationRecords[id] {
            return operationPayload(record)
        }
        guard let latest = operationSnapshot(limit: 1).first else {
            return nil
        }
        return operationPayload(latest)
    }

    func currentOperationPayload(kindPrefix: String) -> Any? {
        operationOrder
            .compactMap { operationRecords[$0] }
            .first { $0.kind.hasPrefix(kindPrefix) && !$0.status.isTerminal }
            .map(operationPayload)
    }

    func activeOperationIDsPayload() -> [String: String] {
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

    func isOperationCancelable(_ record: MCPOperationRecord) -> Bool {
        if record.status.isTerminal {
            return false
        }
        if record.kind == "generation", record.id == activeGenerationOperationID {
            return true
        }
        return operationTasks[record.id] != nil
    }

    func operationID(from arguments: [String: Any]) throws -> String {
        let id = stringValue(arguments["id"] ?? arguments["operationID"] ?? arguments["operationId"])?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        guard !id.isEmpty else {
            throw PiMPAudioCommandFacadeError.missingRequiredArgument("id")
        }
        return id
    }

    func trimOperationRecords() {
        let pinned = Set(
            [activeGenerationOperationID, activeEditorOperationID].compactMap { $0 }
        )
        let trimmed = Self.trimmedOperationOrder(
            order: operationOrder,
            isPinned: { pinned.contains($0) },
            isTerminal: { operationRecords[$0]?.status.isTerminal ?? true },
            maxCount: maxOperationRecords
        )
        guard trimmed.evicted.isEmpty == false else { return }
        operationOrder = trimmed.kept
        for id in trimmed.evicted {
            operationRecords[id] = nil
            operationTasks[id] = nil
        }
    }

    /// Newest-first `order`. Evict oldest terminal records first. Never drop
    /// pinned or non-terminal ids, even if that exceeds `maxCount`.
    nonisolated static func trimmedOperationOrder(
        order: [String],
        isPinned: (String) -> Bool,
        isTerminal: (String) -> Bool,
        maxCount: Int
    ) -> (kept: [String], evicted: [String]) {
        let cap = max(1, maxCount)
        var live: [String] = []
        var terminal: [String] = []
        for id in order {
            if isPinned(id) || !isTerminal(id) {
                live.append(id)
            } else {
                terminal.append(id)
            }
        }
        let terminalBudget = max(0, cap - live.count)
        let keptTerminal = Array(terminal.prefix(terminalBudget))
        let evicted = Array(terminal.dropFirst(terminalBudget))
        return (live + keptTerminal, evicted)
    }
}
