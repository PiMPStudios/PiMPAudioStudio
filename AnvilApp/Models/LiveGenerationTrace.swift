import Foundation

enum LiveGenerationTraceStatus: String, Sendable, Equatable {
    case running
    case complete
    case failed
    case cancelled

    var displayName: String {
        switch self {
        case .running: return "Running"
        case .complete: return "Complete"
        case .failed: return "Failed"
        case .cancelled: return "Cancelled"
        }
    }
}

enum LiveTraceEventLevel: String, Sendable, Equatable {
    case info
    case success
    case warning
    case error
}

enum LiveGenerationTraceSource: String, Sendable, Equatable {
    case live
    case saved

    var displayName: String {
        switch self {
        case .live: return "Live Run"
        case .saved: return "Saved Trace"
        }
    }
}

struct LiveGenerationTraceEvent: Identifiable, Sendable, Equatable {
    let id: UUID
    let timestamp: Date
    let stage: String
    let message: String
    let detail: String?
    let progress: Double?
    let level: LiveTraceEventLevel
}

struct LiveGenerationProgress: Sendable, Equatable {
    let label: String
    let fraction: Double?
    let step: Int?
    let totalSteps: Int?
    let eta: TimeInterval?
}

struct LiveGenerationTraceSession: Identifiable, Sendable, Equatable {
    let id: UUID
    let startedAt: Date
    var updatedAt: Date
    var completedAt: Date?
    var status: LiveGenerationTraceStatus
    var source: LiveGenerationTraceSource
    var pipeline: String
    var prompt: String
    var lyrics: String
    var request: GenerationTraceRequestSnapshot
    var primaryModel: HistoryModelSnapshot?
    var supportingModels: [HistoryModelSnapshot]
    var adapters: [HistoryAdapterSnapshot]
    var events: [LiveGenerationTraceEvent]
    var progress: LiveGenerationProgress?
    var audio: GenerationTraceAudioSnapshot?
    var traceFileName: String?
    var errorMessage: String?
    var listeningNote: String?
    var isPreferred: Bool?
}

struct GenerationTraceComparison: Sendable, Equatable {
    let left: GenerationTrace
    let right: GenerationTrace
    let loadedAt: Date
}

@MainActor
final class GenerationTraceMonitor: ObservableObject {
    @Published private(set) var currentSession: LiveGenerationTraceSession?
    @Published private(set) var inspectedSession: LiveGenerationTraceSession?
    @Published private(set) var comparison: GenerationTraceComparison?
    @Published private(set) var recentSessions: [LiveGenerationTraceSession] = []

    private let maxEvents = 300
    private let maxRecentSessions = 8

    func start(request: GenerationRequest) {
        let now = Date()
        inspectedSession = nil
        comparison = nil
        currentSession = LiveGenerationTraceSession(
            id: UUID(),
            startedAt: now,
            updatedAt: now,
            completedAt: nil,
            status: .running,
            source: .live,
            pipeline: "Routing",
            prompt: request.text,
            lyrics: request.lyrics,
            request: Self.requestSnapshot(from: request),
            primaryModel: nil,
            supportingModels: [],
            adapters: [],
            events: [],
            progress: LiveGenerationProgress(
                label: "Preparing request",
                fraction: nil,
                step: nil,
                totalSteps: nil,
                eta: nil
            ),
            audio: nil,
            traceFileName: nil,
            errorMessage: nil,
            listeningNote: nil,
            isPreferred: nil
        )
        appendEvent(stage: "Request", message: "Generation request accepted", detail: request.text)
    }

    func inspect(trace: GenerationTrace) {
        comparison = nil
        inspectedSession = Self.session(from: trace)
    }

    func clearInspection() {
        inspectedSession = nil
    }

    func compare(left: GenerationTrace, right: GenerationTrace) {
        inspectedSession = nil
        comparison = GenerationTraceComparison(
            left: left,
            right: right,
            loadedAt: Date()
        )
    }

    func clearComparison() {
        comparison = nil
    }

    func updateComparison(left: GenerationTrace, right: GenerationTrace) {
        guard comparison != nil else { return }
        comparison = GenerationTraceComparison(
            left: left,
            right: right,
            loadedAt: Date()
        )
    }

    func configure(
        pipeline: String,
        primaryModel: HistoryModelSnapshot?,
        supportingModels: [HistoryModelSnapshot],
        adapters: [HistoryAdapterSnapshot]
    ) {
        guard var session = currentSession else { return }
        session.pipeline = pipeline
        session.primaryModel = primaryModel
        session.supportingModels = supportingModels
        session.adapters = adapters
        session.updatedAt = Date()
        currentSession = session

        let modelName = primaryModel?.displayName ?? primaryModel?.id ?? "Unknown model"
        appendEvent(
            stage: "Routing",
            message: "Using \(pipeline) pipeline",
            detail: modelName,
            level: .info
        )

        if adapters.isEmpty {
            appendEvent(stage: "Adapters", message: "No LoRA adapters selected")
        } else {
            let names = adapters.map { adapter in
                let label = adapter.name ?? adapter.id
                return "\(label) x\(String(format: "%.2f", adapter.weight))"
            }
            appendEvent(stage: "Adapters", message: "Loaded adapter plan", detail: names.joined(separator: ", "))
        }
    }

    func recordStage(
        _ stage: String,
        message: String,
        detail: String? = nil,
        level: LiveTraceEventLevel = .info
    ) {
        appendEvent(stage: stage, message: message, detail: detail, level: level)
        updateProgress(label: message, fraction: nil, step: nil, totalSteps: nil, eta: nil)
    }

    func updateLoadingWeights(progress: Float) {
        let fraction = Double(progress.clamped(to: 0...1))
        updateProgress(
            label: "Loading model weights",
            fraction: fraction,
            step: nil,
            totalSteps: nil,
            eta: nil
        )
    }

    func updateGenerationStep(step: Int, totalSteps: Int, eta: TimeInterval) {
        let total = max(totalSteps, 1)
        let fraction = Double(step) / Double(total)
        updateProgress(
            label: "Diffusion step \(step) / \(totalSteps)",
            fraction: fraction,
            step: step,
            totalSteps: totalSteps,
            eta: eta
        )

        if step == 1 || step == totalSteps || step.isMultiple(of: 5) {
            appendEvent(
                stage: "Diffusion",
                message: "Step \(step) / \(totalSteps)",
                detail: eta > 0 ? "~\(Int(eta))s remaining" : nil,
                progress: fraction
            )
        }
    }

    func finish(trace: GenerationTrace) {
        guard var session = currentSession else { return }
        let now = Date()
        session.status = .complete
        session.updatedAt = now
        session.completedAt = now
        session.audio = trace.audio
        session.traceFileName = trace.traceFileName
        session.progress = LiveGenerationProgress(
            label: "Trace written",
            fraction: 1,
            step: nil,
            totalSteps: nil,
            eta: nil
        )
        currentSession = session
        appendEvent(
            stage: "Complete",
            message: "Generation trace written",
            detail: trace.traceFileName,
            progress: 1,
            level: .success
        )
        archiveCurrentSession()
    }

    func fail(_ message: String) {
        guard var session = currentSession else { return }
        let now = Date()
        session.status = .failed
        session.updatedAt = now
        session.completedAt = now
        session.errorMessage = message
        session.progress = LiveGenerationProgress(
            label: "Failed",
            fraction: nil,
            step: nil,
            totalSteps: nil,
            eta: nil
        )
        currentSession = session
        appendEvent(stage: "Error", message: message, level: .error)
        archiveCurrentSession()
    }

    func cancel() {
        guard var session = currentSession else { return }
        let now = Date()
        session.status = .cancelled
        session.updatedAt = now
        session.completedAt = now
        session.progress = LiveGenerationProgress(
            label: "Cancelled",
            fraction: nil,
            step: nil,
            totalSteps: nil,
            eta: nil
        )
        currentSession = session
        appendEvent(stage: "Cancelled", message: "Generation cancelled", level: .warning)
        archiveCurrentSession()
    }

    func failureTrace(fileName: String, errorMessage: String) -> GenerationFailureTrace? {
        guard let session = currentSession else { return nil }
        return GenerationFailureTrace(
            schemaVersion: GenerationFailureTrace.currentSchemaVersion,
            traceType: "failure",
            id: session.id,
            createdAt: session.startedAt,
            completedAt: session.completedAt,
            traceFileName: fileName,
            pipeline: session.pipeline,
            prompt: session.prompt,
            lyrics: session.lyrics,
            request: session.request,
            primaryModel: session.primaryModel,
            supportingModels: session.supportingModels,
            adapters: session.adapters,
            errorMessage: errorMessage,
            events: session.events.map(Self.eventSnapshot)
        )
    }

    func currentEventSnapshots() -> [GenerationTraceEventSnapshot] {
        currentSession?.events.map(Self.eventSnapshot) ?? []
    }

    private func updateProgress(
        label: String,
        fraction: Double?,
        step: Int?,
        totalSteps: Int?,
        eta: TimeInterval?
    ) {
        guard var session = currentSession else { return }
        session.progress = LiveGenerationProgress(
            label: label,
            fraction: fraction,
            step: step,
            totalSteps: totalSteps,
            eta: eta
        )
        session.updatedAt = Date()
        currentSession = session
    }

    private func appendEvent(
        stage: String,
        message: String,
        detail: String? = nil,
        progress: Double? = nil,
        level: LiveTraceEventLevel = .info
    ) {
        guard var session = currentSession else { return }
        let event = LiveGenerationTraceEvent(
            id: UUID(),
            timestamp: Date(),
            stage: stage,
            message: message,
            detail: detail,
            progress: progress,
            level: level
        )
        session.events.append(event)
        if session.events.count > maxEvents {
            session.events.removeFirst(session.events.count - maxEvents)
        }
        session.updatedAt = event.timestamp
        currentSession = session
    }

    private func archiveCurrentSession() {
        guard let session = currentSession else { return }
        recentSessions.removeAll { $0.id == session.id }
        recentSessions.insert(session, at: 0)
        if recentSessions.count > maxRecentSessions {
            recentSessions.removeLast(recentSessions.count - maxRecentSessions)
        }
    }

    private static func requestSnapshot(from request: GenerationRequest) -> GenerationTraceRequestSnapshot {
        GenerationTraceRequestSnapshot(
            duration: request.duration,
            seed: request.seed,
            method: request.method.rawValue,
            guidanceScale: request.effectiveGuidanceScale,
            guidancePreset: request.guidancePreset.rawValue,
            guidancePresetName: request.guidancePreset.displayName,
            cfgIntervalStart: request.cfgIntervalStart,
            cfgIntervalEnd: request.cfgIntervalEnd,
            precision: request.precision.rawValue,
            stepsOverride: request.stepsOverride,
            shiftOverride: request.shiftOverride,
            resolvedMode: request.resolvedMode.rawValue,
            pipelineOverride: request.pipelineOverride.rawValue,
            coverStrength: request.coverStrength,
            referenceAudioFileName: request.referenceAudioURL?.lastPathComponent
        )
    }

    private static func session(from trace: GenerationTrace) -> LiveGenerationTraceSession {
        let loadedAt = Date()
        let traceEvents = trace.events?.map { snapshot in
            LiveGenerationTraceEvent(
                id: UUID(),
                timestamp: snapshot.timestamp,
                stage: snapshot.stage,
                message: snapshot.message,
                detail: snapshot.detail,
                progress: snapshot.progress,
                level: LiveTraceEventLevel(rawValue: snapshot.level) ?? .info
            )
        } ?? []
        let events = traceEvents.isEmpty ? [
            LiveGenerationTraceEvent(
                id: UUID(),
                timestamp: loadedAt,
                stage: "Trace",
                message: "Loaded saved generation trace",
                detail: trace.traceFileName,
                progress: 1,
                level: .success
            ),
            LiveGenerationTraceEvent(
                id: UUID(),
                timestamp: trace.createdAt,
                stage: "Audio",
                message: "Saved audio snapshot",
                detail: String(format: "absMean %.4f, absMax %.3f", trace.audio.absMean, trace.audio.absMax),
                progress: nil,
                level: .info
            ),
            LiveGenerationTraceEvent(
                id: UUID(),
                timestamp: trace.createdAt,
                stage: "Complete",
                message: "Generation completed",
                detail: trace.audioFileName,
                progress: 1,
                level: .success
            )
        ] : traceEvents

        return LiveGenerationTraceSession(
            id: trace.historyEntryID,
            startedAt: trace.createdAt,
            updatedAt: loadedAt,
            completedAt: trace.createdAt.addingTimeInterval(trace.timing.totalTime),
            status: .complete,
            source: .saved,
            pipeline: trace.pipeline,
            prompt: trace.prompt,
            lyrics: trace.lyrics,
            request: trace.request,
            primaryModel: trace.primaryModel,
            supportingModels: trace.supportingModels,
            adapters: trace.adapters,
            events: events,
            progress: LiveGenerationProgress(
                label: "Saved trace loaded",
                fraction: 1,
                step: nil,
                totalSteps: nil,
                eta: nil
            ),
            audio: trace.audio,
            traceFileName: trace.traceFileName,
            errorMessage: nil,
            listeningNote: trace.listeningNote,
            isPreferred: trace.isPreferred
        )
    }

    private static func eventSnapshot(_ event: LiveGenerationTraceEvent) -> GenerationTraceEventSnapshot {
        GenerationTraceEventSnapshot(
            timestamp: event.timestamp,
            stage: event.stage,
            message: event.message,
            detail: event.detail,
            progress: event.progress,
            level: event.level.rawValue
        )
    }
}

private extension Comparable {
    func clamped(to limits: ClosedRange<Self>) -> Self {
        min(max(self, limits.lowerBound), limits.upperBound)
    }
}
