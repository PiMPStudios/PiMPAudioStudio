import Foundation
import AVFoundation
import PiMPAudioCore

@MainActor
final class PiMPAudioCommandFacade {
    let registry: ModelRegistry
    let promptEnhancer: PromptEnhancer
    let lyricWriterManager: LyricWriterManager
    var historyStore: HistoryStore
    var generationService: GenerationService?
    var editorService: EditorService?
    var trainingService: TrainingService?
    let dateFormatter = ISO8601DateFormatter()
    let renderFileDateFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd_HHmmss"
        return formatter
    }()
    let maxOperationRecords = 50
    var operationRecords: [String: MCPOperationRecord] = [:]
    var operationOrder: [String] = []
    var operationTasks: [String: Task<Void, Never>] = [:]
    var activeGenerationOperationID: String?
    let generationProgressPlan: [String: (base: Double, span: Double)] = [
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
    let editorLoadProgressPlan: [String: (base: Double, span: Double)] = [
        "queued": (0.00, 0.02),
        "readingFile": (0.02, 0.18),
        "preparingAudio": (0.20, 0.15),
        "encodingLatents": (0.35, 0.60),
        "complete": (1.00, 0.00),
        "error": (1.00, 0.00),
        "cancelled": (1.00, 0.00)
    ]
    let editorInpaintProgressPlan: [String: (base: Double, span: Double)] = [
        "queued": (0.00, 0.02),
        "buildingMask": (0.02, 0.06),
        "encoding": (0.08, 0.10),
        "diffusion": (0.18, 0.70),
        "decoding": (0.88, 0.10),
        "complete": (1.00, 0.00),
        "error": (1.00, 0.00),
        "cancelled": (1.00, 0.00)
    ]
    let editorRenderProgressPlan: [String: (base: Double, span: Double)] = [
        "queued": (0.00, 0.02),
        "readingBuffer": (0.02, 0.18),
        "preparingAudio": (0.20, 0.30),
        "writingFile": (0.50, 0.45),
        "complete": (1.00, 0.00),
        "error": (1.00, 0.00),
        "cancelled": (1.00, 0.00)
    ]
    let editorDSPProgressPlan: [String: (base: Double, span: Double)] = [
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

}
