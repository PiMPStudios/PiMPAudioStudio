import Foundation
import AVFoundation
import PiMPAudioCore

enum MCPTrainingRecipe: String {
    case narrow
    case wide
}

enum MCPTrainingProjectionCoverage {
    case queryValue
    case fullAttention
}

struct MCPTrainingDefaults {
    let dimension: Int
    let epochs: Int
    let layerCount: Int
    let projectionCoverage: MCPTrainingProjectionCoverage
    let alphaScale: Float
    let dropout: Float
    let targetLayers: Set<TargetLayer>
    let gradientAccumulationSteps: Int
    let learningRate: Float
    let weightDecompose: Bool
}

struct MCPTrainingRequest {
    let kind: AdapterKind
    let recipe: MCPTrainingRecipe
    let dataset: DatasetPackageDocument
    let importedFiles: [TrainingImportedFile]
    let adapterName: String
    let config: LoRAConfig
    let lokrConfig: LoKRConfig?
}

enum PiMPAudioCommandFacadeError: LocalizedError {
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
            return "Adapter strength must be a number from 0.0 to 6.0."
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
            return "Unsupported training recipe '\(value)'. Use narrow or wide."
        case .invalidTrainingConfiguration(let reason):
            return reason
        case .encodingFailed:
            return "Could not encode the trace as JSON."
        }
    }
}

struct AdapterSelection {
    let adapter: LoRAAdapter
    let strength: Float
}

struct OrchestrationAdapterPlan {
    let shouldClear: Bool
    let applyArguments: [String: Any]?

    var hasAction: Bool {
        shouldClear || applyArguments != nil
    }
}

struct EditorAudioLoadSource {
    let kind: String
    let url: URL
    let fileName: String
    let historyEntryID: String?
    let originalPrompt: String?
}

struct EditorRenderRequest {
    let format: EditorRenderFormat
    let outputURL: URL
    let selection: InpaintSelection?
    let overwrite: Bool
}

struct EditorDSPRequest {
    let steps: [EditorDSPEffectStep]
    let description: String
}

struct EditorDSPEffectStep {
    let effect: DSPEffect
    let target: ApplyTarget
    let parameters: DSPParameters
}

enum EditorRenderFormat: String {
    case wav
    case aiff

    var fileExtension: String {
        switch self {
        case .wav: return "wav"
        case .aiff: return "aiff"
        }
    }
}

enum MCPOperationStatus: String {
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

struct MCPOperationRecord {
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

struct WorkflowPromptArgument {
    let name: String
    let description: String
    let required: Bool
}

struct WorkflowPromptDefinition {
    let id: String
    let title: String
    let description: String
    let promptName: String
    let arguments: [WorkflowPromptArgument]
    let resources: [String]
    let tools: [String]
    let template: String
}

let workflowPromptDefinitions: [WorkflowPromptDefinition] = [
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

struct PiMPAudioResourceURI {
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

extension ModelStatus {
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
