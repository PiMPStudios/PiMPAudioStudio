import Foundation
import MCP

enum MCPToolCatalog {
    static var sdkTools: [Tool] {
        tools.compactMap(sdkTool)
    }

    static let tools: [[String: Any]] = [
        tool(
            name: "list_models",
            title: "List Models",
            description: "Return PiMP Audio Studio model registry status, active models, readiness, and runtime support notes.",
            properties: [:]
        ),
        tool(
            name: "list_adapters",
            title: "List Adapters",
            description: "Return installed LoRA adapters with base model, rank, provenance, and checkpoint status.",
            properties: [:]
        ),
        tool(
            name: "training_status",
            title: "Training Status",
            description: "Return the current local LoRA/LoKR training state, active progress metrics, run-log summary, and diagnostics log paths.",
            properties: [:]
        ),
        tool(
            name: "list_training_datasets",
            title: "List Training Datasets",
            description: "Return local training dataset packages from the current output folder and app support fallback.",
            properties: [
                "limit": [
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 200,
                    "description": "Maximum number of dataset packages to return. Defaults to 50.",
                ],
            ]
        ),
        tool(
            name: "list_training_runs",
            title: "List Training Runs",
            description: "Return installed adapter training logs with latest loss/LR/norm metrics and checkpoint status.",
            properties: [
                "limit": [
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "description": "Maximum number of training runs to return. Defaults to 25.",
                ],
                "includeAdaptersWithoutLogs": [
                    "type": "boolean",
                    "description": "Include installed adapters that do not have training_log.json. Defaults to false.",
                ],
            ]
        ),
        tool(
            name: "inspect_training_dataset",
            title: "Inspect Training Dataset",
            description: "Return detailed read-only health, annotation coverage, recommendation, and sample-clip metadata for a local dataset package.",
            properties: [
                "id": [
                    "type": "string",
                    "description": "Dataset id, folder name, or partial dataset name. Defaults to the newest visible dataset.",
                ],
                "name": [
                    "type": "string",
                    "description": "Alias for id.",
                ],
                "path": [
                    "type": "string",
                    "description": "Optional full path to a dataset package containing captions.json.",
                ],
                "sampleLimit": [
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 50,
                    "description": "Maximum number of sample clip rows to include. Defaults to 12.",
                ],
            ]
        ),
        tool(
            name: "inspect_training_run",
            title: "Inspect Training Run",
            description: "Return detailed read-only adapter training metadata, configuration, health metrics, progress tail, and event tail.",
            properties: [
                "id": [
                    "type": "string",
                    "description": "Adapter id, adapter name, or partial adapter name. Defaults to the newest installed training run.",
                ],
                "adapterID": [
                    "type": "string",
                    "description": "Alias for id.",
                ],
                "name": [
                    "type": "string",
                    "description": "Alias for id.",
                ],
                "progressLimit": [
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 500,
                    "description": "Maximum number of progress rows from the end of training_log.json. Defaults to 50.",
                ],
                "eventLimit": [
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 100,
                    "description": "Maximum number of run events from the end of training_log.json. Defaults to 25.",
                ],
            ]
        ),
        tool(
            name: "compare_training_runs",
            title: "Compare Training Runs",
            description: "Return a compact read-only comparison table for recent or selected adapter training runs.",
            properties: [
                "ids": [
                    "type": "array",
                    "items": ["type": "string"],
                    "description": "Optional adapter ids or names to compare. Defaults to recent logged runs.",
                ],
                "adapterIDs": [
                    "type": "array",
                    "items": ["type": "string"],
                    "description": "Alias for ids.",
                ],
                "limit": [
                    "type": "integer",
                    "minimum": 2,
                    "maximum": 20,
                    "description": "Maximum number of recent runs to compare when ids are omitted. Defaults to 6.",
                ],
                "includeAdaptersWithoutLogs": [
                    "type": "boolean",
                    "description": "Include installed adapters that do not have training_log.json. Defaults to false.",
                ],
            ]
        ),
        tool(
            name: "tail_training_log",
            title: "Tail Training Log",
            description: "Return recent lines from the active or latest AUDIODIAG diagnostics log, filtered to training lines by default.",
            properties: [
                "limit": [
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 500,
                    "description": "Maximum number of matching lines to return. Defaults to 120.",
                ],
                "contains": [
                    "type": "string",
                    "description": "Case-insensitive line filter. Defaults to training. Pass an empty string for unfiltered lines.",
                ],
                "filter": [
                    "type": "string",
                    "description": "Alias for contains.",
                ],
                "fileName": [
                    "type": "string",
                    "description": "Optional diagnostics log file name under the app Diagnostics folder.",
                ],
                "path": [
                    "type": "string",
                    "description": "Optional full path to a diagnostics log under the app Diagnostics folder.",
                ],
                "logPath": [
                    "type": "string",
                    "description": "Alias for path.",
                ],
            ]
        ),
        tool(
            name: "training_preflight",
            title: "Training Preflight",
            description: "Resolve a local LoRA/LoKR dataset and configuration without starting training, then return the same memory/config guard report used before local training.",
            properties: trainingControlProperties(includeStartControls: false)
        ),
        tool(
            name: "start_training",
            title: "Start Training",
            description: "Preflight or start a local LoRA/LoKR training run from a reviewed dataset package. Defaults to dryRun: true; actual runs require dryRun: false plus confirm: true or MCP form approval.",
            properties: trainingControlProperties(includeStartControls: true)
        ),
        tool(
            name: "stop_training",
            title: "Stop Training",
            description: "Stop the active local LoRA/LoKR training run through the same TrainingService stop path as the UI. Requires confirm: true or MCP form approval.",
            properties: [
                "confirm": [
                    "type": "boolean",
                    "description": "Required for clients that do not support MCP elicitation.",
                ],
                "saveCheckpoint": [
                    "type": "boolean",
                    "description": "Whether to save a checkpoint when possible. Defaults to the app's canSaveCheckpointOnStop state.",
                ],
                "checkpoint": [
                    "type": "boolean",
                    "description": "Alias for saveCheckpoint.",
                ],
                "save": [
                    "type": "boolean",
                    "description": "Alias for saveCheckpoint.",
                ],
            ]
        ),
        tool(
            name: "runpod_status",
            title: "RunPod Status",
            description: "Return RunPod API-key state, cached live GPU/template launch options, pods, and the recommended official template for an optional GPU type.",
            properties: [
                "gpuType": [
                    "type": "string",
                    "description": "Optional GPU type to use when returning the recommended official RunPod PyTorch template.",
                ],
                "refreshLaunchOptions": [
                    "type": "boolean",
                    "description": "When true, refresh GPU availability and official RunPod templates before returning status.",
                ],
                "refreshPods": [
                    "type": "boolean",
                    "description": "When true, refresh the RunPod pod list before returning status.",
                ],
            ]
        ),
        tool(
            name: "refresh_runpod_launch_options",
            title: "Refresh RunPod Launch Options",
            description: "Refresh live Secure RunPod GPU availability and official RunPod pod templates, then return the recommended official PyTorch template for an optional GPU type.",
            properties: [
                "gpuType": [
                    "type": "string",
                    "description": "Optional GPU type to use when returning the recommended official RunPod PyTorch template.",
                ],
            ]
        ),
        tool(
            name: "launch_runpod_pod",
            title: "Launch RunPod Pod",
            description: "Preflight or launch a RunPod pod for remote training. Defaults to dryRun: true; actual launches require dryRun: false plus confirm: true or MCP form approval because they create billable GPU time.",
            required: ["gpuType"],
            properties: [
                "gpuType": [
                    "type": "string",
                    "description": "RunPod GPU type id or display name, for example NVIDIA GeForce RTX 4090.",
                ],
                "region": [
                    "type": "string",
                    "description": "Optional region filter for Secure Cloud launch data centers: any, asia, northAmerica, europe, oceania.",
                ],
                "dataCenterIDs": [
                    "type": "array",
                    "items": ["type": "string"],
                    "description": "Optional explicit RunPod data center ids. Overrides region when provided.",
                ],
                "templateID": [
                    "type": "string",
                    "description": "Optional RunPod template id. If omitted with no imageName, PiMP Audio Studio tries the recommended official RunPod PyTorch pod template.",
                ],
                "imageName": [
                    "type": "string",
                    "description": "Optional container image. A custom image overrides templateID at launch time.",
                ],
                "useRecommendedTemplate": [
                    "type": "boolean",
                    "description": "Defaults to true. When templateID and imageName are empty, use the recommended official RunPod PyTorch pod template if available.",
                ],
                "name": [
                    "type": "string",
                    "description": "Optional RunPod pod name. Defaults to a PiMP Audio Studio generated name.",
                ],
                "containerDiskGB": [
                    "type": "integer",
                    "minimum": 20,
                    "description": "Container disk size in GB. Defaults to 80 and is raised to the selected template's default when needed.",
                ],
                "volumeGB": [
                    "type": "integer",
                    "minimum": 20,
                    "description": "Persistent volume size in GB. Defaults to 80 and is raised to the selected template's default when needed.",
                ],
                "dryRun": [
                    "type": "boolean",
                    "description": "Defaults to true. Set false, with confirmation, to actually create billable RunPod GPU time.",
                ],
                "validateOnly": [
                    "type": "boolean",
                    "description": "Alias for dryRun.",
                ],
                "confirm": [
                    "type": "boolean",
                    "description": "Required for actual launches when the client does not support MCP elicitation.",
                ],
            ]
        ),
        tool(
            name: "stop_runpod_pod",
            title: "Terminate RunPod Pod",
            description: "Terminate/delete a RunPod pod through the same RunPodService path as the UI. Requires confirm: true or MCP form approval. This uses RunPod DELETE /v2/pods/{id}, not a pause/resume stop action.",
            required: ["podID"],
            properties: [
                "podID": [
                    "type": "string",
                    "description": "RunPod pod id to terminate.",
                ],
                "confirm": [
                    "type": "boolean",
                    "description": "Required for clients that do not support MCP elicitation.",
                ],
            ]
        ),
        tool(
            name: "list_history",
            title: "List History",
            description: "Return recent generation history rows with model, seed, prompt summary, and trace availability.",
            properties: [
                "limit": [
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 200,
                    "description": "Maximum number of history rows to return. Defaults to 25.",
                ],
            ]
        ),
        tool(
            name: "get_trace",
            title: "Get Trace",
            description: "Return a generation trace by history id, trace filename, or audio filename.",
            properties: [
                "id": [
                    "type": "string",
                    "description": "History UUID, trace filename, or audio filename.",
                ],
                "historyEntryID": ["type": "string"],
                "traceFileName": ["type": "string"],
                "audioFileName": ["type": "string"],
            ]
        ),
        tool(
            name: "compare_traces",
            title: "Compare Traces",
            description: "Return the same grouped changed-field diff used by the Generation Analyzer compare view.",
            required: ["left", "right"],
            properties: [
                "left": [
                    "type": "string",
                    "description": "Left trace identifier: history UUID, trace filename, or audio filename.",
                ],
                "right": [
                    "type": "string",
                    "description": "Right trace identifier: history UUID, trace filename, or audio filename.",
                ],
            ]
        ),
        tool(
            name: "mark_preferred_take",
            title: "Mark Preferred Take",
            description: "Save a preferred flag and optional listening note into an existing generation trace and mirrored history row. Requires confirm: true or MCP form approval.",
            properties: [
                "id": [
                    "type": "string",
                    "description": "History UUID, trace filename, or audio filename.",
                ],
                "historyEntryID": ["type": "string"],
                "traceFileName": ["type": "string"],
                "audioFileName": ["type": "string"],
                "isPreferred": [
                    "type": "boolean",
                    "description": "Whether the take should be marked preferred. Defaults to true.",
                ],
                "preferred": [
                    "type": "boolean",
                    "description": "Alias for isPreferred.",
                ],
                "listeningNote": [
                    "type": "string",
                    "description": "Optional listening note. Omit to keep the existing note; pass an empty string to clear it.",
                ],
                "note": [
                    "type": "string",
                    "description": "Alias for listeningNote.",
                ],
                "confirm": [
                    "type": "boolean",
                    "description": "Required when the client does not support MCP elicitation.",
                ],
            ]
        ),
        tool(
            name: "enhance_prompt",
            title: "Enhance Prompt",
            description: "Run the same prompt enhancer used by the Generate view and return the enhanced prompt without mutating app state.",
            required: ["prompt"],
            properties: [
                "prompt": [
                    "type": "string",
                    "description": "Brief prompt or audio description to enhance.",
                ],
                "mode": [
                    "type": "string",
                    "enum": ["music", "soundFX", "ambient"],
                    "description": "Generation mode. Defaults to music.",
                ],
                "generationMode": [
                    "type": "string",
                    "description": "Alias for mode.",
                ],
            ]
        ),
        tool(
            name: "write_lyrics",
            title: "Write Lyrics",
            description: "Generate cleaned, duration-aware lyrics using the same lyric writer and sanitizer as the Generate view. Requires confirm: true or MCP form approval because this may call a paid API.",
            required: ["prompt", "duration"],
            properties: [
                "prompt": [
                    "type": "string",
                    "description": "Song prompt or creative direction.",
                ],
                "style": [
                    "type": "string",
                    "description": "Optional extra style guidance.",
                ],
                "duration": [
                    "type": "number",
                    "minimum": 1,
                    "description": "Target song duration in seconds.",
                ],
                "durationSeconds": [
                    "type": "number",
                    "minimum": 1,
                    "description": "Alias for duration.",
                ],
                "confirm": [
                    "type": "boolean",
                    "description": "Required when the client does not support MCP elicitation.",
                ],
            ]
        ),
        tool(
            name: "apply_adapter",
            title: "Apply Adapter",
            description: "Select, stack, or clear ACE-Step LoRA adapters for the next generation. Updates the Generate view when it is open. Requires confirm: true or MCP form approval.",
            properties: [
                "id": [
                    "type": "string",
                    "description": "Adapter id or display name to apply as the primary adapter.",
                ],
                "adapterID": [
                    "type": "string",
                    "description": "Alias for id.",
                ],
                "name": [
                    "type": "string",
                    "description": "Alias for id.",
                ],
                "strength": [
                    "type": "number",
                    "minimum": 0,
                    "maximum": Double(AdapterStrengthLimits.maximum),
                    "description": "Primary adapter gain. 1.0 is normal strength; values above 1.0 boost weak adapters.",
                ],
                "adapters": [
                    "type": "array",
                    "description": "Optional stack of adapter objects, each with id/name and optional strength.",
                    "items": [
                        "type": "object",
                        "properties": [
                            "id": ["type": "string"],
                            "adapterID": ["type": "string"],
                            "name": ["type": "string"],
                            "strength": [
                                "type": "number",
                                "minimum": 0,
                                "maximum": Double(AdapterStrengthLimits.maximum),
                            ],
                            "weight": [
                                "type": "number",
                                "minimum": 0,
                                "maximum": Double(AdapterStrengthLimits.maximum),
                            ],
                            "scale": ["type": "number", "minimum": 0, "maximum": Double(AdapterStrengthLimits.maximum)],
                        ],
                    ],
                ],
                "clear": [
                    "type": "boolean",
                    "description": "Clear all selected adapters when true.",
                ],
                "confirm": [
                    "type": "boolean",
                    "description": "Required when the client does not support MCP elicitation.",
                ],
            ]
        ),
        tool(
            name: "generate_song",
            title: "Generate Song",
            description: "Orchestrate a song generation: optionally enhance the prompt, optionally write duration-aware lyrics, optionally apply adapters, then submit generation. Defaults to dryRun: true; actual runs require dryRun: false plus confirm: true or MCP form approval.",
            required: ["prompt"],
            properties: [
                "prompt": [
                    "type": "string",
                    "description": "Prompt or creative direction.",
                ],
                "text": [
                    "type": "string",
                    "description": "Alias for prompt.",
                ],
                "enhancePrompt": [
                    "type": "boolean",
                    "description": "Run the prompt enhancer before generation. Defaults to false.",
                ],
                "enhance": [
                    "type": "boolean",
                    "description": "Alias for enhancePrompt.",
                ],
                "writeLyrics": [
                    "type": "boolean",
                    "description": "Write duration-aware lyrics when no lyrics are supplied. Defaults to false.",
                ],
                "generateLyrics": [
                    "type": "boolean",
                    "description": "Alias for writeLyrics.",
                ],
                "lyrics": [
                    "type": "string",
                    "description": "Optional lyrics. Supplied lyrics are used as-is and prevent automatic lyric writing.",
                ],
                "style": [
                    "type": "string",
                    "description": "Optional lyric style guidance.",
                ],
                "lyricStyle": [
                    "type": "string",
                    "description": "Alias for style.",
                ],
                "adapter": [
                    "type": "string",
                    "description": "Optional adapter id or display name to apply before generation.",
                ],
                "lora": [
                    "type": "string",
                    "description": "Alias for adapter.",
                ],
                "adapterID": [
                    "type": "string",
                    "description": "Alias for adapter.",
                ],
                "strength": [
                    "type": "number",
                    "minimum": 0,
                    "maximum": Double(AdapterStrengthLimits.maximum),
                    "description": "Adapter gain when adapter/lora is supplied. 1.0 is normal strength; values above 1.0 boost weak adapters.",
                ],
                "adapters": [
                    "type": "array",
                    "description": "Optional stack of adapter objects, each with id/name and optional strength.",
                    "items": [
                        "type": "object",
                        "properties": [
                            "id": ["type": "string"],
                            "adapterID": ["type": "string"],
                            "name": ["type": "string"],
                            "strength": [
                                "type": "number",
                                "minimum": 0,
                                "maximum": Double(AdapterStrengthLimits.maximum),
                            ],
                            "weight": [
                                "type": "number",
                                "minimum": 0,
                                "maximum": Double(AdapterStrengthLimits.maximum),
                            ],
                            "scale": ["type": "number", "minimum": 0, "maximum": Double(AdapterStrengthLimits.maximum)],
                        ],
                    ],
                ],
                "clearAdapter": [
                    "type": "boolean",
                    "description": "Clear selected adapters before generation. When adapter/adapters are also supplied, the clear happens first and the requested adapter selection is then applied.",
                ],
                "duration": [
                    "type": "number",
                    "minimum": 1,
                    "description": "Duration in seconds. Defaults to 30. Stable Audio requests are capped at 47 seconds.",
                ],
                "seed": [
                    "description": "Integer seed, 'random', empty, or omitted for random.",
                ],
                "mode": [
                    "type": "string",
                    "enum": ["music", "soundFX", "ambient"],
                    "description": "Resolved generation mode. Defaults to music.",
                ],
                "pipeline": [
                    "type": "string",
                    "enum": ["auto", "aceStep", "stableAudio"],
                    "description": "Pipeline override. Defaults to auto.",
                ],
                "guidanceScale": [
                    "type": "number",
                    "description": "CFG guidance scale. Defaults to 7.5.",
                ],
                "guidancePreset": [
                    "type": "string",
                    "enum": ["constant", "lateEmphasis", "midToLate", "softLate"],
                    "description": "Guidance schedule preset. Defaults to softLate for non-XL ACE-Step models and constant for XL DiT / XL adapters.",
                ],
                "method": [
                    "type": "string",
                    "enum": ["ode", "sde"],
                    "description": "Diffusion method. Defaults to ode.",
                ],
                "precision": [
                    "type": "string",
                    "enum": ["float32", "float16"],
                    "description": "Runtime precision. Defaults to float32.",
                ],
                "steps": [
                    "description": "Positive integer step override, 'auto', or omitted for auto.",
                ],
                "shift": [
                    "type": "number",
                    "description": "ACE-Step shift override. Defaults to 3.0.",
                ],
                "dryRun": [
                    "type": "boolean",
                    "description": "Defaults to true. Set false, with confirmation, to start generation.",
                ],
                "confirm": [
                    "type": "boolean",
                    "description": "Required for actual generation when the client does not support MCP elicitation.",
                ],
            ]
        ),
        tool(
            name: "generate",
            title: "Generate Audio",
            description: "Submit a generation request through the same path as the Generate view. Defaults to dryRun: true; actual runs require dryRun: false plus confirm: true or MCP form approval. Returns immediately; poll generation_status or history for completion.",
            required: ["prompt"],
            properties: [
                "prompt": [
                    "type": "string",
                    "description": "Prompt or audio description.",
                ],
                "text": [
                    "type": "string",
                    "description": "Alias for prompt.",
                ],
                "lyrics": [
                    "type": "string",
                    "description": "Optional lyrics for music generation.",
                ],
                "duration": [
                    "type": "number",
                    "minimum": 1,
                    "description": "Duration in seconds. Defaults to 30. Stable Audio requests are capped at 47 seconds.",
                ],
                "seed": [
                    "description": "Integer seed, 'random', empty, or omitted for random.",
                ],
                "mode": [
                    "type": "string",
                    "enum": ["music", "soundFX", "ambient"],
                    "description": "Resolved generation mode. Defaults to music.",
                ],
                "pipeline": [
                    "type": "string",
                    "enum": ["auto", "aceStep", "stableAudio"],
                    "description": "Pipeline override. Defaults to auto.",
                ],
                "guidanceScale": [
                    "type": "number",
                    "description": "CFG guidance scale. Defaults to 7.5.",
                ],
                "guidancePreset": [
                    "type": "string",
                    "enum": ["constant", "lateEmphasis", "midToLate", "softLate"],
                    "description": "Guidance schedule preset. Defaults to softLate for non-XL ACE-Step models and constant for XL DiT / XL adapters.",
                ],
                "method": [
                    "type": "string",
                    "enum": ["ode", "sde"],
                    "description": "Diffusion method. Defaults to ode.",
                ],
                "precision": [
                    "type": "string",
                    "enum": ["float32", "float16"],
                    "description": "Runtime precision. Defaults to float32.",
                ],
                "steps": [
                    "description": "Positive integer step override, 'auto', or omitted for auto.",
                ],
                "shift": [
                    "type": "number",
                    "description": "ACE-Step shift override. Defaults to 3.0.",
                ],
                "referenceAudioPath": [
                    "type": "string",
                    "description": "Optional local audio path for cover-song generation. If the client advertises MCP roots, prefer a file under pimpaudio://client-roots.",
                ],
                "coverStrength": [
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Cover-song rework amount. 0 stays closest to the source recording; 1 follows coarse acoustic hints with the most creative freedom. Defaults to 0.7.",
                ],
                "dryRun": [
                    "type": "boolean",
                    "description": "Defaults to true. Set false, with confirmation, to start generation.",
                ],
                "confirm": [
                    "type": "boolean",
                    "description": "Required for actual generation when the client does not support MCP elicitation.",
                ],
            ]
        ),
        tool(
            name: "generation_status",
            title: "Generation Status",
            description: "Return the current generation state, latest completed result metadata, and current MCP operation when present.",
            properties: [:]
        ),
        tool(
            name: "editor_status",
            title: "Editor Status",
            description: "Return editor readiness, loaded audio metadata, supported editor actions, and the active editor operation when present.",
            properties: [:]
        ),
        tool(
            name: "load_editor_audio",
            title: "Load Editor Audio",
            description: "Load a history audio id or readable local audio path into the editor coordinator as a background operation. Defaults to dryRun: true; actual loads require dryRun: false plus confirm: true or MCP form approval. Returns an operationID for polling. Prefer history/audio ids; when using paths, prefer files under advertised MCP client roots.",
            properties: [
                "id": [
                    "type": "string",
                    "description": "History entry id or audio resource id. Prefer this over path for sandbox-safe access.",
                ],
                "historyID": [
                    "type": "string",
                    "description": "History entry id to load into the editor.",
                ],
                "historyId": [
                    "type": "string",
                    "description": "Alias for historyID.",
                ],
                "audioID": [
                    "type": "string",
                    "description": "Audio resource id to load into the editor.",
                ],
                "audioId": [
                    "type": "string",
                    "description": "Alias for audioID.",
                ],
                "path": [
                    "type": "string",
                    "description": "Readable local audio file path. History/audio ids are preferred; if MCP client roots are advertised, choose a file under pimpaudio://client-roots. Sandboxed builds may still reject arbitrary paths.",
                ],
                "audioPath": [
                    "type": "string",
                    "description": "Alias for path.",
                ],
                "filePath": [
                    "type": "string",
                    "description": "Alias for path.",
                ],
                "originalPrompt": [
                    "type": "string",
                    "description": "Optional source prompt/context associated with the audio.",
                ],
                "prompt": [
                    "type": "string",
                    "description": "Alias for originalPrompt.",
                ],
                "dryRun": [
                    "type": "boolean",
                    "description": "Defaults to true. Set false, with confirmation, to load audio.",
                ],
                "validateOnly": [
                    "type": "boolean",
                    "description": "Alias for dryRun.",
                ],
                "confirm": [
                    "type": "boolean",
                    "description": "Required for an actual load when the client does not support MCP elicitation.",
                ],
            ]
        ),
        tool(
            name: "submit_editor_inpaint",
            title: "Submit Editor Inpaint",
            description: "Submit a confirmed editor inpaint/repaint request against the loaded editor audio as a background operation. Use dryRun to validate without mutating audio.",
            properties: [
                "mode": [
                    "type": "string",
                    "enum": ["inpaint", "repaint"],
                    "description": "Editor edit mode. Defaults to inpaint.",
                ],
                "editMode": [
                    "type": "string",
                    "enum": ["inpaint", "repaint"],
                    "description": "Alias for mode.",
                ],
                "regionPrompt": [
                    "type": "string",
                    "description": "Prompt describing the replacement content for the selected region.",
                ],
                "prompt": [
                    "type": "string",
                    "description": "Alias for regionPrompt.",
                ],
                "originalPrompt": [
                    "type": "string",
                    "description": "Optional source/full-track context prompt. Defaults to loaded audio context when available.",
                ],
                "sourcePrompt": [
                    "type": "string",
                    "description": "Alias for originalPrompt.",
                ],
                "selection": [
                    "type": "object",
                    "additionalProperties": false,
                    "description": "Selected time range and optional frequency band.",
                    "properties": [
                        "startTime": ["type": "number"],
                        "endTime": ["type": "number"],
                        "start": ["type": "number"],
                        "end": ["type": "number"],
                        "frequencyBand": [
                            "type": "object",
                            "additionalProperties": false,
                            "properties": [
                                "lowCutHz": ["type": "number"],
                                "highCutHz": ["type": "number"],
                                "lowHz": ["type": "number"],
                                "highHz": ["type": "number"],
                                "low": ["type": "number"],
                                "high": ["type": "number"],
                            ],
                        ],
                    ],
                ],
                "selectionStart": [
                    "type": "number",
                    "description": "Selected region start time in seconds.",
                ],
                "selectionEnd": [
                    "type": "number",
                    "description": "Selected region end time in seconds.",
                ],
                "startTime": [
                    "type": "number",
                    "description": "Alias for selectionStart.",
                ],
                "endTime": [
                    "type": "number",
                    "description": "Alias for selectionEnd.",
                ],
                "start": [
                    "type": "number",
                    "description": "Alias for selectionStart.",
                ],
                "end": [
                    "type": "number",
                    "description": "Alias for selectionEnd.",
                ],
                "frequencyBand": [
                    "type": "object",
                    "additionalProperties": false,
                    "description": "Optional frequency-limited edit band in Hz.",
                    "properties": [
                        "lowCutHz": ["type": "number"],
                        "highCutHz": ["type": "number"],
                        "lowHz": ["type": "number"],
                        "highHz": ["type": "number"],
                        "low": ["type": "number"],
                        "high": ["type": "number"],
                    ],
                ],
                "lowCutHz": [
                    "type": "number",
                    "description": "Optional low frequency bound in Hz.",
                ],
                "highCutHz": [
                    "type": "number",
                    "description": "Optional high frequency bound in Hz.",
                ],
                "lowHz": [
                    "type": "number",
                    "description": "Alias for lowCutHz.",
                ],
                "highHz": [
                    "type": "number",
                    "description": "Alias for highCutHz.",
                ],
                "low": [
                    "type": "number",
                    "description": "Alias for lowCutHz.",
                ],
                "high": [
                    "type": "number",
                    "description": "Alias for highCutHz.",
                ],
                "preset": [
                    "type": "string",
                    "enum": ["subtle", "moderate", "full"],
                    "description": "Optional strength/guidance preset.",
                ],
                "strength": [
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Denoising strength from 0.0 to 1.0. Defaults to 0.6.",
                ],
                "denoiseStrength": [
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Alias for strength.",
                ],
                "editStrength": [
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Alias for strength.",
                ],
                "guidanceScale": [
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "description": "Positive classifier-free guidance scale. Defaults to 7.0.",
                ],
                "guidance": [
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "description": "Alias for guidanceScale.",
                ],
                "pipeline": [
                    "type": "string",
                    "enum": ["auto", "aceStep", "stableAudio"],
                    "description": "Generation pipeline to use for the edit. Defaults to auto.",
                ],
                "pipelineOverride": [
                    "type": "string",
                    "enum": ["auto", "aceStep", "stableAudio"],
                    "description": "Alias for pipeline.",
                ],
                "seed": [
                    "type": ["integer", "string", "null"],
                    "description": "Optional deterministic seed, or 'random'.",
                ],
                "confirm": [
                    "type": "boolean",
                    "description": "Set true after explicit approval. If omitted, clients advertising MCP form elicitation may be asked to approve through elicitation/create.",
                ],
                "dryRun": [
                    "type": "boolean",
                    "description": "Validate and echo the resolved request without mutating editor audio.",
                ],
                "validateOnly": [
                    "type": "boolean",
                    "description": "Alias for dryRun.",
                ],
            ]
        ),
        tool(
            name: "apply_editor_effect_chain",
            title: "Apply Editor Effect Chain",
            description: "Apply one or more confirmed DSP effects to the loaded editor audio as a background operation. Use dryRun to validate without mutating audio.",
            properties: [
                "effect": [
                    "type": "string",
                    "enum": [
                        "trimToSelection",
                        "deleteSelection",
                        "fadeIn",
                        "fadeOut",
                        "normalize",
                        "eq",
                        "reverb",
                        "timeStretch",
                        "pitchShift",
                    ],
                    "description": "Single effect to apply when effects is omitted.",
                ],
                "name": [
                    "type": "string",
                    "description": "Alias for effect.",
                ],
                "effects": [
                    "type": "array",
                    "description": "Optional ordered effect chain. Items can be effect names or objects with effect, target, selection, and parameters.",
                    "items": [
                        "oneOf": [
                            [
                                "type": "string",
                                "enum": [
                                    "trimToSelection",
                                    "deleteSelection",
                                    "fadeIn",
                                    "fadeOut",
                                    "normalize",
                                    "eq",
                                    "reverb",
                                    "timeStretch",
                                    "pitchShift",
                                ],
                            ],
                            [
                                "type": "object",
                                "additionalProperties": true,
                                "properties": [
                                    "effect": ["type": "string"],
                                    "name": ["type": "string"],
                                    "target": [
                                        "type": "string",
                                        "enum": ["wholeTrack", "selection", "whole", "region"],
                                    ],
                                    "selection": [
                                        "type": "object",
                                        "additionalProperties": false,
                                        "properties": [
                                            "startTime": ["type": "number"],
                                            "endTime": ["type": "number"],
                                            "start": ["type": "number"],
                                            "end": ["type": "number"],
                                        ],
                                    ],
                                    "parameters": [
                                        "type": "object",
                                        "additionalProperties": true,
                                    ],
                                ],
                            ],
                        ],
                    ],
                ],
                "effectChain": [
                    "type": "array",
                    "description": "Alias for effects.",
                ],
                "chain": [
                    "type": "array",
                    "description": "Alias for effects.",
                ],
                "target": [
                    "type": "string",
                    "enum": ["wholeTrack", "selection", "whole", "region"],
                    "description": "Default target for effects. Selection-only effects require a selected range.",
                ],
                "selection": [
                    "type": "object",
                    "additionalProperties": false,
                    "description": "Default selected time range for effects that target a region.",
                    "properties": [
                        "startTime": ["type": "number"],
                        "endTime": ["type": "number"],
                        "start": ["type": "number"],
                        "end": ["type": "number"],
                    ],
                ],
                "selectionStart": [
                    "type": "number",
                    "description": "Default selected region start time in seconds.",
                ],
                "selectionEnd": [
                    "type": "number",
                    "description": "Default selected region end time in seconds.",
                ],
                "startTime": [
                    "type": "number",
                    "description": "Alias for selectionStart.",
                ],
                "endTime": [
                    "type": "number",
                    "description": "Alias for selectionEnd.",
                ],
                "start": [
                    "type": "number",
                    "description": "Alias for selectionStart.",
                ],
                "end": [
                    "type": "number",
                    "description": "Alias for selectionEnd.",
                ],
                "parameters": [
                    "type": "object",
                    "additionalProperties": true,
                    "description": "Effect parameters for single-effect calls. Use bands for eq, preset or reverb values for reverb, rate for timeStretch, and pitch/cents/semitones for pitchShift.",
                ],
                "params": [
                    "type": "object",
                    "additionalProperties": true,
                    "description": "Alias for parameters.",
                ],
                "bands": [
                    "type": "array",
                    "description": "EQ band array. Each band accepts frequency, gain, and q.",
                    "items": [
                        "type": "object",
                        "additionalProperties": false,
                        "properties": [
                            "frequency": ["type": "number"],
                            "freq": ["type": "number"],
                            "hz": ["type": "number"],
                            "gain": ["type": "number"],
                            "gainDB": ["type": "number"],
                            "db": ["type": "number"],
                            "q": ["type": "number"],
                            "quality": ["type": "number"],
                        ],
                    ],
                ],
                "preset": [
                    "type": "string",
                    "enum": ["room", "hall", "plate"],
                    "description": "Reverb preset.",
                ],
                "decayTime": [
                    "type": "number",
                    "description": "Reverb decay time in seconds.",
                ],
                "wetDryMix": [
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Reverb wet/dry mix from 0.0 to 1.0.",
                ],
                "preDelay": [
                    "type": "number",
                    "description": "Reverb pre-delay in milliseconds.",
                ],
                "damping": [
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Reverb damping from 0.0 to 1.0.",
                ],
                "rate": [
                    "type": "number",
                    "minimum": 0.25,
                    "maximum": 4,
                    "description": "Time-stretch rate. 1.0 keeps the original duration.",
                ],
                "ratio": [
                    "type": "number",
                    "minimum": 0.25,
                    "maximum": 4,
                    "description": "Alias for rate.",
                ],
                "pitch": [
                    "type": "number",
                    "minimum": -2400,
                    "maximum": 2400,
                    "description": "Pitch shift in cents.",
                ],
                "cents": [
                    "type": "number",
                    "minimum": -2400,
                    "maximum": 2400,
                    "description": "Alias for pitch.",
                ],
                "semitones": [
                    "type": "number",
                    "minimum": -24,
                    "maximum": 24,
                    "description": "Pitch shift in semitones.",
                ],
                "confirm": [
                    "type": "boolean",
                    "description": "Set true after explicit approval. If omitted, clients advertising MCP form elicitation may be asked to approve through elicitation/create.",
                ],
                "dryRun": [
                    "type": "boolean",
                    "description": "Validate and echo the resolved DSP request without mutating editor audio.",
                ],
                "validateOnly": [
                    "type": "boolean",
                    "description": "Alias for dryRun.",
                ],
            ]
        ),
        tool(
            name: "render_editor_audio",
            title: "Render Editor Audio",
            description: "Render the loaded editor buffer or a selected time range to a WAV/AIFF file as a background operation. Defaults to dryRun: true; actual renders require dryRun: false plus confirm: true or MCP form approval.",
            properties: [
                "format": [
                    "type": "string",
                    "enum": ["wav", "aiff"],
                    "description": "Lossless output format. Defaults to wav.",
                ],
                "fileFormat": [
                    "type": "string",
                    "enum": ["wav", "aiff"],
                    "description": "Alias for format.",
                ],
                "outputPath": [
                    "type": "string",
                    "description": "Optional output file path. The extension is normalized to the requested format. Prefer the default MCPExports path or a path under advertised MCP client roots; sandboxed builds may reject arbitrary paths.",
                ],
                "path": [
                    "type": "string",
                    "description": "Alias for outputPath.",
                ],
                "outputDirectory": [
                    "type": "string",
                    "description": "Optional output directory. Defaults to MCPExports beside the loaded source audio. If MCP client roots are advertised, prefer a directory under pimpaudio://client-roots.",
                ],
                "directory": [
                    "type": "string",
                    "description": "Alias for outputDirectory.",
                ],
                "fileName": [
                    "type": "string",
                    "description": "Optional file name when outputDirectory is used.",
                ],
                "selection": [
                    "type": "object",
                    "additionalProperties": false,
                    "description": "Optional selected time range to render. Omit to render the full editor buffer.",
                    "properties": [
                        "startTime": ["type": "number"],
                        "endTime": ["type": "number"],
                        "start": ["type": "number"],
                        "end": ["type": "number"],
                    ],
                ],
                "selectionStart": [
                    "type": "number",
                    "description": "Optional selected region start time in seconds.",
                ],
                "selectionEnd": [
                    "type": "number",
                    "description": "Optional selected region end time in seconds.",
                ],
                "startTime": [
                    "type": "number",
                    "description": "Alias for selectionStart.",
                ],
                "endTime": [
                    "type": "number",
                    "description": "Alias for selectionEnd.",
                ],
                "start": [
                    "type": "number",
                    "description": "Alias for selectionStart.",
                ],
                "end": [
                    "type": "number",
                    "description": "Alias for selectionEnd.",
                ],
                "overwrite": [
                    "type": "boolean",
                    "description": "Allow replacing an existing output file. Defaults to false.",
                ],
                "dryRun": [
                    "type": "boolean",
                    "description": "Defaults to true. Set false, with confirmation, to write the file.",
                ],
                "validateOnly": [
                    "type": "boolean",
                    "description": "Alias for dryRun.",
                ],
                "confirm": [
                    "type": "boolean",
                    "description": "Required for an actual render when the client does not support MCP elicitation.",
                ],
            ]
        ),
        tool(
            name: "list_operations",
            title: "List Operations",
            description: "Return recent MCP-observable operations with status, progress, result, error, and cancellation metadata.",
            properties: [
                "limit": [
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "description": "Maximum number of operation records to return. Defaults to 20.",
                ],
            ]
        ),
        tool(
            name: "get_operation",
            title: "Get Operation",
            description: "Return one MCP-observable operation by id.",
            properties: [
                "id": [
                    "type": "string",
                    "description": "Operation id returned by generate, generate_song, load_editor_audio, submit_editor_inpaint, apply_editor_effect_chain, render_editor_audio, list_operations, editor_status, or generation_status.",
                ],
                "operationID": [
                    "type": "string",
                    "description": "Alias for id.",
                ],
                "operationId": [
                    "type": "string",
                    "description": "Alias for id.",
                ],
            ]
        ),
        tool(
            name: "cancel_operation",
            title: "Cancel Operation",
            description: "Cancel a running MCP-observable operation by id. Generation and editor background operations use the same operation registry. Requires confirm: true or MCP form approval.",
            properties: [
                "id": [
                    "type": "string",
                    "description": "Operation id returned by generate, generate_song, load_editor_audio, submit_editor_inpaint, apply_editor_effect_chain, render_editor_audio, list_operations, editor_status, or generation_status.",
                ],
                "operationID": [
                    "type": "string",
                    "description": "Alias for id.",
                ],
                "operationId": [
                    "type": "string",
                    "description": "Alias for id.",
                ],
                "reason": [
                    "type": "string",
                    "description": "Optional human-readable cancellation reason.",
                ],
                "confirm": [
                    "type": "boolean",
                    "description": "Required when the client does not support MCP elicitation.",
                ],
            ]
        ),

        tool(
            name: "create_remote_training_job",
            title: "Create Remote Training Job",
            description: "Validate or export a remote training job zip from a reviewed dataset package. Defaults to dryRun: true; actual export requires dryRun: false plus confirm: true or MCP form approval.",
            properties: [
                "path": [
                    "type": "string",
                    "description": "Dataset package path.",
                ],
                "dataset": [
                    "type": "string",
                    "description": "Dataset id, folder name, or partial name.",
                ],
                "id": [
                    "type": "string",
                    "description": "Alias for dataset.",
                ],
                "kind": [
                    "type": "string",
                    "enum": ["lora", "lokr"],
                    "description": "Adapter kind. Defaults to lora.",
                ],
                "recipe": [
                    "type": "string",
                    "enum": ["narrow", "wide"],
                    "description": "Remote recipe. Defaults to wide.",
                ],
                "bundle": [
                    "type": "string",
                    "enum": ["full_mix", "vocals", "instrumental"],
                    "description": "Training bundle. Defaults to full_mix.",
                ],
                "baseModel": [
                    "type": "string",
                    "description": "xlSFT, xlBase, or xlTurbo. Defaults to xlSFT.",
                ],
                "adapterName": [
                    "type": "string",
                    "description": "Friendly adapter name for the remote job.",
                ],
                "epochs": [
                    "type": "integer",
                    "minimum": 1,
                    "description": "Training epochs.",
                ],
                "saveEvery": [
                    "type": "integer",
                    "minimum": 1,
                    "description": "Checkpoint save interval.",
                ],
                "seed": [
                    "type": "integer",
                    "description": "Training seed. Defaults to 42.",
                ],
                "destinationPath": [
                    "type": "string",
                    "description": "Output zip path or parent directory.",
                ],
                "dryRun": [
                    "type": "boolean",
                    "description": "Defaults to true.",
                ],
                "confirm": [
                    "type": "boolean",
                    "description": "Required for actual export without elicitation.",
                ],
            ]
        ),
        tool(
            name: "upload_remote_training_job",
            title: "Upload Remote Training Job",
            description: "Upload a remote job zip and portable worker to a RunPod pod over SSH. Requires confirm: true or MCP form approval. SSH key must already be configured in the app or passed as sshKeyPath.",
            required: ["podID"],
            properties: [
                "podID": [
                    "type": "string",
                    "description": "RunPod pod id.",
                ],
                "archivePath": [
                    "type": "string",
                    "description": "Local remote job zip path. Defaults to last created archive.",
                ],
                "sshKeyPath": [
                    "type": "string",
                    "description": "Optional private key path override.",
                ],
                "confirm": [
                    "type": "boolean",
                    "description": "Required without elicitation.",
                ],
            ]
        ),
        tool(
            name: "start_remote_training_job",
            title: "Start Remote Training Job",
            description: "Start detached remote worker setup/training on a RunPod pod. Requires confirm: true or MCP form approval.",
            required: ["podID"],
            properties: [
                "podID": [
                    "type": "string",
                    "description": "RunPod pod id.",
                ],
                "archivePath": [
                    "type": "string",
                    "description": "Local remote job zip path.",
                ],
                "sshKeyPath": [
                    "type": "string",
                    "description": "Optional private key path override.",
                ],
                "confirm": [
                    "type": "boolean",
                    "description": "Required without elicitation.",
                ],
            ]
        ),
        tool(
            name: "remote_training_job_status",
            title: "Remote Training Job Status",
            description: "Return remote worker job phase/logs for a pod and optional archive. Defaults to refreshing/monitoring when podID and archivePath are available.",
            properties: [
                "podID": [
                    "type": "string",
                    "description": "RunPod pod id.",
                ],
                "archivePath": [
                    "type": "string",
                    "description": "Local remote job zip path.",
                ],
                "sshKeyPath": [
                    "type": "string",
                    "description": "Optional private key path override.",
                ],
                "refresh": [
                    "type": "boolean",
                    "description": "Refresh/monitor remote job. Defaults to true when pod and archive are present.",
                ],
                "refreshPods": [
                    "type": "boolean",
                    "description": "Also refresh pod list.",
                ],
            ]
        ),
        tool(
            name: "pull_remote_training_job",
            title: "Pull Remote Training Job",
            description: "Download remote_artifacts from a pod and optionally import the final adapter. Requires confirm: true or MCP form approval.",
            required: ["podID"],
            properties: [
                "podID": [
                    "type": "string",
                    "description": "RunPod pod id.",
                ],
                "archivePath": [
                    "type": "string",
                    "description": "Local remote job zip path.",
                ],
                "sshKeyPath": [
                    "type": "string",
                    "description": "Optional private key path override.",
                ],
                "destinationDirectory": [
                    "type": "string",
                    "description": "Parent directory for pullback. Defaults beside the job zip.",
                ],
                "importAdapter": [
                    "type": "boolean",
                    "description": "Import final_adapter locally. Defaults to true.",
                ],
                "confirm": [
                    "type": "boolean",
                    "description": "Required without elicitation.",
                ],
            ]
        ),
        tool(
            name: "build_training_dataset",
            title: "Build Training Dataset",
            description: "Build a local dataset package from source audio files/directories. Defaults to dryRun: true; actual builds require dryRun: false plus confirm: true and return an operationID.",
            properties: [
                "paths": [
                    "type": "array",
                    "items": ["type": "string"],
                    "description": "Source audio file paths.",
                ],
                "directories": [
                    "type": "array",
                    "items": ["type": "string"],
                    "description": "Source directories to scan.",
                ],
                "buildMode": [
                    "type": "string",
                    "enum": ["whole_tracks", "clip_extraction"],
                    "description": "Defaults to whole_tracks.",
                ],
                "datasetName": [
                    "type": "string",
                    "description": "Optional dataset name.",
                ],
                "outputDirectory": [
                    "type": "string",
                    "description": "Output base directory.",
                ],
                "clipLength": [
                    "type": "number",
                    "description": "Clip length seconds for extract mode.",
                ],
                "clipsPerTrack": [
                    "type": "integer",
                    "description": "Clips per source in extract mode.",
                ],
                "maxTotalClips": [
                    "type": "integer",
                    "description": "Max total clips.",
                ],
                "samplingMode": [
                    "type": "string",
                    "description": "evenlySpread, sequential, or randomSeeded.",
                ],
                "samplingSeed": [
                    "type": "integer",
                    "description": "Seed for random sampling.",
                ],
                "dryRun": [
                    "type": "boolean",
                    "description": "Defaults to true.",
                ],
                "confirm": [
                    "type": "boolean",
                    "description": "Required for actual build.",
                ],
            ]
        ),
        tool(
            name: "enrich_training_dataset",
            title: "Enrich Training Dataset",
            description: "Run dataset enrichment (captions/lyrics/stems). Defaults to dryRun: true; actual runs require dryRun: false plus confirm: true and return an operationID.",
            properties: [
                "path": [
                    "type": "string",
                    "description": "Dataset package path.",
                ],
                "dataset": [
                    "type": "string",
                    "description": "Dataset id or name.",
                ],
                "backend": [
                    "type": "string",
                    "enum": ["automatic", "lightweight", "native_ace"],
                    "description": "Annotation backend.",
                ],
                "createStems": [
                    "type": "boolean",
                    "description": "Create vocal/instrumental stems.",
                ],
                "force": [
                    "type": "boolean",
                    "description": "Re-run even when cached.",
                ],
                "dryRun": [
                    "type": "boolean",
                    "description": "Defaults to true.",
                ],
                "confirm": [
                    "type": "boolean",
                    "description": "Required for actual enrichment.",
                ],
            ]
        ),
        tool(
            name: "update_training_dataset_review",
            title: "Update Training Dataset Review",
            description: "Update captions/lyrics/include/approve fields for a dataset clip or approve all included assets. Requires confirm: true or MCP form approval.",
            properties: [
                "path": [
                    "type": "string",
                    "description": "Dataset package path.",
                ],
                "dataset": [
                    "type": "string",
                    "description": "Dataset id or name.",
                ],
                "clipID": [
                    "type": "string",
                    "description": "Clip/asset id or filename.",
                ],
                "caption": [
                    "type": "string",
                    "description": "Manual caption.",
                ],
                "lyrics": [
                    "type": "string",
                    "description": "Manual lyrics.",
                ],
                "includeInTraining": [
                    "type": "boolean",
                    "description": "Include clip in training plan.",
                ],
                "approved": [
                    "type": "boolean",
                    "description": "Mark clip approved.",
                ],
                "notes": [
                    "type": "string",
                    "description": "Review notes.",
                ],
                "approveAllIncluded": [
                    "type": "boolean",
                    "description": "Approve every included clip.",
                ],
                "confirm": [
                    "type": "boolean",
                    "description": "Required without elicitation.",
                ],
            ]
        ),
        tool(
            name: "import_adapter",
            title: "Import Adapter",
            description: "Import a local native/PEFT/Side-Step adapter folder into the registry. Requires confirm: true or MCP form approval.",
            required: ["path"],
            properties: [
                "path": [
                    "type": "string",
                    "description": "Adapter folder path.",
                ],
                "name": [
                    "type": "string",
                    "description": "Optional preferred adapter name.",
                ],
                "confirm": [
                    "type": "boolean",
                    "description": "Required without elicitation.",
                ],
            ]
        ),
        tool(
            name: "delete_adapter",
            title: "Delete Adapter",
            description: "Delete an installed LoRA/LoKR adapter. Requires confirm: true or MCP form approval.",
            required: ["id"],
            properties: [
                "id": [
                    "type": "string",
                    "description": "Adapter id or name.",
                ],
                "confirm": [
                    "type": "boolean",
                    "description": "Required without elicitation.",
                ],
            ]
        ),
        tool(
            name: "merge_adapters",
            title: "Merge Adapters",
            description: "Merge two LoRA adapters into a new adapter. Requires confirm: true or MCP form approval.",
            required: ["adapterA", "adapterB"],
            properties: [
                "adapterA": [
                    "type": "string",
                    "description": "First adapter id/name.",
                ],
                "adapterB": [
                    "type": "string",
                    "description": "Second adapter id/name.",
                ],
                "alpha": [
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Blend weight on A. Defaults to 0.5.",
                ],
                "name": [
                    "type": "string",
                    "description": "Merged adapter name.",
                ],
                "confirm": [
                    "type": "boolean",
                    "description": "Required without elicitation.",
                ],
            ]
        ),
        tool(
            name: "download_model",
            title: "Download Model",
            description: "Download a registry model. Defaults to dryRun: true; actual download requires dryRun: false plus confirm: true and returns an operationID.",
            required: ["id"],
            properties: [
                "id": [
                    "type": "string",
                    "description": "Model id or display name.",
                ],
                "dryRun": [
                    "type": "boolean",
                    "description": "Defaults to true.",
                ],
                "confirm": [
                    "type": "boolean",
                    "description": "Required for actual download.",
                ],
            ]
        ),
        tool(
            name: "set_active_model",
            title: "Set Active Model",
            description: "Activate a ready model for its category. Requires confirm: true or MCP form approval.",
            required: ["id"],
            properties: [
                "id": [
                    "type": "string",
                    "description": "Model id or display name.",
                ],
                "confirm": [
                    "type": "boolean",
                    "description": "Required without elicitation.",
                ],
            ]
        ),
        tool(
            name: "delete_history_entries",
            title: "Delete History Entries",
            description: "Delete selected history takes or all history. Requires confirm: true or MCP form approval.",
            properties: [
                "ids": [
                    "type": "array",
                    "items": ["type": "string"],
                    "description": "History entry UUIDs or filenames.",
                ],
                "deleteAll": [
                    "type": "boolean",
                    "description": "Delete every history entry.",
                ],
                "confirm": [
                    "type": "boolean",
                    "description": "Required without elicitation.",
                ],
            ]
        ),
        tool(
            name: "save_generation_preset",
            title: "Save Generation Preset",
            description: "Save a named generation preset. Requires confirm: true or MCP form approval.",
            required: ["name"],
            properties: [
                "name": [
                    "type": "string",
                    "description": "Preset name.",
                ],
                "prompt": [
                    "type": "string",
                    "description": "Prompt text.",
                ],
                "lyrics": [
                    "type": "string",
                    "description": "Lyrics text.",
                ],
                "duration": [
                    "type": "number",
                    "description": "Duration seconds.",
                ],
                "seed": [
                    "type": "string",
                    "description": "Seed text.",
                ],
                "guidanceScale": [
                    "type": "number",
                    "description": "Guidance scale.",
                ],
                "guidancePreset": [
                    "type": "string",
                    "description": "Guidance preset id.",
                ],
                "mode": [
                    "type": "string",
                    "description": "music, soundFX, or ambient.",
                ],
                "pipeline": [
                    "type": "string",
                    "description": "auto, aceStep, or stableAudio.",
                ],
                "adapterID": [
                    "type": "string",
                    "description": "Optional adapter id.",
                ],
                "coverStrength": [
                    "type": "number",
                    "description": "Cover strength.",
                ],
                "confirm": [
                    "type": "boolean",
                    "description": "Required without elicitation.",
                ],
            ]
        ),
        tool(
            name: "delete_generation_preset",
            title: "Delete Generation Preset",
            description: "Delete a saved generation preset by id or name. Requires confirm: true or MCP form approval.",
            required: ["id"],
            properties: [
                "id": [
                    "type": "string",
                    "description": "Preset id or name.",
                ],
                "confirm": [
                    "type": "boolean",
                    "description": "Required without elicitation.",
                ],
            ]
        ),
        tool(
            name: "set_verbose_diagnostics",
            title: "Set Verbose Diagnostics",
            description: "Enable or disable verbose audio diagnostics logging.",
            required: ["enabled"],
            properties: [
                "enabled": [
                    "type": "boolean",
                    "description": "true to enable AUDIODIAG file logging.",
                ],
            ]
        ),
        tool(
            name: "tail_diagnostics_log",
            title: "Tail Diagnostics Log",
            description: "Return recent AUDIODIAG lines. Unlike tail_training_log, defaults to unfiltered lines.",
            properties: [
                "limit": [
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 500,
                    "description": "Max lines. Defaults to 120.",
                ],
                "contains": [
                    "type": "string",
                    "description": "Optional case-insensitive filter.",
                ],
                "fileName": [
                    "type": "string",
                    "description": "Optional diagnostics log file name.",
                ],
                "path": [
                    "type": "string",
                    "description": "Optional full diagnostics log path.",
                ],
            ]
        ),
        tool(
            name: "cancel_generation",
            title: "Cancel Generation",
            description: "Cancel the active generation run through the same path as the Generate Cancel button. Requires confirm: true or MCP form approval.",
            properties: [
                "reason": [
                    "type": "string",
                    "description": "Optional cancellation reason.",
                ],
                "confirm": [
                    "type": "boolean",
                    "description": "Required when the client does not support MCP elicitation.",
                ],
            ]
        ),
        tool(
            name: "run_adapter_audition",
            title: "Run Adapter Audition",
            description: "Render fixed-seed base/gain audition snapshots for an installed adapter. Requires confirm: true or MCP form approval and returns an operationID.",
            required: ["id"],
            properties: [
                "id": [
                    "type": "string",
                    "description": "Adapter id or name.",
                ],
                "seed": [
                    "type": "integer",
                    "description": "Fixed seed. Defaults to 42.",
                ],
                "duration": [
                    "type": "number",
                    "description": "Snapshot duration seconds. Defaults to 30.",
                ],
                "strengths": [
                    "type": "array",
                    "items": ["type": "string"],
                    "description": "Gain ladder. Defaults to 1.0, 1.6, 2.0.",
                ],
                "includeBase": [
                    "type": "boolean",
                    "description": "Include base generation. Defaults to true.",
                ],
                "prompt": [
                    "type": "string",
                    "description": "Optional audition prompt.",
                ],
                "lyrics": [
                    "type": "string",
                    "description": "Optional lyrics.",
                ],
                "confirm": [
                    "type": "boolean",
                    "description": "Required without elicitation.",
                ],
            ]
        ),
        tool(
            name: "release_memory",
            title: "Release Memory",
            description: "Cancel active generation if needed, unload model weights, and flush MLX GPU cache. Requires confirm: true or MCP form elicitation approval.",
            properties: [
                "confirm": [
                    "type": "boolean",
                    "description": "Set true after explicit approval. If omitted, clients advertising MCP form elicitation may be asked to approve through elicitation/create.",
                ],
            ]
        ),
    ]

    private static func trainingControlProperties(includeStartControls: Bool) -> [String: Any] {
        var properties: [String: Any] = [
            "path": [
                "type": "string",
                "description": "Full path to a dataset package containing captions.json. Defaults to the newest visible dataset.",
            ],
            "directory": [
                "type": "string",
                "description": "Alias for path.",
            ],
            "datasetPath": [
                "type": "string",
                "description": "Alias for path.",
            ],
            "dataset": [
                "type": "string",
                "description": "Dataset id, folder name, or partial dataset name. Defaults to the newest visible dataset.",
            ],
            "id": [
                "type": "string",
                "description": "Alias for dataset when identifying a dataset package.",
            ],
            "kind": [
                "type": "string",
                "enum": ["lora", "lokr"],
                "description": "Adapter type to train. Defaults to lora.",
            ],
            "adapterKind": [
                "type": "string",
                "enum": ["lora", "lokr"],
                "description": "Alias for kind.",
            ],
            "mode": [
                "type": "string",
                "enum": ["lora", "lokr"],
                "description": "Alias for kind.",
            ],
            "recipe": [
                "type": "string",
                "enum": ["narrow", "wide"],
                "description": "Configuration preset. LoRA and LoKR support narrow and wide.",
            ],
            "preset": [
                "type": "string",
                "description": "Alias for recipe.",
            ],
            "adapterName": [
                "type": "string",
                "description": "Name for the adapter if training is started.",
            ],
            "name": [
                "type": "string",
                "description": "Alias for adapterName when no dataset id/name/path is provided.",
            ],
            "rank": [
                "type": "integer",
                "minimum": 1,
                "maximum": 256,
                "description": "LoRA rank, or LoKR linear dimension if linearDim is omitted.",
            ],
            "linearDim": [
                "type": "integer",
                "minimum": 1,
                "maximum": 256,
                "description": "LoKR linear dimension.",
            ],
            "dimension": [
                "type": "integer",
                "minimum": 1,
                "maximum": 256,
                "description": "Alias for rank/linearDim.",
            ],
            "alpha": [
                "type": "number",
                "exclusiveMinimum": 0,
                "description": "Adapter alpha. Defaults to alphaScale * rank/linearDim.",
            ],
            "alphaScale": [
                "description": "Alpha multiplier. Accepts 1x/rank, 2x/double, or a positive number.",
            ],
            "scale": [
                "description": "Alias for alphaScale.",
            ],
            "dropout": [
                "type": "number",
                "minimum": 0,
                "maximum": 0.95,
            ],
            "weightDecompose": [
                "type": "boolean",
                "description": "LoKR-only DoRA/weight-decomposition mode. Defaults to true for LoKR.",
            ],
            "weight_decompose": [
                "type": "boolean",
                "description": "Alias for weightDecompose.",
            ],
            "dora": [
                "type": "boolean",
                "description": "Alias for LoKR weightDecompose.",
            ],
            "epochs": [
                "type": "integer",
                "minimum": 1,
                "maximum": 1000,
            ],
            "maxEpochs": [
                "type": "integer",
                "minimum": 1,
                "maximum": 1000,
                "description": "Alias for epochs.",
            ],
            "layerCoverage": [
                "description": "Layer count or label: last4, last8, last12, last16, all32.",
            ],
            "trainingLayerCount": [
                "type": "integer",
                "minimum": 1,
                "maximum": 32,
                "description": "Alias for layerCoverage.",
            ],
            "layerCount": [
                "type": "integer",
                "minimum": 1,
                "maximum": 32,
                "description": "Alias for layerCoverage.",
            ],
            "projectionCoverage": [
                "type": "string",
                "enum": ["qv", "qkvo", "full"],
                "description": "Projection surface. qv is local-safe; qkvo/full is wider.",
            ],
            "projections": [
                "type": "string",
                "description": "Alias for projectionCoverage.",
            ],
            "projection": [
                "type": "string",
                "description": "Alias for projectionCoverage.",
            ],
            "targetLayers": [
                "description": "String or array containing selfAttention, crossAttention, mlp, attention, or all.",
            ],
            "targets": [
                "description": "Alias for targetLayers.",
            ],
            "exactInjectionPatterns": [
                "type": "array",
                "items": ["type": "string"],
                "description": "Exact module patterns to train. Overrides projectionCoverage.",
            ],
            "injectionPatterns": [
                "type": "array",
                "items": ["type": "string"],
                "description": "Alias for exactInjectionPatterns.",
            ],
            "learningRate": [
                "type": "number",
                "minimum": 0.000001,
                "maximum": 0.01,
            ],
            "lr": [
                "type": "number",
                "description": "Alias for learningRate.",
            ],
            "gradientAccumulationSteps": [
                "type": "integer",
                "minimum": 1,
                "maximum": 64,
            ],
            "gradAccumulation": [
                "type": "integer",
                "description": "Alias for gradientAccumulationSteps.",
            ],
            "gradAccum": [
                "type": "integer",
                "description": "Alias for gradientAccumulationSteps.",
            ],
            "batchSize": [
                "type": "integer",
                "minimum": 1,
                "maximum": 16,
            ],
            "warmupFraction": [
                "type": "number",
                "minimum": 0,
                "maximum": 0.5,
            ],
            "warmup": [
                "type": "number",
                "description": "Alias for warmupFraction.",
            ],
            "clearCache": [
                "type": "boolean",
                "description": "Whether preflight clears the MLX cache before measuring headroom. Defaults to true.",
            ],
        ]

        if includeStartControls {
            properties["dryRun"] = [
                "type": "boolean",
                "description": "Defaults to true. Set false, with confirmation, to actually start training.",
            ]
            properties["validateOnly"] = [
                "type": "boolean",
                "description": "Alias for dryRun.",
            ]
            properties["confirm"] = [
                "type": "boolean",
                "description": "Required for actual training when the client does not support MCP elicitation.",
            ]
        }

        return properties
    }

    private static func tool(
        name: String,
        title: String,
        description: String,
        required: [String] = [],
        properties: [String: Any]
    ) -> [String: Any] {
        [
            "name": name,
            "title": title,
            "description": description,
            "inputSchema": [
                "type": "object",
                "additionalProperties": false,
                "required": required,
                "properties": properties,
            ],
        ]
    }

    private static func sdkTool(_ definition: [String: Any]) -> Tool? {
        guard let name = definition["name"] as? String,
              let description = definition["description"] as? String else {
            return nil
        }

        let inputSchema = (try? MCPValueConverter.value(from: definition["inputSchema"] ?? [:]))
            ?? .object([:])
        return Tool(
            name: name,
            title: definition["title"] as? String,
            description: description,
            inputSchema: inputSchema
        )
    }
}
