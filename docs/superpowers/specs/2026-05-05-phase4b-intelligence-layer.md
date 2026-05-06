# Phase 4b: Intelligence Layer — Design Spec

## Overview

Add an intelligence layer on top of Phase 4a's generation flow: live intent
classification, lyric writing with streaming, and optional prompt enhancement.
All three features share a single local LLM instance (different system prompts),
with optional API backends for lyric writing.

**Scope boundary:** Phase 4b does NOT include audio editing, A/B comparison,
loop playback, or LoRA fine-tuning. It delivers smart routing, AI-assisted
lyrics, and prompt expansion on top of the existing generation pipeline.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| LLM sharing | Single model, three system prompts | Lowest memory, no second download |
| Prompt enhancement | Optional "Enhance" button | User stays in control, can edit result |
| Lyric flow | Inline streaming into TextEditor | Direct, no modal interruption |
| Intent routing | Live classification as user types | Immediate feedback, debounced 300ms |
| Model choice | Configurable with sensible default | Creative tool users want flexibility |
| API backends | Raw URLSession, no SDK deps | Zero dependency footprint |
| Key storage | Keychain + env var fallback | Works for both GUI users and developers |
| Architecture split | LLMActor in AnvilCore, services in AnvilApp | Testable inference, app-level orchestration |

## Component 1: LLMActor (AnvilCore)

Swift actor providing local LLM inference via MLX-Swift. Lives at
`AnvilCore/Sources/AnvilCore/LLM/`.

### API

```swift
public actor LLMActor {
    private var model: (any Module)?
    private var tokenizer: Tokenizer?
    private let modelPath: URL

    public init(modelPath: URL)

    public func loadModel() async throws
    public var isLoaded: Bool { get }

    public func generate(
        systemPrompt: String,
        userPrompt: String,
        maxTokens: Int = 512,
        temperature: Float = 0.7
    ) -> AsyncStream<String>

    public func classify(
        systemPrompt: String,
        userPrompt: String
    ) async -> String
}
```

### Behavior

- `generate` streams tokens one at a time via AsyncStream. Uses autoregressive
  generation with KV cache for efficiency.
- `classify` is a convenience wrapper: calls generate with `maxTokens: 20`,
  `temperature: 0`, returns the concatenated output as one string. Optimized
  for IntentRouter where streaming is unnecessary and latency matters.
- Model loading is explicit via `loadModel()`. Callers check `isLoaded` before
  invoking generation. If not loaded, services show appropriate disabled states.
- Supports any MLX-format model directory (safetensors + tokenizer.json +
  config.json).

### Default Model

`mlx-community/Llama-3.2-3B-Instruct-4bit` — good creative writing ability,
fast on Apple Silicon, ~1.8GB download. User can swap to any MLX-compatible
model via Settings.

## Component 2: IntentRouter (AnvilApp)

Real-time prompt classification. Lives at `AnvilApp/Services/IntentRouter.swift`.

### API

```swift
@MainActor
final class IntentRouter: ObservableObject {
    @Published var detectedMode: GenerationMode = .music
    @Published var isClassifying: Bool = false

    private let llm: LLMActor
    private var debounceTask: Task<Void, Never>?

    init(llm: LLMActor)

    func classify(prompt: String)
    func cancel()
}

enum GenerationMode: String, CaseIterable, Sendable {
    case music
    case soundFX
    case ambient
}
```

### Behavior

- `classify` debounces input by 300ms. Each call cancels the previous pending
  classification.
- After the debounce delay, calls `llm.classify()` with the classification
  system prompt.
- Classification system prompt:
  `"You are a classifier. Given a user's audio generation prompt, respond with exactly one word: \"music\", \"soundfx\", or \"ambient\". No explanation."`
- Expected latency: ~50-100ms on Apple Silicon with 4-bit 3B model for a
  10-20 token output.
- `detectedMode` updates trigger UI reflection in the mode picker when Auto
  is selected.

### Mode Resolution

The GenerateView has a mode picker with four options: Auto, Music, Sound FX,
Ambient. When the user selects Auto, `intentRouter.detectedMode` drives the
UI indicator. When manually pinned, the router still runs (so switching back
to Auto is instant) but its result is ignored for generation purposes.

The resolved mode for generation:
- Manual selection: use that mode directly
- Auto: use `intentRouter.detectedMode`

## Component 3: LyricWriterService (AnvilApp)

Protocol-based lyric generation with multiple backends. Lives at
`AnvilApp/Services/LyricWriter/`.

### Protocol

```swift
public protocol LyricWriter: Sendable {
    var displayName: String { get }
    var requiresAPIKey: Bool { get }
    func write(
        prompt: String,
        style: String,
        duration: TimeInterval
    ) -> AsyncStream<String>
}
```

### Backends

| Backend | Key Required | API Format |
|---------|-------------|------------|
| `LocalLyricWriter` | No | LLMActor.generate() |
| `ClaudeLyricWriter` | `ANTHROPIC_API_KEY` | Anthropic Messages API (streaming) |
| `OpenAILyricWriter` | `OPENAI_API_KEY` | Chat Completions (streaming) |
| `GeminiLyricWriter` | `GOOGLE_API_KEY` | Gemini generateContent (streaming) |
| `XAILyricWriter` | `XAI_API_KEY` | OpenAI-compatible chat completions (streaming) |

All API backends use raw `URLSession` with async bytes — no SDK dependencies.
xAI uses the same request format as OpenAI with a different base URL
(`https://api.x.ai/v1/`).

### Lyric System Prompt

```text
You are a songwriter. Write lyrics for a song based on the user's description.
Use section markers: [Verse], [Chorus], [Bridge], [Outro].
Match the mood, genre, and energy of the description.
Target duration: {duration} seconds (roughly {duration/15} lines per section).
Output only the lyrics with section markers. No commentary.
```

### LyricWriterManager

```swift
@MainActor
final class LyricWriterManager: ObservableObject {
    @Published var isWriting: Bool = false
    @Published var activeBackend: String = "local"

    private let backends: [String: any LyricWriter]
    private var writeTask: Task<Void, Never>?

    init(localLLM: LLMActor, apiKeyStore: APIKeyStore)

    func writeLyrics(
        prompt: String,
        style: String,
        duration: TimeInterval,
        into binding: Binding<String>
    )
    func cancel()
    func availableBackends() -> [String]
}
```

### Behavior

- `writeLyrics` clears the lyrics field, then streams tokens into it via the
  binding. User sees text appearing in real-time.
- Cancel stops the stream and leaves whatever has been written so far.
- `availableBackends()` returns backends that are usable (local is always
  available if LLM loaded; API backends available if their key exists in
  APIKeyStore).
- Only visible/enabled when resolved mode is `.music`.

## Component 4: PromptEnhancer (AnvilApp)

Expands terse prompts into rich descriptions. Lives at
`AnvilApp/Services/PromptEnhancer.swift`.

### API

```swift
@MainActor
final class PromptEnhancer: ObservableObject {
    @Published var isEnhancing: Bool = false

    private let llm: LLMActor
    private var enhanceTask: Task<Void, Never>?

    init(llm: LLMActor)

    func enhance(prompt: String, mode: GenerationMode) async -> String
    func cancel()
}
```

### Behavior

- Non-streaming: returns the full enhanced prompt as a single string. The
  output is short (~50-150 words) and completes in 1-2 seconds on a 3B 4-bit
  model.
- Mode-aware system prompt:
  `"You are an audio prompt enhancer. Given a brief description and the generation mode ({mode}), expand it into a detailed, vivid description that will produce high-quality audio. Include genre, instruments, tempo, mood, texture, and production style where relevant. Output only the enhanced prompt, nothing else. Keep it under 200 words."`
- Result replaces the prompt field text. User can undo (Cmd+Z) or edit.
- Disabled when LLM is not loaded or prompt is empty.
- Works for all modes (music, soundFX, ambient).

## Component 5: APIKeyStore (AnvilApp)

Unified API key management. Lives at `AnvilApp/Services/APIKeyStore.swift`.

### API

```swift
@MainActor
final class APIKeyStore: ObservableObject {
    @Published var availableProviders: Set<String> = []

    func apiKey(for provider: String) -> String?
    func setAPIKey(_ key: String, for provider: String)
    func removeAPIKey(for provider: String)
    func hasKey(for provider: String) -> Bool
}
```

### Providers

`"anthropic"`, `"openai"`, `"google"`, `"xai"`

### Resolution Order

1. Keychain (user explicitly set via Settings)
2. Environment variable (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`,
   `XAI_API_KEY`)

Keychain takes priority if both exist. `availableProviders` is recomputed on
init and after any mutation.

### Keychain Implementation

Uses `Security.framework` directly (SecItemAdd/Update/Delete/CopyMatching).
Service name: `"com.pimpstudios.anvil.apikeys"`. Account name: the provider
string. No third-party Keychain libraries.

## Component 6: LLMModelManager (AnvilApp)

Manages local LLM weight download and selection. Lives at
`AnvilApp/Services/LLMModelManager.swift`.

### API

```swift
@MainActor
final class LLMModelManager: ObservableObject {
    @Published var availableModels: [LLMModelSpec] = []
    @Published var activeModel: LLMModelSpec?
    @Published var downloadState: LLMDownloadState = .idle

    func downloadModel(_ spec: LLMModelSpec) async throws
    func deleteModel(_ spec: LLMModelSpec)
    func setActiveModel(_ spec: LLMModelSpec)
    func resolveModelPath() -> URL?
}

struct LLMModelSpec: Identifiable, Codable, Sendable {
    let id: String
    let displayName: String
    let repoID: String
    let sizeBytes: Int64
    let isDefault: Bool
}

enum LLMDownloadState: Equatable, Sendable {
    case idle
    case downloading(progress: Float)
    case error(String)
}
```

### Storage

Models download to `~/Library/Application Support/Anvil/LLM/<model-id>/`.
Each model directory contains safetensors, tokenizer.json, and config.json.

### Default Model

One pre-configured `LLMModelSpec` with `isDefault: true`. Downloaded during
first-launch flow (alongside music model weights) or deferred until the user
first triggers an intelligence feature.

### Custom Models

Settings exposes "Add Model" — user pastes a HuggingFace repo ID. The manager
validates the repo has required files (config.json, tokenizer.json,
*.safetensors) before adding to `availableModels`.

### Model Switching

When `setActiveModel` is called, `LLMModelManager` creates a new `LLMActor`
instance with the new model path and publishes it. The old actor is released,
freeing its weights from memory. All dependent services (IntentRouter,
PromptEnhancer, LocalLyricWriter) receive the new actor reference via the
environment/dependency injection.

### Lifecycle

`LLMModelManager` owns the `LLMActor` instance:

```swift
@Published var llmActor: LLMActor?
```

On app launch, if an active model is configured and downloaded, it creates
the `LLMActor` but does NOT call `loadModel()` — loading is deferred until
the first feature that needs it (first classify, enhance, or write lyrics
call). This avoids slow startup. Services check `llmActor?.isLoaded` to
determine button enabled states.

## Component 7: UI Changes

### GenerateView Modifications

- **Mode picker** gains an "Auto" option (4 total: Auto, Music, Sound FX,
  Ambient). When Auto is active, a subtle label shows the detected mode
  (e.g., "Auto: Music").
- **"Enhance" button** — sparkle icon to the right of the "Prompt" label.
  Disabled when LLM not loaded or prompt empty. Shows spinner while enhancing.
  Tapping replaces prompt text with enhanced version.
- **"Write Lyrics" button** — below the lyrics TextEditor. Only visible when
  resolved mode is `.music`. Shows "Cancel" while streaming. Lyrics field is
  non-editable during streaming to avoid conflicts.

### Settings Panel — Intelligence Section

- **Model** — dropdown of available models + download state + "Add Custom
  Model" row.
- **Lyric Writer Backend** — picker (Local, Claude, OpenAI, Gemini, xAI).
  Non-local options disabled without configured key.
- **API Keys** — one row per provider: name, status indicator (configured /
  from environment / not set), SecureField, Remove button.

### State Flow

```text
User types prompt
  → IntentRouter.classify(prompt) [debounced 300ms]
  → mode picker updates to show detected mode

User taps "Enhance"
  → PromptEnhancer.enhance(prompt, mode)
  → prompt field text replaced with result

User taps "Write Lyrics" (music mode only)
  → LyricWriterManager.writeLyrics(prompt, style, duration, into: $lyricsText)
  → tokens stream into lyrics field

User taps "Generate"
  → uses resolved mode + (possibly enhanced) prompt + (possibly generated) lyrics
  → existing GenerationService flow from Phase 4a
```

## Package Structure

```text
AnvilCore/Sources/AnvilCore/
├── LLM/
│   └── LLMActor.swift                    (new)

AnvilApp/
├── Services/
│   ├── IntentRouter.swift                (new)
│   ├── PromptEnhancer.swift              (new)
│   ├── APIKeyStore.swift                 (new)
│   ├── LLMModelManager.swift             (new)
│   └── LyricWriter/
│       ├── LyricWriter.swift             (new — protocol)
│       ├── LyricWriterManager.swift      (new)
│       ├── LocalLyricWriter.swift        (new)
│       ├── ClaudeLyricWriter.swift       (new)
│       ├── OpenAILyricWriter.swift       (new)
│       ├── GeminiLyricWriter.swift       (new)
│       └── XAILyricWriter.swift          (new)
├── UI/
│   ├── Components/
│   │   └── GenerateView.swift            (modified)
│   └── Settings/
│       └── IntelligenceSettingsView.swift (new)
├── Models/
│   └── GenerationMode.swift              (new)
```

## Testing Strategy

1. **LLMActor** — unit test with a tiny test model (or mock): verify streaming
   output, verify classify returns single token, verify cancellation stops
   generation.
2. **IntentRouter** — test debounce behavior (rapid calls only trigger one
   classification), test mode parsing from LLM output.
3. **LyricWriterManager** — test backend selection, test streaming into binding,
   test cancellation leaves partial output.
4. **PromptEnhancer** — test enhancement returns non-empty string, test mode
   is included in system prompt.
5. **APIKeyStore** — test Keychain read/write/delete, test env var fallback,
   test priority (Keychain over env).
6. **LLMModelManager** — test model spec persistence, test active model
   selection, test path resolution.
7. **API backends** — test request formatting for each provider (mock
   URLSession), verify streaming SSE parsing.

## Dependencies

- **MLX, MLXNN, MLXRandom** (existing — for LLMActor inference)
- **swift-transformers** (existing — for tokenizer)
- **Security.framework** (system — for Keychain)
- No new external packages

## Non-Goals (Later Phases)

- Voice cloning / speaker selection
- Multi-language lyric writing
- Prompt templates / presets library
- Collaborative prompt editing
- Fine-tuning the local LLM on user style
