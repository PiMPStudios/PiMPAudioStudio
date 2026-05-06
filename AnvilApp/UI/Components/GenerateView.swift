import SwiftUI
import AnvilCore

struct GenerateView: View {
    @EnvironmentObject private var generationService: GenerationService
    @EnvironmentObject private var playerService: AudioPlayerService
    @EnvironmentObject private var intentRouter: IntentRouter
    @EnvironmentObject private var promptEnhancer: PromptEnhancer
    @EnvironmentObject private var lyricWriterManager: LyricWriterManager
    @EnvironmentObject private var sfxModelManager: SFXModelManager

    @State private var promptText: String = ""
    @State private var lyricsText: String = ""
    @State private var duration: Double = 30
    @State private var seed: String = ""
    @State private var guidanceScale: Double = 1.0
    @State private var method: InferMethod = .ode
    @State private var precision: Precision = .float16
    @State private var modeSelection: ModeSelection = .auto
    @State private var pipelineOverride: PipelineChoice = .auto
    @State private var selectedAdapterID: String?
    @AppStorage("selectedAdapterID") private var persistedAdapterID: String = ""

    var body: some View {
        ScrollView {
            VStack(spacing: 16) {
                modePickerSection
                promptSection
                lyricsSection
                controlsBar
                adapterPickerSection
                progressSection
                playerSection
            }
            .padding()
        }
        .onChange(of: promptText) { _, newValue in
            intentRouter.classify(prompt: newValue)
        }
    }

    private var resolvedMode: GenerationMode {
        switch modeSelection {
        case .auto: return intentRouter.detectedMode
        case .manual(let mode): return mode
        }
    }

    private var modePickerSection: some View {
        HStack(spacing: 8) {
            modeButton("Auto", isSelected: modeSelection == .auto) {
                modeSelection = .auto
            }
            ForEach(GenerationMode.allCases) { mode in
                modeButton(mode.displayName, isSelected: modeSelection == .manual(mode), isSFXMode: mode == .soundFX || mode == .ambient) {
                    modeSelection = .manual(mode)
                }
            }

            if case .auto = modeSelection {
                Text("→ \(intentRouter.detectedMode.displayName)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                if intentRouter.isClassifying {
                    ProgressView()
                        .controlSize(.mini)
                }
            }

            Spacer()
        }
    }

    private func modeButton(_ title: String, isSelected: Bool, isSFXMode: Bool = false, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack(spacing: 4) {
                Text(title)
                if isSFXMode && sfxModelManager.downloadState == .notDownloaded {
                    Image(systemName: "arrow.down.circle")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }
        }
        .buttonStyle(.bordered)
        .tint(isSelected ? .accentColor : .secondary)
        .controlSize(.small)
    }

    private var promptSection: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text("Prompt")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
                Button {
                    Task {
                        let enhanced = await promptEnhancer.enhance(
                            prompt: promptText,
                            mode: resolvedMode
                        )
                        promptText = enhanced
                    }
                } label: {
                    if promptEnhancer.isEnhancing {
                        ProgressView()
                            .controlSize(.mini)
                    } else {
                        Image(systemName: "sparkles")
                    }
                }
                .buttonStyle(.borderless)
                .disabled(!promptEnhancer.isAvailable || promptText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || promptEnhancer.isEnhancing)
                .help("Enhance prompt")
            }
            TextEditor(text: $promptText)
                .font(.body)
                .frame(minHeight: 60, maxHeight: 100)
                .overlay(RoundedRectangle(cornerRadius: 8).stroke(.quaternary))
        }
    }

    private var lyricsSection: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text("Lyrics")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
                if resolvedMode == .music {
                    if lyricWriterManager.isWriting {
                        Button("Cancel") {
                            lyricWriterManager.cancel()
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.mini)
                    } else {
                        Button("Write Lyrics") {
                            lyricWriterManager.writeLyrics(
                                prompt: promptText,
                                style: "",
                                duration: duration,
                                into: $lyricsText
                            )
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.mini)
                        .disabled(promptText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    }
                }
            }
            TextEditor(text: $lyricsText)
                .font(.body)
                .frame(minHeight: 80, maxHeight: 120)
                .overlay(RoundedRectangle(cornerRadius: 8).stroke(.quaternary))
                .disabled(lyricWriterManager.isWriting)
        }
    }

    private var controlsBar: some View {
        HStack {
            Button("Generate") {
                let request = GenerationRequest(
                    text: promptText,
                    lyrics: lyricsText,
                    duration: effectiveDuration,
                    seed: Int(seed),
                    method: method,
                    guidanceScale: Float(guidanceScale),
                    precision: precision,
                    resolvedMode: resolvedMode,
                    pipelineOverride: pipelineOverride
                )
                generationService.generate(request: request)
            }
            .buttonStyle(.borderedProminent)
            .disabled(!canGenerate)

            if case .generating = generationService.state {
                Button("Cancel", role: .destructive) {
                    generationService.cancel()
                }
                .buttonStyle(.bordered)
            }

            Spacer()

            if let eta = estimatedTime {
                Text("~\(Int(eta))s remaining")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private var adapterPickerSection: some View {
        AdapterPicker(
            selectedAdapterID: $selectedAdapterID,
            activeModelCategory: .aceStepDiT
        )
        .onChange(of: selectedAdapterID) { _, newValue in
            persistedAdapterID = newValue ?? ""
            generationService.selectedAdapterID = newValue
        }
        .onAppear {
            if !persistedAdapterID.isEmpty {
                selectedAdapterID = persistedAdapterID
                generationService.selectedAdapterID = persistedAdapterID
            }
        }
    }

    @ViewBuilder
    private var progressSection: some View {
        switch generationService.state {
        case .idle, .complete, .cancelled:
            EmptyView()
        case .downloadingModel(let progress):
            ProgressView("Downloading Sound FX model...", value: Double(progress), total: 1.0)
        case .loadingWeights(let progress):
            ProgressView("Loading weights...", value: Double(progress), total: 1.0)
        case .conditioning:
            ProgressView("Encoding text and lyrics...")
                .progressViewStyle(.linear)
        case .generating(let step, let total, _):
            VStack(spacing: 4) {
                ProgressView(value: Double(step), total: Double(total))
                Text("Step \(step) / \(total)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        case .decoding:
            ProgressView("Decoding audio...")
                .progressViewStyle(.linear)
        case .error(let message):
            Label(message, systemImage: "exclamationmark.triangle")
                .foregroundStyle(.red)
        }
    }

    @ViewBuilder
    private var playerSection: some View {
        if let result = generationService.currentResult {
            PlayerView(fileURL: result.fileURL)
        }
    }

    private var effectiveDuration: TimeInterval {
        if shouldUseStableAudio {
            return min(duration, 47.0)
        }
        return duration
    }

    private var shouldUseStableAudio: Bool {
        switch pipelineOverride {
        case .aceStep: return false
        case .stableAudio: return true
        case .auto: return resolvedMode == .soundFX || resolvedMode == .ambient
        }
    }

    private var canGenerate: Bool {
        let isReady: Bool
        switch generationService.state {
        case .idle, .complete, .cancelled, .error:
            isReady = true
        default:
            isReady = false
        }
        return isReady && !promptText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private var estimatedTime: TimeInterval? {
        if case .generating(_, _, let eta) = generationService.state, eta > 0 {
            return eta
        }
        return nil
    }
}
