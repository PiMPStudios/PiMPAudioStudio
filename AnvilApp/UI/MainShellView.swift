import SwiftUI
import AnvilCore

struct MainShellView: View {
    @EnvironmentObject private var appState: AppState
    @EnvironmentObject private var outputManager: OutputDirectoryManager
    @EnvironmentObject private var sfxModelManager: SFXModelManager
    @EnvironmentObject private var llmManager: LLMModelManager
    @EnvironmentObject private var generationParameters: GenerationParameters
    @EnvironmentObject private var presetStore: PresetStore
    @State private var selectedSidebarItem: SidebarItem = .generate
    @State private var editAudioURL: URL?
    @State private var editOriginalPrompt: String?

    var body: some View {
        if !outputManager.hasValidDirectory() {
            OutputDirectoryPicker()
        } else {
            NavigationSplitView {
                sidebar
            } content: {
                canvas
            } detail: {
                inspector
            }
            .navigationSplitViewStyle(.balanced)
            .onReceive(presetStore.$presetToLoad) { preset in
                guard preset != nil else { return }
                selectedSidebarItem = .generate
            }
            .onReceive(NotificationCenter.default.publisher(for: .navigateToEditor)) { notification in
                if let url = notification.userInfo?["url"] as? URL {
                    editAudioURL = url
                    editOriginalPrompt = notification.userInfo?["prompt"] as? String
                    selectedSidebarItem = .edit
                }
            }
            .overlay(alignment: .bottom) {
                BackgroundDownloadIndicator(
                    phase: appState.backgroundDownloadPhase,
                    onRetry: {
                        Task {
                            await appState.startBackgroundDownloads(
                                sfxManager: sfxModelManager,
                                llmManager: llmManager
                            )
                        }
                    }
                )
                .padding(.bottom, 8)
            }
            .task {
                if appState.backgroundDownloadPhase == .idle {
                    await appState.startBackgroundDownloads(
                        sfxManager: sfxModelManager,
                        llmManager: llmManager
                    )
                }
            }
            .onChange(of: appState.backgroundDownloadPhase) { _, newPhase in
                if newPhase == .complete {
                    Task {
                        try? await Task.sleep(for: .seconds(3))
                        if appState.backgroundDownloadPhase == .complete {
                            appState.backgroundDownloadPhase = .idle
                        }
                    }
                }
            }
        }
    }

    private var sidebar: some View {
        List(selection: $selectedSidebarItem) {
            Section("Create") {
                Label("Generate", systemImage: "waveform")
                    .tag(SidebarItem.generate)
                Label("Edit", systemImage: "waveform.badge.minus")
                    .tag(SidebarItem.edit)
            }
            Section("Library") {
                Label("History", systemImage: "clock")
                    .tag(SidebarItem.history)
                Label("Presets", systemImage: "slider.horizontal.3")
                    .tag(SidebarItem.presets)
            }
        }
        .listStyle(.sidebar)
        .navigationTitle("Anvil")
    }

    @ViewBuilder
    private var canvas: some View {
        switch selectedSidebarItem {
        case .generate:
            GenerateView()
        case .edit:
            EditorView(preloadedAudioURL: editAudioURL, preloadedPrompt: editOriginalPrompt)
        case .history:
            HistoryListView()
        case .presets:
            PresetsView()
        }
    }

    private var inspector: some View {
        Form {
            Section("Pipeline") {
                LabeledContent("Active") {
                    Text(resolvedPipelineName)
                        .foregroundStyle(.secondary)
                }
                if sfxModelManager.downloadState == .downloaded {
                    Picker("Override", selection: $generationParameters.pipelineOverride) {
                        Text("Auto").tag(PipelineChoice.auto)
                        Text("ACE-Step (Music)").tag(PipelineChoice.aceStep)
                        Text("Stable Audio (SFX)").tag(PipelineChoice.stableAudio)
                    }
                    .pickerStyle(.menu)
                }
            }
            Section("Parameters") {
                LabeledContent("Duration") {
                    HStack(spacing: 4) {
                        TextField("30", value: $generationParameters.duration, format: .number)
                            .multilineTextAlignment(.trailing)
                        Text("s").foregroundStyle(.secondary)
                    }
                }
                LabeledContent("Seed") {
                    TextField("Random", text: $generationParameters.seedText)
                        .multilineTextAlignment(.trailing)
                }
                LabeledContent("Guidance") {
                    TextField("7.5", value: $generationParameters.guidanceScale, format: .number.precision(.fractionLength(1)))
                        .multilineTextAlignment(.trailing)
                }
            }
            ModelInspectorSection()
        }
        .formStyle(.grouped)
        .frame(minWidth: 250)
    }

    private var resolvedPipelineName: String {
        switch generationParameters.pipelineOverride {
        case .aceStep: return "ACE-Step"
        case .stableAudio: return "Stable Audio"
        case .auto: return "Auto"
        }
    }
}

enum SidebarItem: Hashable {
    case generate
    case edit
    case history
    case presets
}

// MARK: - Model Inspector Section

/// Live model status + quick variant switcher for the main-window inspector.
/// Shows the active DiT name, a dot-badge for ready/downloading/error,
/// storage size, a Picker to switch between downloaded variants, and a link
/// that opens the full Models settings tab.
private struct ModelInspectorSection: View {
    @EnvironmentObject private var appState: AppState
    @EnvironmentObject private var sfxModelManager: SFXModelManager
    @Environment(\.openSettings) private var openSettings

    @State private var ditSpecs: [ModelSpec] = []
    @State private var activeDiTID: String? = nil
    @State private var ditStatuses: [String: ModelStatus] = [:]
    @State private var isSwitching = false

    private var activeDiTSpec: ModelSpec? {
        guard let id = activeDiTID else { return nil }
        return ditSpecs.first { $0.id == id }
    }

    /// Only runnable variants that are fully downloaded appear in the quick picker.
    private var readyDiTSpecs: [ModelSpec] {
        ditSpecs.filter { ditStatuses[$0.id] == .ready }
    }

    var body: some View {
        Section("Models") {
            aceStepRow
            sfxRow
            Button("Manage Models…") { openSettings() }
                .buttonStyle(.link)
                .frame(maxWidth: .infinity, alignment: .trailing)
        }
        .task { await loadModelInfo() }
    }

    // MARK: ACE-Step row

    @ViewBuilder
    private var aceStepRow: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 6) {
                statusDot(activeDiTStatusColor)
                Text("ACE-Step")
                    .font(.subheadline)
                Spacer()
                if let spec = activeDiTSpec {
                    Text(spec.totalSizeFormatted)
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }
            }

            if readyDiTSpecs.count > 1 {
                // Multiple downloaded variants → show a compact switcher.
                Picker("", selection: Binding(
                    get: { activeDiTID ?? "" },
                    set: { id in switchDiT(to: id) }
                )) {
                    ForEach(readyDiTSpecs, id: \.id) { spec in
                        Text(shortVariantName(spec)).tag(spec.id)
                    }
                }
                .pickerStyle(.menu)
                .labelsHidden()
                .disabled(isSwitching)
                .frame(maxWidth: .infinity, alignment: .leading)
            } else {
                // Single variant or none — just name it.
                Text(activeDiTSpec.map { shortVariantName($0) } ?? "No model selected")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.vertical, 2)
    }

    // MARK: Stable Audio row

    @ViewBuilder
    private var sfxRow: some View {
        HStack(spacing: 6) {
            statusDot(sfxStatusColor)
            Text("Stable Audio")
                .font(.subheadline)
            Spacer()
            Text(sfxStatusLabel)
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .padding(.vertical, 2)
    }

    // MARK: Helpers

    private func statusDot(_ color: Color) -> some View {
        Circle()
            .fill(color)
            .frame(width: 7, height: 7)
    }

    private var activeDiTStatusColor: Color {
        guard let id = activeDiTID else { return .secondary }
        switch ditStatuses[id] {
        case .ready:           return .green
        case .downloading:     return .yellow
        case .failed:          return .red
        case .verifying:       return .yellow
        case .notDownloaded, nil: return .secondary
        }
    }

    private var sfxStatusColor: Color {
        switch sfxModelManager.downloadState {
        case .downloaded:      return .green
        case .downloading:     return .yellow
        case .error:           return .red
        case .notDownloaded:   return .secondary
        }
    }

    private var sfxStatusLabel: String {
        switch sfxModelManager.downloadState {
        case .downloaded:             return "Ready"
        case .downloading(let p):     return "\(Int(p * 100))%"
        case .error(let msg):         return msg
        case .notDownloaded:          return "Not downloaded"
        }
    }

    private func shortVariantName(_ spec: ModelSpec) -> String {
        switch spec.id {
        case "ace-step-dit-turbo": return "Turbo  –  8 steps"
        case "ace-step-dit-base":  return "Full  –  50 steps"
        case "ace-step-dit-sft":   return "SFT  –  50 steps"
        default:
            return spec.id.contains("xl") ? "XL  –  50 steps" : spec.displayName
        }
    }

    private func loadModelInfo() async {
        let registry = appState.registry
        let allSpecs = await registry.specs(for: .aceStepDiT)
        let runnable  = allSpecs.filter { $0.isRunnable }
        let activeID  = await registry.activeModelID(for: .aceStepDiT)
        var statuses: [String: ModelStatus] = [:]
        for spec in runnable {
            statuses[spec.id] = await registry.status(for: spec.id)
        }
        ditSpecs      = runnable
        activeDiTID   = activeID
        ditStatuses   = statuses
    }

    private func switchDiT(to id: String) {
        guard !isSwitching, id != activeDiTID else { return }
        isSwitching = true
        Task {
            try? await appState.registry.setActiveModel(id: id, for: .aceStepDiT)
            await loadModelInfo()
            isSwitching = false
        }
    }
}

