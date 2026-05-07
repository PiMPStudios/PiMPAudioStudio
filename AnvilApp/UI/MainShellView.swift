import SwiftUI
import AnvilCore

struct MainShellView: View {
    @EnvironmentObject private var appState: AppState
    @EnvironmentObject private var outputManager: OutputDirectoryManager
    @EnvironmentObject private var sfxModelManager: SFXModelManager
    @EnvironmentObject private var llmManager: LLMModelManager
    @State private var selectedSidebarItem: SidebarItem = .generate
    @State private var pipelineOverride: PipelineChoice = .auto
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
            Text("Presets coming in Phase 4b")
                .foregroundStyle(.secondary)
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
                    Picker("Override", selection: $pipelineOverride) {
                        Text("Auto").tag(PipelineChoice.auto)
                        Text("ACE-Step (Music)").tag(PipelineChoice.aceStep)
                        Text("Stable Audio (SFX)").tag(PipelineChoice.stableAudio)
                    }
                    .pickerStyle(.menu)
                }
            }
            Section("Parameters") {
                LabeledContent("Duration") { TextField("30s", text: .constant("")) }
                LabeledContent("Seed") { TextField("Random", text: .constant("")) }
                LabeledContent("Guidance") { TextField("1.0", text: .constant("")) }
            }
            Section("Model") {
                Text("ACE-Step DiT v1.5")
                    .foregroundStyle(.secondary)
                if sfxModelManager.downloadState == .downloaded {
                    Text("Stable Audio Open 1.0")
                        .foregroundStyle(.secondary)
                }
            }
        }
        .formStyle(.grouped)
        .frame(minWidth: 250)
    }

    private var resolvedPipelineName: String {
        switch pipelineOverride {
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

