import SwiftUI
import PiMPAudioCore
import AppKit

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
            Section("Build") {
                Label("Datasets", systemImage: "folder.badge.plus")
                    .tag(SidebarItem.datasets)
                Label("Training", systemImage: "graduationcap")
                    .tag(SidebarItem.training)
            }
            Section("Library") {
                Label("History", systemImage: "clock")
                    .tag(SidebarItem.history)
                Label("Presets", systemImage: "slider.horizontal.3")
                    .tag(SidebarItem.presets)
            }
        }
        .listStyle(.sidebar)
        .navigationTitle(AppRuntimeIdentity.displayName)
    }

    @ViewBuilder
    private var canvas: some View {
        switch selectedSidebarItem {
        case .generate:
            GenerateView()
        case .edit:
            EditorView(preloadedAudioURL: editAudioURL, preloadedPrompt: editOriginalPrompt)
        case .datasets:
            BuildDatasetsView()
        case .training:
            BuildTrainingView()
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
                        .lineLimit(1)
                }
                LabeledContent("Guidance") {
                    HStack(spacing: 6) {
                        TextField("7.5", value: $generationParameters.guidanceScale, format: .number.precision(.fractionLength(1)))
                            .multilineTextAlignment(.trailing)
                            .disabled(generationParameters.guidancePreset.overridesGuidanceScale)
                        if let override = generationParameters.guidancePreset.guidanceScaleOverride {
                            Text("→ \(String(format: "%.1f", override))")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .help(generationParameters.guidancePreset.detail)
                        }
                    }
                }
                if generationParameters.guidancePreset.overridesGuidanceScale {
                    Text("\(generationParameters.guidancePreset.displayName) forces guidance \(String(format: "%.1f", generationParameters.guidancePreset.guidanceScaleOverride ?? 0))")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
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
    case datasets
    case training
    case history
    case presets
}

// MARK: - Build Section

private struct BuildDatasetsView: View {
    @EnvironmentObject private var outputManager: OutputDirectoryManager

    @State private var showingYouTubeSheet = false
    @State private var showingLocalFilesSheet = false
    @State private var packages: [DatasetPackageSummary] = []
    @State private var selectedPackageID: String?
    @State private var selectedPackageDirectory: URL?
    @State private var packageNotice: String?
    @State private var trainingPresentation: TrainingPresentation?

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            header
            Divider()
            HSplitView {
                packageSidebar
                    .frame(minWidth: 260, idealWidth: 320, maxWidth: 420)
                if let selectedPackageDirectory {
                    DatasetReviewView(datasetDirectory: selectedPackageDirectory) { files in
                        trainingPresentation = TrainingPresentation(files: files)
                    } onPackageUpdated: {
                        refreshPackages()
                    }
                    .id(selectedPackageDirectory.path(percentEncoded: false))
                } else {
                    VStack(spacing: 8) {
                        Image(systemName: "folder.badge.gearshape")
                            .font(.largeTitle)
                            .foregroundStyle(.secondary)
                        Text("Select a Dataset")
                            .font(.headline)
                        Text("Build or open a package to review captions, lyrics, BPM, key, stems, and training readiness.")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                            .multilineTextAlignment(.center)
                            .frame(maxWidth: 420)
                    }
                    .padding()
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                }
            }
        }
        .onAppear(perform: refreshPackages)
        .onChange(of: outputManager.directory) { _, _ in refreshPackages() }
        .onChange(of: showingLocalFilesSheet) { _, isShowing in
            if !isShowing { refreshPackages() }
        }
        .onChange(of: showingYouTubeSheet) { _, isShowing in
            if !isShowing { refreshPackages() }
        }
        .sheet(isPresented: $showingLocalFilesSheet) {
            LocalFilesImportSheet { directory in
                showingLocalFilesSheet = false
                DispatchQueue.main.async {
                    refreshPackages()
                    selectPackage(directory)
                }
            }
            .environmentObject(outputManager)
        }
        .sheet(isPresented: $showingYouTubeSheet) {
            YouTubeImportSheet { files in
                showingYouTubeSheet = false
                DispatchQueue.main.async {
                    refreshPackages()
                    trainingPresentation = TrainingPresentation(files: files)
                }
            }
            .environmentObject(outputManager)
        }
        .sheet(item: $trainingPresentation) { presentation in
            TrainingSheet(
                selectedEntries: presentation.selectedEntries,
                initialImportedFiles: presentation.files,
                initialName: presentation.initialName,
                initialConfig: presentation.initialConfig,
                initialLoKRConfig: presentation.initialLoKRConfig,
                resumeAdapterID: presentation.resumeAdapterID
            )
        }
    }

    private var header: some View {
        HStack {
            Text("Datasets")
                .font(.headline)
            Spacer()
            Button {
                showingLocalFilesSheet = true
            } label: {
                Label("From Local Files\u{2026}", systemImage: "folder.badge.plus")
            }
            .buttonStyle(.borderedProminent)
            Button {
                showingYouTubeSheet = true
            } label: {
                Label("From YouTube\u{2026}", systemImage: "play.rectangle")
            }
            .buttonStyle(.bordered)
        }
        .padding()
    }

    private var packageSidebar: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text("\(packages.count) package\(packages.count == 1 ? "" : "s")")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
                Button {
                    openPackageFolder()
                } label: {
                    Image(systemName: "folder")
                }
                .help("Open dataset package folder")
                .buttonStyle(.borderless)
                Button {
                    refreshPackages()
                } label: {
                    Image(systemName: "arrow.clockwise")
                }
                .help("Refresh packages")
                .buttonStyle(.borderless)
            }
            .padding(.horizontal)
            .padding(.vertical, 8)

            if let packageNotice {
                Text(packageNotice)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .padding(.horizontal)
                    .padding(.bottom, 6)
            }

            List(selection: $selectedPackageID) {
                Section("Packages") {
                    ForEach(packages) { package in
                        packageRow(package)
                            .tag(package.id)
                    }
                }
                Section("Build") {
                    Button {
                        showingLocalFilesSheet = true
                    } label: {
                        Label("From Local Files\u{2026}", systemImage: "folder.badge.plus")
                    }
                    Button {
                        showingYouTubeSheet = true
                    } label: {
                        Label("From YouTube\u{2026}", systemImage: "play.rectangle")
                    }
                }
            }
            .listStyle(.sidebar)
            .onChange(of: selectedPackageID) { _, id in
                selectedPackageDirectory = packages.first(where: { $0.id == id })?.directory
            }
        }
    }

    private func packageRow(_ package: DatasetPackageSummary) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 6) {
                Image(systemName: package.isValid ? "checkmark.circle.fill" : "exclamationmark.triangle.fill")
                    .foregroundStyle(package.isValid ? .green : .orange)
                    .frame(width: 16)
                Text(package.name)
                    .font(.callout)
                    .fontWeight(.medium)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
            HStack(spacing: 6) {
                Text("\(package.clipCount) clips")
                Text(package.enrichmentStatus)
                if package.transcriptionStatus != "not_run" {
                    Text("lyrics \(package.transcriptionStatus)")
                }
            }
            .font(.caption2)
            .foregroundStyle(.secondary)
            Text(package.validationSummary)
                .font(.caption2)
                .foregroundStyle(.tertiary)
                .lineLimit(1)
        }
        .padding(.vertical, 4)
    }

    private func refreshPackages() {
        guard let outputDirectory = outputManager.directory else {
            packages = []
            selectedPackageID = nil
            selectedPackageDirectory = nil
            return
        }

        packages = TrainingDatasetPackageEditor.listPackages(in: outputDirectory)
        if let selectedPackageID,
           let selected = packages.first(where: { $0.id == selectedPackageID }) {
            selectedPackageDirectory = selected.directory
        } else {
            selectedPackageID = packages.first?.id
            selectedPackageDirectory = packages.first?.directory
        }
    }

    private func selectPackage(_ directory: URL) {
        let summary = TrainingDatasetPackageEditor.packageSummary(at: directory)
        if !packages.contains(where: { $0.id == summary.id }) {
            packages.insert(summary, at: 0)
        }
        selectedPackageID = summary.id
        selectedPackageDirectory = directory
        packageNotice = nil
    }

    private func openPackageFolder() {
        let panel = NSOpenPanel()
        panel.title = "Open Dataset Package"
        panel.prompt = "Open"
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.canCreateDirectories = false

        if let outputDirectory = outputManager.directory {
            panel.directoryURL = outputDirectory.appendingPathComponent("TrainingDatasets", isDirectory: true)
        }

        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            _ = try TrainingDatasetPackageEditor.loadPackage(at: url)
            let summary = TrainingDatasetPackageEditor.packageSummary(at: url)
            if !packages.contains(where: { $0.id == summary.id }) {
                packages.insert(summary, at: 0)
            }
            selectedPackageID = summary.id
            selectedPackageDirectory = url
            packageNotice = nil
        } catch {
            packageNotice = error.localizedDescription
        }
    }
}

private struct BuildTrainingView: View {
    @EnvironmentObject private var trainingService: TrainingService
    @EnvironmentObject private var outputManager: OutputDirectoryManager
    @EnvironmentObject private var runPodService: RunPodService

    @State private var selectedTrainingMode: BuildTrainingMode = .lora
    @State private var trainingPresentation: TrainingPresentation?
    @State private var showingStopConfirmation = false
    @State private var showingRunPodLaunchConfirmation = false
    @State private var pendingRunPodStop: RunPodPodSummary?
    @State private var remotePackages: [DatasetPackageSummary] = []
    @State private var selectedRemotePackageID: String?
    @State private var remoteNotice: String?
    @State private var isExportingRemotePackage = false
    @State private var remotePreparedArchiveURL: URL?
    @State private var remotePreparedJobID: String?
    @State private var remoteAdapterKind: AdapterKind = .lokr
    @State private var remoteRecipe: RemoteJobRecipe = .wide
    @State private var remoteBundle: RemoteJobBundle = .fullMix
    @State private var remoteBaseModel: RemoteJobBaseModel = .xlSFT
    @State private var remoteAdapterName = ""
    @State private var remoteEpochs = RemoteJobRecipe.wide.settings(for: .lokr).epochs
    @State private var remoteSaveEvery = RemoteJobRecipe.wide.settings(for: .lokr).saveEvery
    @State private var remoteSeed = 42
    @State private var runPodAPIKey = ""
    @AppStorage(AppUserDefaultsKeys.runPodTemplateID) private var runPodTemplateID = ""
    @AppStorage(AppUserDefaultsKeys.runPodAutomaticTemplate) private var runPodAutomaticTemplate = true
    @AppStorage(AppUserDefaultsKeys.runPodImageName) private var runPodImageName = ""
    @AppStorage(AppUserDefaultsKeys.runPodGPUType) private var runPodGPUType = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
    @AppStorage(AppUserDefaultsKeys.runPodRegion) private var runPodRegion: RunPodRegion = .any
    @AppStorage(AppUserDefaultsKeys.runPodInterruptible) private var runPodInterruptible = false
    @AppStorage(AppUserDefaultsKeys.runPodContainerDiskGB) private var runPodContainerDiskGB = 80
    @AppStorage(AppUserDefaultsKeys.runPodVolumeGB) private var runPodVolumeGB = 80
    @AppStorage(AppUserDefaultsKeys.runPodSSHKeyPath) private var runPodSSHKeyPath = "~/.ssh/id_ed25519"
    @AppStorage(AppUserDefaultsKeys.runPodLastArchivePath) private var runPodLastArchivePath = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            header
            Divider()
            trainingModeBar
            Divider()
            workspaceContent
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .alert("Pause or Stop Training?", isPresented: $showingStopConfirmation) {
            Button("Continue Training", role: .cancel) {}
            if trainingService.canSaveOnStop {
                Button("Save & Pause") {
                    trainingService.stopTraining(saveCheckpoint: true)
                }
            }
            Button("Discard & Stop", role: .destructive) {
                trainingService.stopTraining(saveCheckpoint: false)
            }
        } message: {
            if trainingService.canSaveOnStop {
                Text("Training has made progress. Save a checkpoint so you can resume this adapter later.")
            } else {
                Text("No training step has completed yet. Progress will be lost.")
            }
        }
        .alert("Launch RunPod Pod?", isPresented: $showingRunPodLaunchConfirmation) {
            Button("Cancel", role: .cancel) {}
            Button("Launch Pod") {
                launchRunPodPod()
            }
        } message: {
            Text(
                "This creates billable RunPod GPU time in \(runPodRegion.launchDescription). Only continue if you are ready to start a remote server."
            )
        }
        .alert("Terminate RunPod Pod?", isPresented: Binding(
            get: { pendingRunPodStop != nil },
            set: { isPresented in
                if !isPresented {
                    pendingRunPodStop = nil
                }
            }
        )) {
            Button("Cancel", role: .cancel) {
                pendingRunPodStop = nil
            }
            Button("Terminate Pod", role: .destructive) {
                guard let pod = pendingRunPodStop else { return }
                pendingRunPodStop = nil
                Task { await runPodService.terminatePod(id: pod.id) }
            }
        } message: {
            Text("This permanently deletes the RunPod pod and releases its storage billing. Download anything you need from the pod first; data not stored in a network volume can be lost.")
        }
        .onChange(of: outputManager.directory) { _, _ in
            refreshRemotePackages()
        }
        .onChange(of: selectedTrainingMode) { _, mode in
            if mode == .remote || mode == .lora || mode == .lokr {
                refreshRemotePackages()
            }
        }
        .onChange(of: selectedRemotePackageID) { _, _ in
            if !isRemoteBundleAvailable(remoteBundle) {
                remoteBundle = .fullMix
            }
            remoteAdapterName = defaultRemoteAdapterName()
        }
        .onChange(of: remoteAdapterKind) { _, _ in
            applyRemoteRecipeDefaults()
        }
        .onChange(of: remoteRecipe) { _, _ in
            applyRemoteRecipeDefaults()
        }
    }

    private var header: some View {
        HStack {
            Text("Training")
                .font(.headline)
            Spacer()
            Button {
                guard let adapterKind = selectedTrainingMode.localAdapterKind else { return }
                trainingPresentation = TrainingPresentation(adapterKind: adapterKind)
            } label: {
                Label("Train from Files\u{2026}", systemImage: "graduationcap")
            }
            .buttonStyle(.borderedProminent)
            .disabled(selectedTrainingMode.localAdapterKind == nil || !canStartNewTraining || trainingPresentation != nil)
        }
        .padding()
    }

    private var trainingModeBar: some View {
        HStack {
            Picker("Training Mode", selection: $selectedTrainingMode) {
                ForEach(BuildTrainingMode.allCases) { mode in
                    Text(mode.title).tag(mode)
                }
            }
            .pickerStyle(.segmented)
            .frame(maxWidth: 360)
            .disabled(!canStartNewTraining || trainingPresentation != nil)

            Spacer()
        }
        .padding(.horizontal)
        .padding(.vertical, 10)
    }

    @ViewBuilder
    private var workspaceContent: some View {
        switch selectedTrainingMode {
        case .lora:
            loraWorkspace
        case .lokr:
            lokrWorkspace
        case .remote:
            remoteWorkspace
        }
    }

    @ViewBuilder
    private var loraWorkspace: some View {
        if let presentation = trainingPresentation, presentation.adapterKind == .lora {
            TrainingSheet(
                selectedEntries: presentation.selectedEntries,
                initialImportedFiles: presentation.files,
                initialName: presentation.initialName,
                initialConfig: presentation.initialConfig,
                initialLoKRConfig: presentation.initialLoKRConfig,
                adapterKind: presentation.adapterKind,
                resumeAdapterID: presentation.resumeAdapterID,
                presentationStyle: .inline
            ) {
                trainingPresentation = nil
            }
        } else {
            packageTrainingWorkspace(
                adapterKind: .lora,
                actionTitle: "Train LoRA",
                actionSystemImage: "graduationcap",
                emptyMessage: "Build or open a reviewed dataset package before training LoRA adapters.",
                footer: "LoRA uses the reviewed package and `preprocess_cache.json` contract, then trains a local low-rank adapter through the native Swift/MLX loop."
            )
        }
    }

    @ViewBuilder
    private var statusView: some View {
        switch trainingService.state {
        case .idle:
            emptyStatus("Ready", systemImage: "graduationcap")
        case .loadingModels:
            progressStatus(title: "Loading Models", detail: "Preparing ACE-Step training models.")
        case .preparingDataset(let current, let total):
            VStack(spacing: 12) {
                Text("Preparing Dataset")
                    .font(.headline)
                Text("Sample \(current) of \(total)")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                ProgressView(value: Double(current), total: Double(max(total, 1)))
                    .frame(maxWidth: 320)
            }
            .padding()
        case .training(let uiState):
            VStack(spacing: 14) {
                Text("Training")
                    .font(.headline)
                Text("Epoch \(uiState.epoch) of \(uiState.totalEpochs)")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                ProgressView(value: Double(uiState.epoch), total: Double(max(uiState.totalEpochs, 1)))
                    .frame(maxWidth: 320)
                HStack(spacing: 20) {
                    trainingMetric("Loss", String(format: "%.4f", uiState.loss))
                    trainingMetric("LR", String(format: "%.1e", uiState.learningRate))
                    trainingMetric("Elapsed", formatDuration(uiState.elapsed))
                    if let remaining = uiState.remaining {
                        trainingMetric("ETA", "~\(formatDuration(remaining))")
                    }
                }
                Button("Pause / Stop Training", role: .destructive) {
                    showingStopConfirmation = true
                }
                .buttonStyle(.bordered)
            }
            .padding()
        case .saving:
            progressStatus(title: "Saving Adapter", detail: "Writing adapter weights.")
        case .complete(let adapter):
            VStack(spacing: 10) {
                Image(systemName: "checkmark.circle.fill")
                    .font(.largeTitle)
                    .foregroundStyle(.green)
                Text("Training Complete")
                    .font(.headline)
                Text(adapter.name)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
            .padding()
        case .paused:
            emptyStatus("Paused", systemImage: "pause.circle")
        case .failed(let message):
            statusMessage("Training Failed", message: message, systemImage: "exclamationmark.triangle", color: .red)
        case .cancelled:
            emptyStatus("Cancelled", systemImage: "xmark.circle")
        }
    }

    private var canStartNewTraining: Bool {
        switch trainingService.state {
        case .loadingModels, .preparingDataset, .training, .saving:
            return false
        default:
            return true
        }
    }

    private var selectedRemotePackage: DatasetPackageSummary? {
        guard let selectedRemotePackageID else { return nil }
        return remotePackages.first { $0.id == selectedRemotePackageID }
    }

    @ViewBuilder
    private var lokrWorkspace: some View {
        if let presentation = trainingPresentation, presentation.adapterKind == .lokr {
            TrainingSheet(
                selectedEntries: presentation.selectedEntries,
                initialImportedFiles: presentation.files,
                initialName: presentation.initialName,
                initialConfig: presentation.initialConfig,
                initialLoKRConfig: presentation.initialLoKRConfig,
                adapterKind: presentation.adapterKind,
                resumeAdapterID: presentation.resumeAdapterID,
                presentationStyle: .inline
            ) {
                trainingPresentation = nil
            }
        } else {
            packageTrainingWorkspace(
                adapterKind: .lokr,
                actionTitle: "Train LoKR",
                actionSystemImage: "bolt.badge.clock",
                emptyMessage: "Build or open a reviewed dataset package before training LoKR adapters.",
                footer: "Experimental LoKR uses the same reviewed package and `preprocess_cache.json` contract as LoRA, then trains a local Kronecker adapter through the native Swift/MLX loop."
            )
        }
    }

    private func packageTrainingWorkspace(
        adapterKind: AdapterKind,
        actionTitle: String,
        actionSystemImage: String,
        emptyMessage: String,
        footer: String
    ) -> some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 10) {
                Text("\(remotePackages.count) package\(remotePackages.count == 1 ? "" : "s")")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                Spacer()

                Button {
                    refreshRemotePackages()
                } label: {
                    Label("Refresh", systemImage: "arrow.clockwise")
                }
                .buttonStyle(.bordered)
                .controlSize(.small)

                Button {
                    openRemotePackageFolder()
                } label: {
                    Label("Finder", systemImage: "folder")
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .disabled(selectedRemotePackage == nil)

                Button {
                    openSelectedPackageInTraining(adapterKind: adapterKind)
                } label: {
                    Label(actionTitle, systemImage: actionSystemImage)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.small)
                .disabled(selectedRemotePackage == nil || selectedRemotePackage?.isValid == false)
            }
            .padding()

            Divider()

            if remotePackages.isEmpty {
                statusMessage(
                    "No Dataset Packages",
                    message: emptyMessage,
                    systemImage: "tray",
                    color: .secondary
                )
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                List(selection: $selectedRemotePackageID) {
                    ForEach(remotePackages) { package in
                        trainingPackageRow(package)
                            .tag(package.id)
                    }
                }
                .listStyle(.inset)
            }

            Divider()
            Text(footer)
                .font(.caption)
                .foregroundStyle(.secondary)
                .padding()
        }
        .onAppear(perform: refreshRemotePackages)
    }

    private var remoteWorkspace: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 10) {
                Text("\(remotePackages.count) package\(remotePackages.count == 1 ? "" : "s")")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                Spacer()

                Button {
                    refreshRemotePackages()
                } label: {
                    Label("Refresh", systemImage: "arrow.clockwise")
                }
                .buttonStyle(.bordered)
                .controlSize(.small)

                Button {
                    openRemotePackageFolder()
                } label: {
                    Label("Finder", systemImage: "folder")
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .disabled(selectedRemotePackage == nil)

                Button {
                    exportRemotePackage()
                } label: {
                    Label("Create Remote Job\u{2026}", systemImage: "shippingbox")
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.small)
                .disabled(selectedRemotePackage == nil || selectedRemotePackage?.isValid == false || isExportingRemotePackage)
            }
            .padding()

            Divider()

            if remotePackages.isEmpty {
                statusMessage(
                    "No Dataset Packages",
                    message: "Build or open a reviewed dataset package before creating a remote GPU job.",
                    systemImage: "tray",
                    color: .secondary
                )
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                HSplitView {
                    List(selection: $selectedRemotePackageID) {
                        ForEach(remotePackages) { package in
                            remotePackageRow(package)
                                .tag(package.id)
                        }
                    }
                    .listStyle(.inset)
                    .frame(minWidth: 360)

                    remoteJobOptionsPanel
                        .frame(minWidth: 360, idealWidth: 430)
                }
            }

            if let remoteNotice {
                Divider()
                Text(remoteNotice)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
                    .padding()
            }
        }
        .onAppear(perform: refreshRemotePackages)
    }

    private var remoteJobOptionsPanel: some View {
        Form {
            Section("Job") {
                Picker("Adapter", selection: $remoteAdapterKind) {
                    Text("LoKR").tag(AdapterKind.lokr)
                    Text("LoRA").tag(AdapterKind.lora)
                }
                .pickerStyle(.segmented)

                Picker("Recipe", selection: $remoteRecipe) {
                    ForEach(RemoteJobRecipe.allCases) { recipe in
                        Text(recipe.title).tag(recipe)
                    }
                }
                .pickerStyle(.segmented)

                Picker("Bundle", selection: $remoteBundle) {
                    ForEach(RemoteJobBundle.allCases) { bundle in
                        Text(bundle.title)
                            .tag(bundle)
                            .disabled(!isRemoteBundleAvailable(bundle))
                    }
                }
                .pickerStyle(.segmented)

                TextField("Adapter Name", text: $remoteAdapterName)

                Picker("Base Model", selection: $remoteBaseModel) {
                    ForEach(RemoteJobBaseModel.allCases) { model in
                        Text(model.title).tag(model)
                    }
                }
            }

            Section("Run") {
                Stepper(value: $remoteEpochs, in: 1...300, step: 10) {
                    LabeledContent("Epochs", value: "\(remoteEpochs)")
                }
                Stepper(value: $remoteSaveEvery, in: 1...50, step: 1) {
                    LabeledContent("Save Every", value: "\(remoteSaveEvery)")
                }
                Stepper(value: $remoteSeed, in: 0...999_999, step: 1) {
                    LabeledContent("Seed", value: "\(remoteSeed)")
                }
            }

            Section("Recipe Details") {
                let settings = remoteRecipe.settings(for: remoteAdapterKind)
                LabeledContent("Learning Rate", value: String(format: "%.2g", settings.learningRate))
                LabeledContent("Targets", value: settings.targetModules.joined(separator: ", "))
                LabeledContent("Attention", value: settings.attentionType)
                if remoteAdapterKind == .lokr {
                    LabeledContent("Linear Dim", value: "\(settings.lokrLinearDim ?? settings.rank)")
                    LabeledContent("LoKR Alpha", value: "\(Int(settings.lokrLinearAlpha ?? settings.alpha))")
                    LabeledContent("DoRA", value: settings.lokrWeightDecompose ? "On" : "Off")
                } else {
                    LabeledContent("Rank", value: "\(settings.rank)")
                    LabeledContent("Alpha", value: "\(Int(settings.alpha))")
                }
            }

            RunPodLifecyclePanel(
                hasSelectedPackage: selectedRemotePackage != nil,
                preparedArchiveURL: effectiveRemotePreparedArchiveURL,
                apiKey: $runPodAPIKey,
                templateID: $runPodTemplateID,
                autoSelectTemplate: $runPodAutomaticTemplate,
                imageName: $runPodImageName,
                gpuType: $runPodGPUType,
                region: $runPodRegion,
                interruptible: $runPodInterruptible,
                containerDiskGB: $runPodContainerDiskGB,
                volumeGB: $runPodVolumeGB,
                sshKeyPath: $runPodSSHKeyPath,
                onLaunchRequested: { showingRunPodLaunchConfirmation = true },
                onSelectArchiveRequested: selectRemoteJobArchive,
                onUploadRequested: uploadRemoteJob,
                onStartRequested: startRemoteJob,
                onMonitorRequested: monitorRemoteJob,
                onPullRequested: pullAndImportRemoteJob,
                onStopRequested: { pendingRunPodStop = $0 },
                onNotice: { remoteNotice = $0 }
            )

            Section {
                Text("The zip includes the reviewed package, remote_job.json, README_REMOTE_GPU.md, and run_remote_training.sh. Provider secrets are not included.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
        .padding(.vertical, 6)
    }

    private func launchRunPodPod() {
        let configuration = RunPodLaunchConfiguration(
            name: runPodLaunchName(),
            templateID: runPodTemplateID,
            imageName: runPodImageName,
            gpuTypeID: runPodGPUType,
            dataCenterIDs: runPodService.launchDataCenterIDs(
                for: runPodGPUType,
                in: runPodRegion
            ),
            interruptible: runPodInterruptible,
            containerDiskGB: runPodContainerDiskGB,
            volumeGB: runPodVolumeGB
        )

        Task {
            await runPodService.launchPod(configuration: configuration)
        }
    }

    private func uploadRemoteJob(to pod: RunPodPodSummary) {
        guard let archiveURL = effectiveRemotePreparedArchiveURL else {
            remoteNotice = "Create or choose a remote job zip before uploading."
            return
        }
        Task {
            await runPodService.uploadRemoteJob(
                to: pod,
                archiveURL: archiveURL,
                sshKeyPath: runPodSSHKeyPath
            )
        }
    }

    private func startRemoteJob(on pod: RunPodPodSummary) {
        guard let archiveURL = effectiveRemotePreparedArchiveURL else {
            remoteNotice = "Create or choose a remote job zip before starting the worker."
            return
        }
        Task {
            await runPodService.startRemoteJob(
                on: pod,
                archiveURL: archiveURL,
                sshKeyPath: runPodSSHKeyPath
            )
        }
    }

    private func monitorRemoteJob(on pod: RunPodPodSummary) {
        guard let archiveURL = effectiveRemotePreparedArchiveURL else {
            remoteNotice = "Choose the remote job zip associated with this worker."
            return
        }
        Task {
            await runPodService.refreshRemoteJob(
                on: pod,
                archiveURL: archiveURL,
                sshKeyPath: runPodSSHKeyPath
            )
        }
    }

    private func pullAndImportRemoteJob(from pod: RunPodPodSummary) {
        guard let archiveURL = effectiveRemotePreparedArchiveURL else {
            remoteNotice = "Choose the completed remote job zip before pulling artifacts."
            return
        }
        guard let destinationParentURL = chooseRemoteResultsDirectory(
            suggestedBy: archiveURL
        ) else {
            return
        }
        let didStartAccessing = destinationParentURL
            .startAccessingSecurityScopedResource()

        Task {
            defer {
                if didStartAccessing {
                    destinationParentURL.stopAccessingSecurityScopedResource()
                }
            }

            do {
                let pullback = try await runPodService.downloadRemoteJobArtifacts(
                    from: pod,
                    archiveURL: archiveURL,
                    sshKeyPath: runPodSSHKeyPath,
                    destinationParentURL: destinationParentURL
                )
                let importer = CommunityService()
                let adapter = try await importer.importLocalAdapter(
                    directoryURL: pullback.finalAdapterURL,
                    preferredName: pullback.result.adapterName
                )
                runPodService.markRemoteAdapterImported(
                    podID: pod.id,
                    archiveURL: archiveURL,
                    adapterName: adapter.name
                )
                remoteNotice = "Downloaded and imported \(adapter.name). The remote result is preserved at \(pullback.directoryURL.path(percentEncoded: false))."
                pendingRunPodStop = pod
            } catch {
                remoteNotice = error.localizedDescription
            }
        }
    }

    private func chooseRemoteResultsDirectory(suggestedBy archiveURL: URL) -> URL? {
        let panel = NSOpenPanel()
        panel.title = "Choose Remote Training Results Location"
        panel.message = "Choose where PiMP Audio Studio should create its Remote Training Results folder."
        panel.prompt = "Download Here"
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.canCreateDirectories = true
        panel.allowsMultipleSelection = false
        panel.directoryURL = archiveURL.deletingLastPathComponent()

        return panel.runModal() == .OK ? panel.url : nil
    }

    private func runPodLaunchName() -> String {
        let base = normalizedRemoteAdapterName()
            .lowercased()
            .replacingOccurrences(of: "[^a-z0-9\\-]+", with: "-", options: .regularExpression)
            .trimmingCharacters(in: CharacterSet(charactersIn: "-"))
        let suffix = remotePreparedJobID?.prefix(8) ?? UUID().uuidString.prefix(8)
        return String("pimp-\(base)-\(suffix)".prefix(120))
    }

    private func remotePackageRow(_ package: DatasetPackageSummary) -> some View {
        HStack(spacing: 10) {
            Image(systemName: package.isValid ? "checkmark.circle.fill" : "exclamationmark.triangle.fill")
                .foregroundStyle(package.isValid ? .green : .orange)
                .frame(width: 18)
            VStack(alignment: .leading, spacing: 3) {
                Text(package.name)
                    .font(.callout)
                    .fontWeight(.medium)
                    .lineLimit(1)
                    .truncationMode(.middle)
                HStack(spacing: 8) {
                    Text("\(package.clipCount) clips")
                    Text(package.enrichmentStatus)
                    if package.transcriptionStatus != "not_run" {
                        Text("lyrics \(package.transcriptionStatus)")
                    }
                    if package.bpmStatus != "not_run" {
                        Text("bpm \(package.bpmStatus)")
                    }
                    if package.keyStatus != "not_run" {
                        Text("key \(package.keyStatus)")
                    }
                    if package.timeSignatureStatus != "not_run" {
                        Text("meter \(package.timeSignatureStatus)")
                    }
                }
                .font(.caption2)
                .foregroundStyle(.secondary)
            }
            Spacer()
            Text(package.isValid ? "Ready" : "Review")
                .font(.caption2)
                .foregroundStyle(package.isValid ? .green : .orange)
        }
        .padding(.vertical, 4)
    }

    private func trainingPackageRow(_ package: DatasetPackageSummary) -> some View {
        HStack(spacing: 10) {
            Image(systemName: package.isValid ? "checkmark.circle.fill" : "exclamationmark.triangle.fill")
                .foregroundStyle(package.isValid ? .green : .orange)
                .frame(width: 18)
            VStack(alignment: .leading, spacing: 3) {
                Text(package.name)
                    .font(.callout)
                    .fontWeight(.medium)
                    .lineLimit(1)
                    .truncationMode(.middle)
                HStack(spacing: 8) {
                    Text("\(package.clipCount) clips")
                    Text("cache \(preprocessCacheStatus(for: package))")
                    if package.keyStatus != "not_run" {
                        Text("key \(package.keyStatus)")
                    }
                    if package.timeSignatureStatus != "not_run" {
                        Text("meter \(package.timeSignatureStatus)")
                    }
                }
                .font(.caption2)
                .foregroundStyle(.secondary)
            }
            Spacer()
            Text(package.isValid ? "Ready" : "Review")
                .font(.caption2)
                .foregroundStyle(package.isValid ? .green : .orange)
        }
        .padding(.vertical, 4)
    }

    private func preprocessCacheStatus(for package: DatasetPackageSummary) -> String {
        let url = package.directory.appendingPathComponent(TrainingDatasetPackageWriter.preprocessCacheFileName)
        guard let data = try? Data(contentsOf: url),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let status = json["status"] as? String else {
            return "missing"
        }
        return status
    }

    private func refreshRemotePackages() {
        guard let outputDirectory = outputManager.directory else {
            remotePackages = []
            selectedRemotePackageID = nil
            return
        }

        remotePackages = TrainingDatasetPackageEditor.listPackages(in: outputDirectory)
        if let selectedRemotePackageID,
           remotePackages.contains(where: { $0.id == selectedRemotePackageID }) {
            return
        }
        selectedRemotePackageID = remotePackages.first?.id
        if remoteAdapterName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            remoteAdapterName = defaultRemoteAdapterName()
        }
    }

    private func openRemotePackageFolder() {
        guard let package = selectedRemotePackage else { return }
        NSWorkspace.shared.open(package.directory)
    }

    private var effectiveRemotePreparedArchiveURL: URL? {
        if let remotePreparedArchiveURL,
           FileManager.default.fileExists(atPath: remotePreparedArchiveURL.path(percentEncoded: false)) {
            return remotePreparedArchiveURL
        }

        let trimmedPath = runPodLastArchivePath.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedPath.isEmpty else { return nil }
        let url = URL(fileURLWithPath: NSString(string: trimmedPath).expandingTildeInPath)
        guard FileManager.default.fileExists(atPath: url.path(percentEncoded: false)) else { return nil }
        return url
    }

    private func rememberRemoteArchive(_ url: URL) {
        remotePreparedArchiveURL = url
        runPodLastArchivePath = url.path(percentEncoded: false)
    }

    private func selectRemoteJobArchive() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.zip]
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        panel.prompt = "Choose Job Zip"
        panel.message = "Choose an exported PiMP Audio Studio remote job zip to upload to the active pod."
        if let effectiveRemotePreparedArchiveURL {
            panel.directoryURL = effectiveRemotePreparedArchiveURL.deletingLastPathComponent()
        }
        guard panel.runModal() == .OK, let url = panel.url else { return }
        rememberRemoteArchive(url)
        remotePreparedJobID = nil
        remoteNotice = "Remote job zip selected: \(url.path(percentEncoded: false))."
    }

    private func openSelectedPackageInTraining(adapterKind: AdapterKind) {
        guard let package = selectedRemotePackage else { return }
        let bundleURL = package.directory.appendingPathComponent(TrainingDatasetPackageWriter.fullMixBundleFileName)
        do {
            let imported = try TrainingDatasetImporter.importFiles(from: bundleURL)
            let baseName = package.name.replacingOccurrences(of: "_local_", with: " ")
            trainingPresentation = TrainingPresentation(
                files: imported,
                adapterKind: adapterKind,
                initialName: "\(baseName) \(adapterTitle(for: adapterKind))"
            )
        } catch {
            remoteNotice = error.localizedDescription
        }
    }

    private func adapterTitle(for adapterKind: AdapterKind) -> String {
        switch adapterKind {
        case .lora:
            return "LoRA"
        case .lokr:
            return "LoKR"
        }
    }

    private func exportRemotePackage() {
        guard let package = selectedRemotePackage else { return }

        do {
            let document = try TrainingDatasetPackageEditor.loadPackage(at: package.directory)
            guard document.isTrainingReady else {
                remoteNotice = "Package needs dataset review before export: \(document.blockerCount) blocker\(document.blockerCount == 1 ? "" : "s")."
                return
            }
            guard isRemoteBundleAvailable(remoteBundle) else {
                remoteNotice = "\(remoteBundle.title) is not available for this package. Run stem enrichment first or choose Full Mix."
                return
            }

            let descriptor = try remotePackageDescriptor(for: package)
            let settings = remoteRecipe.settings(for: remoteAdapterKind)
            let adapterName = normalizedRemoteAdapterName()
            let job = RemoteTrainingJobRequest.makeTemplate(
                from: descriptor,
                task: remoteAdapterKind == .lokr ? .trainLoKR : .trainLoRA,
                selectedBundleName: remoteBundle.rawValue,
                adapterName: adapterName,
                baseModelID: remoteBaseModel.appModelID,
                baseModelLabel: remoteBaseModel.title,
                workerModelVariant: remoteBaseModel.workerVariant,
                recipe: remoteRecipe.rawValue,
                rank: remoteAdapterKind == .lora ? settings.rank : nil,
                alpha: remoteAdapterKind == .lora ? settings.alpha : nil,
                epochs: remoteEpochs,
                saveEvery: remoteSaveEvery,
                batchSize: settings.batchSize,
                gradientAccumulationSteps: settings.gradientAccumulationSteps,
                learningRate: settings.learningRate,
                dropout: settings.dropout,
                targetModules: settings.targetModules,
                attentionType: settings.attentionType,
                cfgRatio: settings.cfgRatio,
                seed: remoteSeed,
                maxDurationSeconds: settings.maxDurationSeconds,
                lokrLinearDim: remoteAdapterKind == .lokr ? settings.lokrLinearDim : nil,
                lokrLinearAlpha: remoteAdapterKind == .lokr ? settings.lokrLinearAlpha : nil,
                lokrWeightDecompose: remoteAdapterKind == .lokr ? settings.lokrWeightDecompose : nil
            )
            let validation = try RemoteTrainingJobValidator.validate(
                job,
                packageDirectory: package.directory
            )
            let extraFiles = try RemoteTrainingJobCodec.archiveExtraFiles(for: job)

            let panel = NSSavePanel()
            panel.allowedContentTypes = [.zip]
            panel.canCreateDirectories = true
            panel.nameFieldStringValue = "\(package.directory.lastPathComponent)_remote_\(remoteAdapterKind.rawValue)_\(remoteRecipe.rawValue).zip"
            guard panel.runModal() == .OK, let destination = panel.url else { return }

            isExportingRemotePackage = true
            Task {
                defer { isExportingRemotePackage = false }
                do {
                    let archiveURL = try await TrainingDatasetPackageWriter.exportPackageArchive(
                        outputDirectory: package.directory,
                        to: destination,
                        extraFiles: extraFiles
                    )
                    rememberRemoteArchive(archiveURL)
                    remotePreparedJobID = job.jobID
                    remoteNotice = "Validated \(validation.selectedAssetCount) package-v2 asset\(validation.selectedAssetCount == 1 ? "" : "s") for \(validation.assetRoles.joined(separator: ", ")); remote job exported to \(archiveURL.path(percentEncoded: false))."
                    refreshRemotePackages()
                } catch is CancellationError {
                    remoteNotice = "Remote job export was cancelled."
                } catch {
                    remoteNotice = error.localizedDescription
                }
            }
        } catch {
            remoteNotice = error.localizedDescription
        }
    }

    private func remotePackageDescriptor(for package: DatasetPackageSummary) throws -> RemoteTrainingPackageDescriptor {
        let descriptorURL = package.directory.appendingPathComponent(TrainingDatasetPackageWriter.remotePackageFileName)
        let data = try Data(contentsOf: descriptorURL)
        return try RemoteTrainingJobCodec.decoder.decode(RemoteTrainingPackageDescriptor.self, from: data)
    }

    private func isRemoteBundleAvailable(_ bundle: RemoteJobBundle) -> Bool {
        guard let package = selectedRemotePackage else { return bundle == .fullMix }
        return FileManager.default.fileExists(
            atPath: package.directory.appendingPathComponent(bundle.fileName).path(percentEncoded: false)
        )
    }

    private func applyRemoteRecipeDefaults() {
        let settings = remoteRecipe.settings(for: remoteAdapterKind)
        remoteEpochs = settings.epochs
        remoteSaveEvery = settings.saveEvery
        if remoteAdapterName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ||
            remoteAdapterName.contains(" LoRA ") ||
            remoteAdapterName.contains(" LoKR ") {
            remoteAdapterName = defaultRemoteAdapterName()
        }
    }

    private func normalizedRemoteAdapterName() -> String {
        let trimmed = remoteAdapterName.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? defaultRemoteAdapterName() : trimmed
    }

    private func defaultRemoteAdapterName() -> String {
        let packageName = selectedRemotePackage?.name.replacingOccurrences(of: "_local_", with: " ") ?? "Remote Adapter"
        return "\(packageName) \(adapterTitle(for: remoteAdapterKind)) \(remoteRecipe.title)"
    }

    private func progressStatus(title: String, detail: String) -> some View {
        VStack(spacing: 10) {
            ProgressView()
                .controlSize(.large)
            Text(title)
                .font(.headline)
            Text(detail)
                .font(.subheadline)
                .foregroundStyle(.secondary)
        }
        .padding()
    }

    private func emptyStatus(_ title: String, systemImage: String) -> some View {
        statusMessage(title, message: nil, systemImage: systemImage, color: .secondary)
    }

    private func statusMessage(
        _ title: String,
        message: String?,
        systemImage: String,
        color: Color
    ) -> some View {
        VStack(spacing: 8) {
            Image(systemName: systemImage)
                .font(.largeTitle)
                .foregroundStyle(color)
            Text(title)
                .font(.headline)
            if let message {
                Text(message)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal)
            }
        }
        .padding()
    }

    private func trainingMetric(_ label: String, _ value: String) -> some View {
        VStack(spacing: 2) {
            Text(value)
                .font(.caption)
                .monospacedDigit()
            Text(label)
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
    }

    private func formatDuration(_ interval: TimeInterval) -> String {
        let total = max(0, Int(interval))
        let minutes = total / 60
        let seconds = total % 60
        if minutes >= 60 {
            let hours = minutes / 60
            let remainingMinutes = minutes % 60
            return "\(hours)h \(remainingMinutes)m"
        }
        return "\(minutes)m \(seconds)s"
    }
}

private enum BuildTrainingMode: String, CaseIterable, Identifiable {
    case lora
    case lokr
    case remote

    var id: String { rawValue }

    var title: String {
        switch self {
        case .lora: return "LoRA"
        case .lokr: return "LoKR"
        case .remote: return "Remote GPU"
        }
    }

    var localAdapterKind: AdapterKind? {
        switch self {
        case .lora:
            return .lora
        case .lokr:
            return .lokr
        case .remote:
            return nil
        }
    }
}

private enum RemoteJobRecipe: String, CaseIterable, Identifiable {
    case narrow
    case wide

    var id: String { rawValue }

    var title: String {
        switch self {
        case .narrow: return "Narrow"
        case .wide: return "Wide"
        }
    }

    func settings(for adapterKind: AdapterKind) -> RemoteJobRecipeSettings {
        switch (adapterKind, self) {
        case (.lora, .narrow):
            return RemoteJobRecipeSettings(
                rank: 64,
                alpha: 64,
                lokrLinearDim: nil,
                lokrLinearAlpha: nil,
                lokrWeightDecompose: false,
                epochs: 100,
                saveEvery: 10,
                batchSize: 1,
                gradientAccumulationSteps: 2,
                learningRate: 0.00008,
                dropout: 0,
                targetModules: ["q_proj", "v_proj"],
                attentionType: "both",
                cfgRatio: 0.15,
                maxDurationSeconds: 240
            )
        case (.lora, .wide):
            return RemoteJobRecipeSettings(
                rank: 96,
                alpha: 192,
                lokrLinearDim: nil,
                lokrLinearAlpha: nil,
                lokrWeightDecompose: false,
                epochs: 150,
                saveEvery: 10,
                batchSize: 1,
                gradientAccumulationSteps: 2,
                learningRate: 0.000065,
                dropout: 0,
                targetModules: ["q_proj", "k_proj", "v_proj", "o_proj"],
                attentionType: "both",
                cfgRatio: 0.15,
                maxDurationSeconds: 240
            )
        case (.lokr, .narrow):
            return RemoteJobRecipeSettings(
                rank: 32,
                alpha: 64,
                lokrLinearDim: 32,
                lokrLinearAlpha: 64,
                lokrWeightDecompose: true,
                epochs: 80,
                saveEvery: 10,
                batchSize: 1,
                gradientAccumulationSteps: 2,
                learningRate: 0.00008,
                dropout: 0,
                targetModules: ["q_proj", "v_proj"],
                attentionType: "both",
                cfgRatio: 0.15,
                maxDurationSeconds: 240
            )
        case (.lokr, .wide):
            return RemoteJobRecipeSettings(
                rank: 128,
                alpha: 256,
                lokrLinearDim: 128,
                lokrLinearAlpha: 256,
                lokrWeightDecompose: true,
                epochs: 150,
                saveEvery: 10,
                batchSize: 1,
                gradientAccumulationSteps: 2,
                learningRate: 0.000065,
                dropout: 0,
                targetModules: ["q_proj", "k_proj", "v_proj", "o_proj"],
                attentionType: "both",
                cfgRatio: 0.15,
                maxDurationSeconds: 240
            )
        }
    }
}

private struct RemoteJobRecipeSettings {
    let rank: Int
    let alpha: Double
    let lokrLinearDim: Int?
    let lokrLinearAlpha: Double?
    let lokrWeightDecompose: Bool
    let epochs: Int
    let saveEvery: Int
    let batchSize: Int
    let gradientAccumulationSteps: Int
    let learningRate: Double
    let dropout: Double
    let targetModules: [String]
    let attentionType: String
    let cfgRatio: Double
    let maxDurationSeconds: Int
}

private enum RemoteJobBundle: String, CaseIterable, Identifiable {
    case fullMix = "full_mix"
    case instrumental
    case vocals

    var id: String { rawValue }

    var title: String {
        switch self {
        case .fullMix: return "Full Mix"
        case .instrumental: return "Instrumental"
        case .vocals: return "Vocals"
        }
    }

    var fileName: String {
        switch self {
        case .fullMix:
            return TrainingDatasetPackageWriter.fullMixBundleFileName
        case .instrumental:
            return TrainingDatasetPackageWriter.instrumentalBundleFileName
        case .vocals:
            return TrainingDatasetPackageWriter.vocalBundleFileName
        }
    }
}

private enum RemoteJobBaseModel: String, CaseIterable, Identifiable {
    case xlSFT
    case xlBase
    case xlTurbo

    var id: String { rawValue }

    var title: String {
        switch self {
        case .xlSFT: return "ACE-Step XL SFT"
        case .xlBase: return "ACE-Step XL Base"
        case .xlTurbo: return "ACE-Step XL Turbo"
        }
    }

    var appModelID: String {
        switch self {
        case .xlSFT: return "ace-step-dit-xl-sft"
        case .xlBase: return "ace-step-dit-xl-base"
        case .xlTurbo: return "ace-step-dit-xl-turbo"
        }
    }

    var workerVariant: String {
        switch self {
        case .xlSFT: return "ace-step-xl-sft"
        case .xlBase: return "ace-step-xl-base"
        case .xlTurbo: return "ace-step-xl-turbo"
        }
    }
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
        ditSpecs
            .filter { ditStatuses[$0.id] == .ready }
            .sorted(by: modelTierSort)
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
        .onReceive(NotificationCenter.default.publisher(for: .modelRegistryDidChange)) { _ in
            Task { await loadModelInfo() }
        }
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
        case "ace-step-dit-turbo":    return "Turbo  –  8 steps"
        case "ace-step-dit-base":     return "Full  –  50 steps"
        case "ace-step-dit-sft":      return "SFT  –  50 steps"
        case "ace-step-dit-xl-turbo": return "XL Turbo  –  8 steps"
        case "ace-step-dit-xl-base":  return "XL Base  –  50 steps"
        case "ace-step-dit-xl-sft":   return "XL SFT  –  50 steps"
        default:
            return spec.id.contains("xl") ? "XL  –  50 steps" : spec.displayName
        }
    }

    private func modelTierSort(_ lhs: ModelSpec, _ rhs: ModelSpec) -> Bool {
        let lhsRank = modelTierRank(lhs.id)
        let rhsRank = modelTierRank(rhs.id)
        if lhsRank != rhsRank {
            return lhsRank < rhsRank
        }
        return lhs.displayName.localizedStandardCompare(rhs.displayName) == .orderedAscending
    }

    private func modelTierRank(_ id: String) -> Int {
        if id.contains("turbo") { return 0 }
        if id.contains("base") { return 1 }
        if id.contains("sft") { return 2 }
        return 50
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
            await MainActor.run {
                NotificationCenter.default.post(name: .modelRegistryDidChange, object: nil)
            }
            await loadModelInfo()
            isSwitching = false
        }
    }
}

extension Notification.Name {
    static let modelRegistryDidChange = Notification.Name("modelRegistryDidChange")
}
