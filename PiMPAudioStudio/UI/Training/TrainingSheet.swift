import SwiftUI
import PiMPAudioCore
import UniformTypeIdentifiers
import AppKit
import AVFoundation

/// Drives `.sheet(item:)` for any training sheet presentation.
/// Carrying all data as value types ensures the sheet closure sees the
/// correct dataset regardless of when SwiftUI evaluates it.
struct TrainingPresentation: Identifiable {
    let id = UUID()
    /// History entries resolved from the manifest (may be empty for new sessions).
    let selectedEntries: [HistoryEntry]
    /// Imported file entries — from manifest, YouTube, or manual selection.
    let files: [TrainingImportedFile]
    /// Pre-fill the adapter name field (empty = blank for new sessions).
    let initialName: String
    /// Pre-fill training config (nil = use defaults for new sessions).
    let initialConfig: LoRAConfig?
    /// Pre-fill LoKR-only training config (nil = derive from initialConfig/defaults).
    let initialLoKRConfig: LoKRConfig?
    /// Adapter implementation for this training session.
    let adapterKind: AdapterKind
    /// Adapter ID to resume from a saved checkpoint. Nil means fresh training/retraining.
    let resumeAdapterID: String?

    /// Blank new training session.
    init(files: [TrainingImportedFile] = [], adapterKind: AdapterKind = .lora, initialName: String = "") {
        self.selectedEntries = []
        self.files = files
        self.initialName = initialName
        self.initialConfig = nil
        self.initialLoKRConfig = nil
        self.adapterKind = adapterKind
        self.resumeAdapterID = nil
    }

    /// Pre-populated retrain session.
    init(
        selectedEntries: [HistoryEntry],
        files: [TrainingImportedFile],
        initialName: String,
        initialConfig: LoRAConfig?,
        initialLoKRConfig: LoKRConfig? = nil,
        adapterKind: AdapterKind = .lora,
        resumeAdapterID: String? = nil
    ) {
        self.selectedEntries = selectedEntries
        self.files = files
        self.initialName = initialName
        self.initialConfig = initialConfig
        self.initialLoKRConfig = initialLoKRConfig
        self.adapterKind = adapterKind
        self.resumeAdapterID = resumeAdapterID
    }
}

enum TrainingSurfacePresentationStyle {
    case sheet
    case inline
}

struct TrainingSheet: View {
    @EnvironmentObject private var trainingService: TrainingService
    @EnvironmentObject private var generationService: GenerationService
    @EnvironmentObject private var traceMonitor: GenerationTraceMonitor
    @EnvironmentObject private var historyStore: HistoryStore
    @EnvironmentObject private var presetStore: PresetStore

    let selectedEntries: [HistoryEntry]

    @Environment(\.dismiss) private var dismiss
    @Environment(\.openWindow) private var openWindow

    private let presentationStyle: TrainingSurfacePresentationStyle
    private let onClose: (() -> Void)?
    private let adapterKind: AdapterKind

    @State private var adapterName: String = ""
    private let resumeAdapterID: String?
    @State private var selectedRecipe: TrainingRecipePreset = .balancedLocal
    @State private var selectedRank: Int = 16
    @State private var selectedEpochs: Int = 10
    @State private var selectedLayerCount: Int = 8
    @State private var selectedProjectionCoverage: TrainingProjectionCoverage = .queryValue
    @State private var selectedAlphaScale: TrainingAlphaScale = .rank
    @State private var selectedDropout: TrainingDropoutPreset = .none
    @State private var selectedTargetLayers: Set<TargetLayer> = [.selfAttention, .crossAttention]
    @State private var warmupFraction: Float = 0.10
    @State private var gradientAccumulationSteps: Int = 1
    @State private var trainingSeedText: String = "\(LoRAConfig.defaultTrainingSeed)"
    @State private var selectedLoKRWeightDecompose: Bool = true
    @State private var importedFiles: [TrainingImportedFile]
    @State private var showingFileImporter: Bool = false
    @State private var showingStopConfirmation: Bool = false
    @State private var baseModelDisplayName: String = "ACE-Step DiT"
    @State private var importErrorMessage: String?
    @State private var importStatusMessage: String?
    @State private var completionStatusMessage: String?
    @State private var importedFileDurations: [String: TimeInterval] = [:]
    @State private var auditionIncludeBase: Bool = true
    @State private var auditionSeedText: String = "42"
    @State private var auditionDurationText: String = "30"
    @State private var auditionStrengthsText: String = "1.0, 1.6, 2.0"
    @State private var auditionTask: Task<Void, Never>?
    @State private var auditionProgressMessage: String?
    @State private var auditionProgressFraction: Double?
    @State private var isAuditioning: Bool = false

    init(
        selectedEntries: [HistoryEntry],
        initialImportedFiles: [TrainingImportedFile] = [],
        initialName: String = "",
        initialConfig: LoRAConfig? = nil,
        initialLoKRConfig: LoKRConfig? = nil,
        adapterKind: AdapterKind = .lora,
        resumeAdapterID: String? = nil,
        presentationStyle: TrainingSurfacePresentationStyle = .sheet,
        onClose: (() -> Void)? = nil
    ) {
        self.selectedEntries = selectedEntries
        self.resumeAdapterID = resumeAdapterID
        self.presentationStyle = presentationStyle
        self.onClose = onClose
        self.adapterKind = adapterKind
        self._importedFiles = State(initialValue: initialImportedFiles)
        self._adapterName = State(initialValue: initialName)
        let effectiveInitialConfig = initialLoKRConfig?.loraCompatibleConfig ?? initialConfig
        if let config = effectiveInitialConfig {
            self._selectedRecipe = State(initialValue: TrainingRecipePreset.infer(from: config, adapterKind: adapterKind))
            self._selectedRank = State(initialValue: config.rank)
            self._selectedEpochs = State(initialValue: config.maxEpochs)
            self._selectedLayerCount = State(initialValue: config.trainingLayerCount ?? (resumeAdapterID == nil ? 8 : 4))
            self._selectedProjectionCoverage = State(initialValue: TrainingProjectionCoverage.infer(from: config))
            self._selectedAlphaScale = State(initialValue: TrainingAlphaScale.infer(from: config.alpha, dimension: config.rank))
            self._selectedDropout = State(initialValue: TrainingDropoutPreset.closest(to: config.dropout))
            self._selectedTargetLayers = State(initialValue: config.targetLayers)
            self._warmupFraction = State(initialValue: config.warmupFraction)
            self._gradientAccumulationSteps = State(initialValue: config.gradientAccumulationSteps)
            self._trainingSeedText = State(initialValue: "\(config.effectiveTrainingSeed)")
            self._selectedLoKRWeightDecompose = State(initialValue: initialLoKRConfig?.weightDecompose ?? false)
        } else if adapterKind == .lokr {
            self._selectedLoKRWeightDecompose = State(initialValue: true)
        }
    }

    private var rankOptions: [Int] {
        adapterKind == .lokr ? [16, 32, 64, 128] : [8, 16, 32, 64, 96]
    }
    private let epochOptions = [5, 10, 20, 50, 75, 100, 150]
    private let layerCountOptions = [4, 8, 12, 16, 32]
    private let accumOptions = [1, 2, 4, 8]
    private let maxLocalLoKRTrainableTensorCount = 512

    private var recipeOptions: [TrainingRecipePreset] {
        TrainingRecipePreset.options(for: adapterKind)
    }

    var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                Text("Train \(adapterKind.title) Adapter")
                    .font(.headline)
                Spacer()
                Button("Close") { close() }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
            }
            .padding()

            Divider()

            // Content based on state
            ScrollView {
                switch trainingService.state {
                case .idle, .failed, .cancelled:
                    configView
                case .paused:
                    configView
                case .loadingModels, .preparingDataset, .training, .saving:
                    progressView
                case .complete(let adapter):
                    completionView(adapter: adapter)
                }
            }
        }
        .frame(
            minWidth: presentationStyle == .sheet ? 500 : 0,
            minHeight: presentationStyle == .sheet ? 500 : 0
        )
        .frame(
            maxWidth: presentationStyle == .inline ? .infinity : nil,
            maxHeight: presentationStyle == .inline ? .infinity : nil,
            alignment: .topLeading
        )
        .onAppear {
            trainingService.resetToIdle()
        }
        .task {
            await refreshBaseModelDisplayName()
        }
        .task(id: importedDurationRefreshKey) {
            await refreshImportedFileDurations()
        }
        .onChange(of: selectedRecipe) { _, newRecipe in
            applyRecipe(newRecipe)
        }
        .onReceive(NotificationCenter.default.publisher(for: .modelRegistryDidChange)) { _ in
            Task { await refreshBaseModelDisplayName() }
        }
        .fileImporter(
            isPresented: $showingFileImporter,
            allowedContentTypes: [.audio, .json],
            allowsMultipleSelection: true
        ) { result in
            if case .success(let urls) = result {
                importSelectedFiles(urls)
            } else if case .failure(let error) = result {
                importErrorMessage = error.localizedDescription
            }
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
    }

    // MARK: - Config View

    private var configView: some View {
        Group {
            if presentationStyle == .inline {
                inlineConfigView
            } else {
                sheetConfigView
            }
        }
    }

    private var sheetConfigView: some View {
        VStack(alignment: .leading, spacing: 16) {
            resumeBanner
            trainingDataSection
            configurationSection
            errorDisplay
            startButtonRow
        }
        .padding()
    }

    private var inlineConfigView: some View {
        VStack(alignment: .leading, spacing: 20) {
            resumeBanner

            HStack(alignment: .top, spacing: 32) {
                trainingDataSection
                    .frame(minWidth: 280, idealWidth: 360, maxWidth: 420, alignment: .topLeading)

                VStack(alignment: .leading, spacing: 16) {
                    configurationSection
                    errorDisplay
                    startButtonRow
                }
                .frame(minWidth: 360, idealWidth: 460, maxWidth: 560, alignment: .topLeading)
            }
            .frame(maxWidth: .infinity, alignment: .topLeading)
        }
        .padding(24)
    }

    @ViewBuilder
    private var resumeBanner: some View {
        if isCheckpointResume {
            HStack {
                Image(systemName: "pause.circle.fill")
                    .foregroundStyle(.orange)
                Text("Training paused. Resume from the saved checkpoint, or start fresh from Retrain.")
                    .font(.callout)
                Spacer()
            }
            .padding(8)
            .background(.orange.opacity(0.1))
            .clipShape(RoundedRectangle(cornerRadius: 8))
        }
    }

    private var trainingDataSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Training Data")
                .font(.subheadline)
                .fontWeight(.semibold)

            VStack(spacing: 4) {
                ForEach(selectedEntries) { entry in
                    HStack {
                        Image(systemName: "waveform")
                            .foregroundStyle(.secondary)
                        Text(entry.prompt)
                            .lineLimit(1)
                            .font(.callout)
                        Spacer()
                        Text("\(Int(entry.duration))s")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
                ForEach(importedFiles.indices, id: \.self) { idx in
                    HStack(alignment: .top, spacing: 8) {
                        Image(systemName: "doc.badge.plus")
                            .foregroundStyle(.secondary)
                            .padding(.top, 6)
                        VStack(alignment: .leading, spacing: 6) {
                            Text(importedFiles[idx].url.lastPathComponent)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .lineLimit(1)
                                .truncationMode(.middle)
                            TextField("Description", text: Binding(
                                get: { importedFiles[idx].description },
                                set: { importedFiles[idx].description = $0 }
                            ))
                            .font(.callout)
                            .textFieldStyle(.roundedBorder)
                            .disabled(isCheckpointResume)
                            if !importedFiles[idx].lyrics.isEmpty {
                                TextField("Lyrics", text: Binding(
                                    get: { importedFiles[idx].lyrics },
                                    set: { importedFiles[idx].lyrics = $0 }
                                ))
                                .font(.caption)
                                .textFieldStyle(.roundedBorder)
                                .disabled(isCheckpointResume)
                            }
                        }
                        Button {
                            importedFiles.remove(at: idx)
                        } label: {
                            Image(systemName: "xmark.circle")
                                .foregroundStyle(.secondary)
                        }
                        .buttonStyle(.borderless)
                        .disabled(isCheckpointResume)
                        .padding(.top, 6)
                    }
                }
            }
            .padding(8)
            .background(.quaternary.opacity(0.3))
            .clipShape(RoundedRectangle(cornerRadius: 8))

            HStack {
                let totalCount = selectedEntries.count + importedFiles.count
                Text("\(totalCount) tracks selected")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
                Button("+ Add audio files or JSON...") {
                    showingFileImporter = true
                }
                .buttonStyle(.borderless)
                .font(.caption)
                .disabled(isCheckpointResume)
            }
            if let importStatusMessage {
                Label(importStatusMessage, systemImage: "checkmark.circle")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            if let importErrorMessage {
                Label(importErrorMessage, systemImage: "exclamationmark.triangle")
                    .font(.caption)
                    .foregroundStyle(.orange)
            }
        }
    }

    private var configurationSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Configuration")
                .font(.subheadline)
                .fontWeight(.semibold)

            LabeledContent("Adapter Name") {
                TextField("e.g. My Jazz Style", text: $adapterName)
                    .textFieldStyle(.roundedBorder)
                    .frame(maxWidth: 240)
                    .disabled(isCheckpointResume)
            }
            LabeledContent("Base Model") {
                Text(baseModelDisplayName)
                    .foregroundStyle(.secondary)
            }
            LabeledContent("Recipe") {
                Picker("Recipe", selection: $selectedRecipe) {
                    ForEach(recipeOptions, id: \.self) { recipe in
                        Text(recipe.displayName(for: adapterKind)).tag(recipe)
                    }
                }
                .labelsHidden()
                .pickerStyle(.segmented)
                .frame(maxWidth: 220)
                .disabled(isCheckpointResume)
            }
            if selectedRecipe != .balancedLocal {
                Text(selectedRecipe.summary(for: adapterKind))
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(.vertical, 2)
            }
            if selectedRecipe == .balancedLocal, let recommendation = trainingRecommendation {
                trainingRecommendationView(recommendation)
            }
            LabeledContent(adapterKind == .lokr ? "Linear Dim" : "Rank") {
                Picker(adapterKind == .lokr ? "Linear Dim" : "Rank", selection: $selectedRank) {
                    ForEach(rankOptions, id: \.self) { Text("\($0)").tag($0) }
                }
                .labelsHidden()
                .pickerStyle(.segmented)
                .frame(maxWidth: 150)
                .disabled(isCheckpointResume)
            }
            LabeledContent("Epochs") {
                Picker("Epochs", selection: $selectedEpochs) {
                    ForEach(epochOptions, id: \.self) { Text("\($0)").tag($0) }
                }
                .labelsHidden()
                .pickerStyle(.segmented)
                .frame(maxWidth: 220)
                .padding(.leading, 8)
                .disabled(isCheckpointResume)
            }
            LabeledContent("Layer Coverage") {
                Picker("Layer Coverage", selection: $selectedLayerCount) {
                    ForEach(layerCountOptions, id: \.self) { count in
                        Text(count >= 32 ? "All 32" : "Last \(count)").tag(count)
                    }
                }
                .labelsHidden()
                .pickerStyle(.segmented)
                .frame(width: 340)
                .disabled(isCheckpointResume)
            }
            LabeledContent("Projections") {
                Picker("Projections", selection: $selectedProjectionCoverage) {
                    ForEach(TrainingProjectionCoverage.allCases, id: \.self) { coverage in
                        Text(coverage.displayName).tag(coverage)
                    }
                }
                .labelsHidden()
                .pickerStyle(.segmented)
                .frame(width: 220)
                .disabled(isCheckpointResume)
            }
            if let trainingConfigurationWarning {
                Label(trainingConfigurationWarning, systemImage: "exclamationmark.triangle.fill")
                    .font(.caption)
                    .foregroundStyle(.orange)
                    .fixedSize(horizontal: false, vertical: true)
            }

            // Target layers
            LabeledContent("Target Layers") {
                HStack(spacing: 8) {
                    ForEach(TargetLayer.allCases, id: \.self) { layer in
                        Toggle(layer.displayName, isOn: Binding(
                            get: { selectedTargetLayers.contains(layer) },
                            set: { isOn in
                                if isOn {
                                    selectedTargetLayers.insert(layer)
                                } else if selectedTargetLayers.count > 1 {
                                    selectedTargetLayers.remove(layer)
                                }
                            }
                        ))
                        .toggleStyle(.checkbox)
                        .disabled(isCheckpointResume)
                    }
                }
            }

            // Advanced: warmup + accumulation
            DisclosureGroup("Advanced") {
                LabeledContent("Alpha Scale") {
                    Picker("Alpha Scale", selection: $selectedAlphaScale) {
                        ForEach(TrainingAlphaScale.allCases, id: \.self) { scale in
                            Text(scale.displayName).tag(scale)
                        }
                    }
                    .labelsHidden()
                    .pickerStyle(.segmented)
                    .frame(maxWidth: 140)
                    .disabled(isCheckpointResume)
                }
                LabeledContent("Dropout") {
                    Picker("Dropout", selection: $selectedDropout) {
                        ForEach(TrainingDropoutPreset.allCases, id: \.self) { dropout in
                            Text(dropout.displayName).tag(dropout)
                        }
                    }
                    .labelsHidden()
                    .pickerStyle(.segmented)
                    .frame(maxWidth: 240)
                    .disabled(isCheckpointResume)
                }
                LabeledContent("LR Warmup") {
                    HStack {
                        Slider(
                            value: Binding(
                                get: { Double(warmupFraction) },
                                set: { warmupFraction = Float($0) }
                            ),
                            in: 0...0.5,
                            step: 0.05
                        )
                        .frame(maxWidth: 150)
                        .disabled(isCheckpointResume)
                        Text("\(Int(warmupFraction * 100))%")
                            .monospacedDigit()
                            .frame(width: 40)
                    }
                }
                LabeledContent("Grad. Accumulation") {
                    Picker("Steps", selection: $gradientAccumulationSteps) {
                        ForEach(accumOptions, id: \.self) { Text("\($0)").tag($0) }
                    }
                    .labelsHidden()
                    .pickerStyle(.segmented)
                    .frame(maxWidth: 200)
                    .disabled(isCheckpointResume)
                }
                LabeledContent("Seed") {
                    VStack(alignment: .leading, spacing: 4) {
                        TextField("Seed", text: $trainingSeedText)
                            .textFieldStyle(.roundedBorder)
                            .monospacedDigit()
                            .frame(width: 120)
                            .disabled(isCheckpointResume)
                        if trainingSeedValue == nil {
                            Text("Invalid seed")
                                .font(.caption2)
                                .foregroundStyle(.red)
                        }
                    }
                }
                if adapterKind == .lokr {
                    LabeledContent("DoRA") {
                        Toggle("Weight Decompose", isOn: $selectedLoKRWeightDecompose)
                            .toggleStyle(.checkbox)
                            .disabled(isCheckpointResume)
                    }
                }
            }
        }
    }

    private func trainingRecommendationView(_ recommendation: TrainingPlanRecommendation) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Label("Suggested Start", systemImage: "slider.horizontal.2.square")
                    .font(.caption)
                    .fontWeight(.semibold)
                Spacer()
                Button("Apply") {
                    applyRecommendation(recommendation)
                }
                .buttonStyle(.bordered)
                .controlSize(.mini)
                .disabled(isCheckpointResume || recommendationMatchesCurrentSelection(recommendation))
            }
            Text(recommendation.summary)
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(8)
        .background(.blue.opacity(0.08), in: RoundedRectangle(cornerRadius: 8))
    }

    @ViewBuilder
    private var errorDisplay: some View {
        if case .failed(let message) = trainingService.state {
            Label(message, systemImage: "exclamationmark.triangle")
                .foregroundStyle(.red)
                .font(.caption)
        }
    }

    private var startButtonRow: some View {
        HStack {
            Spacer()
            if isCheckpointResume {
                Button("Resume Training") {
                    startTraining()
                }
                .buttonStyle(.borderedProminent)
                .disabled(!canStartTraining)
            } else {
                Button("Start Training") {
                    startTraining()
                }
                .buttonStyle(.borderedProminent)
                .disabled(!canStartTraining)
            }
        }
    }

    // MARK: - Progress View

    private var progressView: some View {
        VStack(spacing: 16) {
            switch trainingService.state {
            case .loadingModels:
                VStack(spacing: 8) {
                    ProgressView()
                        .controlSize(.large)
                    Text("Loading ACE-Step Models")
                        .font(.headline)
                    Text("Preparing DiT, VAE, tokenizer, text encoders, and silence prior for training.")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                }
                .padding(.top, 40)

            case .preparingDataset(let current, let total):
                VStack(spacing: 8) {
                    Text("Preparing Dataset")
                        .font(.headline)
                    Text("Encoding sample \(current) of \(total)")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                    ProgressView(value: Double(current), total: Double(total))
                        .frame(maxWidth: 300)
                }
                .padding(.top, 40)

            case .training(let uiState):
                VStack(spacing: 12) {
                    Text("Training \"\(adapterName)\"")
                        .font(.headline)
                    Text("Epoch \(uiState.epoch) of \(uiState.totalEpochs)")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)

                    ProgressView(value: Double(uiState.epoch), total: Double(uiState.totalEpochs))
                        .frame(maxWidth: 300)

                    // Loss curve
                    LossCurveView(
                        lossHistory: uiState.lossHistory,
                        validationHistory: uiState.validationHistory
                    )
                        .padding(.horizontal)

                    // Stats row
                    HStack(spacing: 20) {
                        statView(value: String(format: "%.4f", uiState.loss), label: "Loss")
                        if let average = uiState.lossMovingAverage {
                            statView(value: String(format: "%.4f", average), label: "Avg")
                        }
                        if let validationLoss = uiState.validationLoss {
                            statView(value: String(format: "%.4f", validationLoss), label: "Val")
                        }
                        statView(
                            value: String(format: "%.1e", uiState.learningRate),
                            label: "LR"
                        )
                        statView(value: formatTime(uiState.elapsed), label: "Elapsed")
                        if let remaining = uiState.remaining {
                            statView(value: "~\(formatTime(remaining))", label: "ETA")
                        }
                    }
                    if uiState.gradientNorm != nil || uiState.parameterNorm != nil || uiState.updateNorm != nil {
                        HStack(spacing: 16) {
                            if let gradientNorm = uiState.gradientNorm {
                                statView(value: formatMetricNorm(gradientNorm), label: "Grad")
                            }
                            if let updateNorm = uiState.updateNorm {
                                statView(value: formatMetricNorm(updateNorm), label: "Update")
                            }
                            if let parameterNorm = uiState.parameterNorm {
                                statView(value: formatMetricNorm(parameterNorm), label: "Params")
                            }
                        }
                    }
                }
                .padding(.top, 20)

            case .saving:
                VStack(spacing: 8) {
                    ProgressView()
                    Text("Saving adapter...")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
                .padding(.top, 40)

            default:
                EmptyView()
            }

            Spacer()

            if case .training = trainingService.state {
                Button("Pause / Stop Training", role: .destructive) {
                    showingStopConfirmation = true
                }
                .buttonStyle(.bordered)
            }
        }
        .padding()
    }

    // MARK: - Completion View

    private func completionView(adapter: LoRAAdapter) -> some View {
        VStack(spacing: 18) {
            Image(systemName: "checkmark.circle.fill")
                .font(.largeTitle)
                .foregroundStyle(.green)
                .padding(.top, 12)

            Text("Training Complete")
                .font(.headline)

            VStack(spacing: 4) {
                Text(adapter.name)
                    .font(.title3)
                    .fontWeight(.semibold)
                    .multilineTextAlignment(.center)
                    .lineLimit(2)
                    .fixedSize(horizontal: false, vertical: true)
                if let loss = adapter.finalLoss {
                    Text("Final loss: \(String(format: "%.4f", loss))")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Text("\(adapter.kind == .lokr ? "Linear dim" : "Rank") \(adapter.kind == .lokr ? adapter.lokrConfig?.linearDim ?? adapter.config.rank : adapter.config.rank) \u{2022} last \(adapter.trainedLayerIndices?.count ?? adapter.config.trainingLayerCount ?? selectedLayerCount) layers \u{2022} \(adapter.sampleCount) tracks")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .frame(maxWidth: 420)

            VStack(spacing: 10) {
                adapterAuditionPanel(adapter)

                Button {
                    testAdapterInGenerate(adapter)
                } label: {
                    Label("Test in Generate", systemImage: "play.circle")
                }
                .buttonStyle(.borderedProminent)
                .help("Open Generate with this adapter selected, strength 1.0, seed 42, and a training caption as the prompt")

                HStack(spacing: 10) {
                    Button {
                        showAdapterInFinder(adapter)
                    } label: {
                        Label("Finder", systemImage: "folder")
                    }
                    .buttonStyle(.bordered)
                    .help("Reveal the adapter folder, including training_log.json")

                    Button {
                        selectAdapterForGeneration(adapter)
                    } label: {
                        Label("Select Adapter", systemImage: "checkmark.circle")
                    }
                    .buttonStyle(.bordered)
                    .help("Make this the selected Generate adapter at strength 1.0")

                    Button("Train Again") {
                        completionStatusMessage = nil
                        trainingService.resetToIdle()
                    }
                    .buttonStyle(.bordered)
                    .help("Return to config with the same dataset — adjust epochs or settings and retrain")

                    Button("Done") { close() }
                        .buttonStyle(.bordered)
                }

                if let completionStatusMessage {
                    Text(completionStatusMessage)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            .frame(maxWidth: 440)
        }
        .frame(maxWidth: .infinity, alignment: .top)
        .padding(.horizontal, 20)
        .padding(.vertical, 22)
    }

    private func adapterAuditionPanel(_ adapter: LoRAAdapter) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                Label("Audition Snapshots", systemImage: "square.grid.2x2")
                    .font(.caption)
                    .fontWeight(.semibold)
                Spacer()
                if isAuditioning {
                    ProgressView()
                        .controlSize(.small)
                }
            }

            VStack(alignment: .leading, spacing: 8) {
                HStack(spacing: 12) {
                    Toggle("Base", isOn: $auditionIncludeBase)
                        .toggleStyle(.checkbox)
                        .controlSize(.small)
                        .disabled(isAuditioning)
                        .help("Include one no-adapter baseline take.")

                    auditionField("Seed", placeholder: "42", text: $auditionSeedText, width: 72)
                    auditionField("Seconds", placeholder: "30", text: $auditionDurationText, width: 60)
                }

                HStack(spacing: 10) {
                    auditionField(
                        "Gain",
                        placeholder: "1.0, 1.6, 2.0",
                        text: $auditionStrengthsText,
                        minWidth: 170
                    )
                    .help("Comma-separated adapter gains to render with the same seed and prompt.")

                    Spacer(minLength: 8)

                    Button {
                        startAdapterAudition(adapter)
                    } label: {
                        Label("Render", systemImage: "play.fill")
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.small)
                    .disabled(!canStartAdapterAudition)

                    if isAuditioning {
                        Button("Cancel", role: .destructive) {
                            cancelAdapterAudition()
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                    }
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            if let auditionProgressFraction {
                ProgressView(value: auditionProgressFraction)
            }

            if let auditionProgressMessage {
                Text(auditionProgressMessage)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(10)
        .background(.secondary.opacity(0.06), in: RoundedRectangle(cornerRadius: 8))
    }

    private func auditionField(
        _ label: String,
        placeholder: String,
        text: Binding<String>,
        width: CGFloat? = nil,
        minWidth: CGFloat? = nil
    ) -> some View {
        HStack(spacing: 6) {
            Text(label)
                .font(.caption)
                .foregroundStyle(.secondary)
            TextField(placeholder, text: text)
                .textFieldStyle(.roundedBorder)
                .font(.caption.monospacedDigit())
                .frame(width: width)
                .frame(minWidth: minWidth)
                .disabled(isAuditioning)
        }
    }

    // MARK: - Helpers

    private var canStartTraining: Bool {
        !adapterName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && !(selectedEntries.isEmpty && importedFiles.isEmpty)
            && !selectedTargetLayers.isEmpty
            && trainingSeedValue != nil
            && trainingConfigurationBlocker == nil
    }

    private var auditionSeedValue: Int? {
        Int(auditionSeedText.trimmingCharacters(in: .whitespacesAndNewlines))
    }

    private var auditionDurationValue: TimeInterval? {
        guard let value = Double(auditionDurationText.trimmingCharacters(in: .whitespacesAndNewlines)),
              value > 0 else {
            return nil
        }
        return value
    }

    private var auditionStrengths: [Float] {
        AdapterAuditionVariant.parseStrengths(auditionStrengthsText)
    }

    private var canStartAdapterAudition: Bool {
        guard !isAuditioning,
              generationService.canSubmitGeneration,
              auditionSeedValue != nil,
              auditionDurationValue != nil else {
            return false
        }
        return !auditionStrengths.isEmpty || auditionIncludeBase
    }

    private var trainingSeedValue: UInt64? {
        let trimmed = trainingSeedText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        return UInt64(trimmed)
    }

    private var trainingConfigurationWarning: String? {
        if let trainingConfigurationBlocker {
            return trainingConfigurationBlocker
        }
        return localTrainingShapeWarning
    }

    private var trainingConfigurationBlocker: String? {
        if let tensorCapMessage = localLoKRTensorCapMessage {
            return tensorCapMessage
        }
        return nil
    }

    private var localTrainingShapeWarning: String? {
        if adapterKind == .lora,
           selectedProjectionCoverage == .fullAttention,
           selectedLayerCount >= 12 {
            return "Full Q/K/V/O LoRA across 12+ local XL layers is a large local shape. Q/V projections or Last 8 remain the safer starting point."
        }

        if adapterKind == .lokr,
           selectedProjectionCoverage == .fullAttention,
           selectedLayerCount >= 16 {
            return "Full Q/K/V/O LoKR across 16+ local XL layers is sensitive and slow. Q/V projections or Last 8 remain the safer starting point."
        }

        if adapterKind == .lokr,
           selectedProjectionCoverage == .fullAttention,
           selectedRank > 32 {
            return "Full Q/K/V/O LoKR above Linear Dim 32 is a large local shape. Linear Dim 32 or Q/V projections are recommended."
        }

        return nil
    }

    private var localLoKRTensorCapMessage: String? {
        guard let trainableTensorCount = estimatedLoKRTrainableTensorCount,
              trainableTensorCount > maxLocalLoKRTrainableTensorCount
        else {
            return nil
        }
        return "This LoKR shape would train \(trainableTensorCount) tensors, which is above the local \(maxLocalLoKRTrainableTensorCount)-tensor cap. Reduce layer coverage, projections, or target layers."
    }

    private var estimatedLoKRTrainableTensorCount: Int? {
        guard adapterKind == .lokr else { return nil }
        let patterns = selectedInjectionPatternsForEstimate()
        guard !patterns.isEmpty else { return nil }
        return min(selectedLayerCount, 32) * patterns.count * (selectedLoKRWeightDecompose ? 3 : 2)
    }

    private func selectedInjectionPatternsForEstimate() -> [String] {
        if let exactPatterns = selectedProjectionCoverage.exactInjectionPatterns(for: selectedTargetLayers) {
            return exactPatterns
        }
        return selectedTargetLayers.flatMap(\.defaultPatterns)
    }

    private var isResumeSession: Bool {
        resumeAdapterID != nil
    }

    private var isPausedState: Bool {
        if case .paused = trainingService.state { return true }
        return false
    }

    private var isCheckpointResume: Bool {
        isResumeSession || isPausedState
    }

    private var trainingRecommendation: TrainingPlanRecommendation? {
        TrainingPlanRecommendation.make(
            adapterKind: adapterKind,
            profile: trainingDatasetProfile
        )
    }

    private var trainingDatasetProfile: TrainingDatasetProfile {
        let selectedDurations = selectedEntries
            .map(\.duration)
            .filter { $0 > 0 }
        let importedDurations = importedFiles
            .compactMap { importedFileDurations[durationKey(for: $0.url)] }
            .filter { $0 > 0 }

        return TrainingDatasetProfile(
            sampleCount: selectedEntries.count + importedFiles.count,
            knownDuration: selectedDurations.reduce(0, +) + importedDurations.reduce(0, +),
            knownDurationSampleCount: selectedDurations.count + importedDurations.count
        )
    }

    private var importedDurationRefreshKey: String {
        importedFiles
            .map { durationKey(for: $0.url) }
            .sorted()
            .joined(separator: "\n")
    }

    private func close() {
        if let onClose {
            onClose()
        } else {
            dismiss()
        }
    }

    private func applyRecommendation(_ recommendation: TrainingPlanRecommendation) {
        selectedRecipe = .balancedLocal
        selectedRank = recommendation.rank
        selectedEpochs = recommendation.epochs
        selectedLayerCount = recommendation.layerCount
        selectedProjectionCoverage = .queryValue
        selectedAlphaScale = .rank
        selectedDropout = .none
        if adapterKind == .lokr {
            selectedLoKRWeightDecompose = true
        }
        gradientAccumulationSteps = recommendation.gradientAccumulationSteps
    }

    private func applyRecipe(_ recipe: TrainingRecipePreset) {
        guard !isCheckpointResume else { return }
        let defaults = recipe.defaults(
            for: adapterKind,
            recommendation: trainingRecommendation
        )
        selectedRank = defaults.rank
        selectedEpochs = defaults.epochs
        selectedLayerCount = defaults.layerCount
        selectedProjectionCoverage = defaults.projectionCoverage
        selectedAlphaScale = defaults.alphaScale
        selectedDropout = defaults.dropout
        if let targetLayers = defaults.targetLayers {
            selectedTargetLayers = targetLayers
        }
        if adapterKind == .lokr {
            selectedLoKRWeightDecompose = true
        }
        gradientAccumulationSteps = defaults.gradientAccumulationSteps
    }

    private func recommendationMatchesCurrentSelection(_ recommendation: TrainingPlanRecommendation) -> Bool {
        selectedRank == recommendation.rank
            && selectedEpochs == recommendation.epochs
            && selectedLayerCount == recommendation.layerCount
            && selectedProjectionCoverage == .queryValue
            && selectedAlphaScale == .rank
            && selectedDropout == .none
            && gradientAccumulationSteps == recommendation.gradientAccumulationSteps
    }

    private func showAdapterInFinder(_ adapter: LoRAAdapter) {
        guard let directory = try? AdapterRegistry.adapterDirectory(id: adapter.id) else { return }
        NSWorkspace.shared.activateFileViewerSelecting([
            directory
        ])
    }

    private func refreshImportedFileDurations() async {
        let urls = importedFiles.map(\.url)
        guard !urls.isEmpty else {
            importedFileDurations = [:]
            return
        }

        let durations = await Task.detached(priority: .utility) {
            var result: [String: TimeInterval] = [:]
            for url in urls {
                guard let duration = Self.audioDuration(for: url), duration > 0 else {
                    continue
                }
                result[Self.durationKey(for: url)] = duration
            }
            return result
        }.value

        importedFileDurations = durations
    }

    nonisolated private static func audioDuration(for url: URL) -> TimeInterval? {
        let didAccess = url.startAccessingSecurityScopedResource()
        defer {
            if didAccess {
                url.stopAccessingSecurityScopedResource()
            }
        }

        guard let audioFile = try? AVAudioFile(forReading: url) else {
            return nil
        }
        let sampleRate = audioFile.fileFormat.sampleRate
        guard sampleRate > 0 else {
            return nil
        }
        return Double(audioFile.length) / sampleRate
    }

    private func durationKey(for url: URL) -> String {
        Self.durationKey(for: url)
    }

    nonisolated private static func durationKey(for url: URL) -> String {
        url.standardizedFileURL.path(percentEncoded: false)
    }

    private func selectAdapterForGeneration(_ adapter: LoRAAdapter) {
        UserDefaults.standard.set(adapter.id, forKey: AppUserDefaultsKeys.selectedAdapterID)
        UserDefaults.standard.set(1.0, forKey: AppUserDefaultsKeys.selectedAdapterStrength)
        NotificationCenter.default.post(
            name: .adapterSelectionDidChange,
            object: nil,
            userInfo: [
                "adapters": [
                    [
                        "id": adapter.id,
                        "strength": 1.0,
                    ]
                ]
            ]
        )
        completionStatusMessage = "Selected for Generate at strength 1.0."
    }

    private func testAdapterInGenerate(_ adapter: LoRAAdapter) {
        selectAdapterForGeneration(adapter)
        presetStore.presetToLoad = GenerationPreset(
            name: "\(adapter.name) Audition",
            prompt: auditionPrompt(for: adapter),
            lyrics: auditionLyrics,
            mode: .music,
            duration: 30,
            seedText: "42",
            guidanceScale: 7.5,
            guidancePreset: .constant,
            pipelineOverride: .aceStep,
            method: .ode,
            precision: .float32,
            stepsOverride: "",
            shiftOverride: 3.0,
            adapterID: adapter.id,
            adapterStrength: 1.0,
            coverStrength: 0.7
        )
        close()
    }

    private func startAdapterAudition(_ adapter: LoRAAdapter) {
        guard let seed = auditionSeedValue else {
            auditionProgressMessage = "Enter a numeric seed."
            return
        }
        guard let duration = auditionDurationValue else {
            auditionProgressMessage = "Enter a duration greater than zero."
            return
        }

        let variants = AdapterAuditionVariant.variants(
            for: adapter,
            strengths: auditionStrengths,
            includeBase: auditionIncludeBase
        )
        guard !variants.isEmpty else {
            auditionProgressMessage = "Add at least one adapter gain or include Base."
            return
        }

        let prompt = auditionPrompt(for: adapter)
        let lyrics = auditionLyrics
        isAuditioning = true
        auditionProgressFraction = 0
        auditionProgressMessage = "Starting audition snapshots..."
        completionStatusMessage = nil

        auditionTask = Task { @MainActor in
            var completedTraces: [GenerationTrace] = []
            for (offset, variant) in variants.enumerated() {
                if Task.isCancelled {
                    break
                }

                let index = offset + 1
                auditionProgressFraction = Double(offset) / Double(max(variants.count, 1))
                auditionProgressMessage = "Rendering \(variant.label) snapshot \(index) of \(variants.count)..."

                let request = GenerationRequest(
                    text: prompt,
                    lyrics: lyrics,
                    duration: duration,
                    seed: seed,
                    method: .ode,
                    guidanceScale: 7.5,
                    guidancePreset: .constant,
                    precision: .float32,
                    stepsOverride: nil,
                    shiftOverride: 3.0,
                    resolvedMode: .music,
                    pipelineOverride: .aceStep,
                    coverStrength: 0.7
                )

                do {
                    let result = try await generationService.generateAndWait(
                        request: request,
                        adapterStack: variant.adapterStack,
                        selectedAdapterID: variant.adapterID
                    )
                    let trace = try saveAuditionNote(
                        for: result,
                        variant: variant,
                        index: index,
                        total: variants.count,
                        seed: seed,
                        duration: duration
                    )
                    completedTraces.append(trace)
                    auditionProgressFraction = Double(index) / Double(max(variants.count, 1))
                } catch is CancellationError {
                    auditionProgressMessage = "Audition cancelled."
                    generationService.cancel()
                    finishAdapterAudition()
                    return
                } catch {
                    auditionProgressMessage = "Audition stopped: \(error.localizedDescription)"
                    finishAdapterAudition()
                    return
                }
            }

            finishAdapterAudition()
            guard !Task.isCancelled else {
                auditionProgressMessage = "Audition cancelled."
                return
            }

            if completedTraces.count >= 2 {
                traceMonitor.compare(left: completedTraces[0], right: completedTraces[1])
                openWindow(id: "generationAnalyzer")
                auditionProgressMessage = "Rendered \(completedTraces.count) snapshots. Compare opened for \(variants[0].label) vs \(variants[1].label)."
            } else if completedTraces.count == 1 {
                traceMonitor.inspect(trace: completedTraces[0])
                openWindow(id: "generationAnalyzer")
                auditionProgressMessage = "Rendered 1 snapshot."
            } else {
                auditionProgressMessage = "No audition snapshots were rendered."
            }
        }
    }

    private func saveAuditionNote(
        for result: GenerationResult,
        variant: AdapterAuditionVariant,
        index: Int,
        total: Int,
        seed: Int,
        duration: TimeInterval
    ) throws -> GenerationTrace {
        let note = variant.listeningNote(
            index: index,
            total: total,
            seed: seed,
            duration: duration
        )

        if let traceFileName = result.entry.traceFileName {
            return try historyStore.saveTraceReview(
                traceFileName: traceFileName,
                listeningNote: note,
                isPreferred: nil
            )
        }

        return try historyStore.loadTrace(for: result.entry)
    }

    private func cancelAdapterAudition() {
        auditionTask?.cancel()
        generationService.cancel()
        finishAdapterAudition()
        auditionProgressMessage = "Audition cancelled."
    }

    private func finishAdapterAudition() {
        isAuditioning = false
        auditionProgressFraction = nil
        auditionTask = nil
    }

    private func auditionPrompt(for adapter: LoRAAdapter) -> String {
        let candidates = importedFiles.map(\.description) + selectedEntries.map(\.prompt)
        return candidates
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .first { !$0.isEmpty }
            ?? "\(adapter.name) music style"
    }

    private var auditionLyrics: String {
        importedFiles
            .map(\.lyrics)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .first { !$0.isEmpty }
            ?? ""
    }

    private func statView(value: String, label: String) -> some View {
        VStack(spacing: 2) {
            Text(value)
                .font(.title3)
                .fontWeight(.semibold)
                .monospacedDigit()
            Text(label)
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
    }

    private func formatTime(_ interval: TimeInterval) -> String {
        let minutes = Int(interval) / 60
        let seconds = Int(interval) % 60
        return "\(minutes):\(String(format: "%02d", seconds))"
    }

    private func formatMetricNorm(_ value: Float) -> String {
        if value == 0 {
            return "0"
        }
        let magnitude = abs(value)
        if magnitude >= 1_000 || magnitude < 0.01 {
            return String(format: "%.1e", value)
        }
        return String(format: "%.3f", value)
    }

    private func startTraining() {
        let audioURLs = selectedEntries.compactMap { try? historyStore.fileURL(for: $0) }
        let exactInjectionPatterns = selectedProjectionCoverage.exactInjectionPatterns(for: selectedTargetLayers)
        let alpha = selectedAlphaScale.alpha(for: selectedRank)
        let trainingSeed = trainingSeedValue ?? LoRAConfig.defaultTrainingSeed
        let config = LoRAConfig(
            rank: selectedRank,
            alpha: alpha,
            dropout: selectedDropout.value,
            targetLayers: selectedTargetLayers,
            exactInjectionPatterns: exactInjectionPatterns,
            learningRate: selectedRecipe.learningRate(for: adapterKind),
            maxEpochs: selectedEpochs,
            warmupFraction: warmupFraction,
            gradientAccumulationSteps: gradientAccumulationSteps,
            trainingLayerCount: selectedLayerCount,
            trainingSeed: trainingSeed
        )
        switch adapterKind {
        case .lora:
            trainingService.startTraining(
                entries: selectedEntries,
                audioURLs: audioURLs,
                importedFiles: importedFiles,
                adapterName: adapterName,
                config: config,
                resumeAdapterID: resumeAdapterID
            )
        case .lokr:
            trainingService.startLoKRTraining(
                entries: selectedEntries,
                audioURLs: audioURLs,
                importedFiles: importedFiles,
                adapterName: adapterName,
                config: LoKRConfig(
                    linearDim: selectedRank,
                    alpha: alpha,
                    dropout: selectedDropout.value,
                    targetLayers: selectedTargetLayers,
                    exactInjectionPatterns: exactInjectionPatterns,
                    learningRate: selectedRecipe.learningRate(for: adapterKind),
                    maxEpochs: selectedEpochs,
                    warmupFraction: warmupFraction,
                    gradientAccumulationSteps: gradientAccumulationSteps,
                    trainingLayerCount: selectedLayerCount,
                    trainingSeed: trainingSeed,
                    weightDecompose: selectedLoKRWeightDecompose
                ),
                resumeAdapterID: resumeAdapterID
            )
        }
    }

    private func refreshBaseModelDisplayName() async {
        let name = await trainingService.activeBaseModelDisplayName()
        await MainActor.run {
            baseModelDisplayName = name
        }
    }

    private func importSelectedFiles(_ urls: [URL]) {
        importErrorMessage = nil
        importStatusMessage = nil
        var importedJSONCount = 0
        var importedAudioCount = 0
        var lyricCount = 0
        var duplicateCount = 0

        for url in urls {
            if url.pathExtension.lowercased() == "json" {
                do {
                    let imported = try TrainingDatasetImporter.importFiles(from: url)
                    let result = appendImportedFiles(imported)
                    importedJSONCount += result.added
                    duplicateCount += result.duplicates
                    lyricCount += imported.filter { !$0.lyrics.isEmpty }.count
                } catch {
                    importErrorMessage = error.localizedDescription
                }
            } else {
                let result = appendImportedFiles([TrainingImportedFile(url: url, description: "")])
                importedAudioCount += result.added
                duplicateCount += result.duplicates
            }
        }

        importStatusMessage = importSummary(
            jsonCount: importedJSONCount,
            audioCount: importedAudioCount,
            lyricCount: lyricCount,
            duplicateCount: duplicateCount
        )
    }

    private func appendImportedFiles(_ files: [TrainingImportedFile]) -> (added: Int, duplicates: Int) {
        var seen = Set(importedFiles.map { $0.url.standardizedFileURL.path })
        var added = 0
        var duplicates = 0

        for file in files {
            if seen.insert(file.url.standardizedFileURL.path).inserted {
                importedFiles.append(file)
                added += 1
            } else {
                duplicates += 1
            }
        }

        return (added, duplicates)
    }

    private func importSummary(
        jsonCount: Int,
        audioCount: Int,
        lyricCount: Int,
        duplicateCount: Int
    ) -> String? {
        guard jsonCount > 0 || audioCount > 0 || duplicateCount > 0 else { return nil }

        var parts: [String] = []
        if jsonCount > 0 {
            let lyricSuffix = lyricCount > 0 ? ", \(lyricCount) with lyrics" : ""
            parts.append("Imported \(jsonCount) from JSON\(lyricSuffix)")
        }
        if audioCount > 0 {
            parts.append("Added \(audioCount) audio file\(audioCount == 1 ? "" : "s")")
        }
        if duplicateCount > 0 {
            parts.append("Skipped \(duplicateCount) duplicate\(duplicateCount == 1 ? "" : "s")")
        }
        return parts.joined(separator: ". ")
    }
}

private enum TrainingRecipePreset: String, CaseIterable, Identifiable {
    case balancedLocal
    case wideLoRA
    case expandedLoKR

    var id: String { rawValue }

    static func options(for adapterKind: AdapterKind) -> [TrainingRecipePreset] {
        switch adapterKind {
        case .lora:
            return [.balancedLocal, .wideLoRA]
        case .lokr:
            return [.balancedLocal, .expandedLoKR]
        }
    }

    static func infer(from config: LoRAConfig, adapterKind: AdapterKind) -> TrainingRecipePreset {
        let projectionCoverage = TrainingProjectionCoverage.infer(from: config)
        let alphaScale = TrainingAlphaScale.infer(from: config.alpha, dimension: config.rank)
        switch adapterKind {
        case .lora:
            let isWide = projectionCoverage == .fullAttention
                && alphaScale == .doubleRank
                && (config.trainingLayerCount ?? 0) >= 32
            guard isWide else { return .balancedLocal }
            return .wideLoRA
        case .lokr:
            let isWide = alphaScale == .doubleRank
                && config.rank >= 32
                && (config.trainingLayerCount ?? 0) >= 16
            guard isWide else { return .balancedLocal }
            return .expandedLoKR
        }
    }

    func displayName(for adapterKind: AdapterKind) -> String {
        switch self {
        case .balancedLocal:
            return "Narrow"
        case .wideLoRA:
            return "Wide"
        case .expandedLoKR:
            return "Wide"
        }
    }

    func summary(for adapterKind: AdapterKind) -> String {
        switch self {
        case .balancedLocal:
            return ""
        case .wideLoRA:
            return adapterKind == .lora
                ? "Uses a wider LoRA footprint: rank 96, alpha 192, dropout 0.08, all 32 layers, and q/k/v/o attention."
                : "Uses a wider local footprint: double alpha, dropout, all 32 layers, and q/k/v/o attention."
        case .expandedLoKR:
            return "Uses a wider local LoKR footprint: linear dim 32, double alpha, DoRA, dropout, last 16 layers, and Q/V attention."
        }
    }

    func defaults(
        for adapterKind: AdapterKind,
        recommendation: TrainingPlanRecommendation?
    ) -> TrainingRecipeDefaults {
        switch self {
        case .balancedLocal:
            return TrainingRecipeDefaults(
                rank: recommendation?.rank ?? 16,
                epochs: recommendation?.epochs ?? 10,
                layerCount: recommendation?.layerCount ?? 8,
                projectionCoverage: .queryValue,
                alphaScale: .rank,
                dropout: .none,
                targetLayers: nil,
                gradientAccumulationSteps: recommendation?.gradientAccumulationSteps ?? 1
            )
        case .wideLoRA:
            return TrainingRecipeDefaults(
                rank: adapterKind == .lora ? 96 : 64,
                epochs: adapterKind == .lora ? 150 : 100,
                layerCount: 32,
                projectionCoverage: .fullAttention,
                alphaScale: .doubleRank,
                dropout: .p08,
                targetLayers: [.selfAttention, .crossAttention],
                gradientAccumulationSteps: 4
            )
        case .expandedLoKR:
            return TrainingRecipeDefaults(
                rank: 32,
                epochs: 100,
                layerCount: 16,
                projectionCoverage: .queryValue,
                alphaScale: .doubleRank,
                dropout: .p08,
                targetLayers: [.selfAttention, .crossAttention],
                gradientAccumulationSteps: 4
            )
        }
    }

    func learningRate(for adapterKind: AdapterKind) -> Float {
        switch self {
        case .wideLoRA where adapterKind == .lora:
            return 0.000075
        default:
            return 0.0001
        }
    }
}

private struct TrainingRecipeDefaults {
    let rank: Int
    let epochs: Int
    let layerCount: Int
    let projectionCoverage: TrainingProjectionCoverage
    let alphaScale: TrainingAlphaScale
    let dropout: TrainingDropoutPreset
    let targetLayers: Set<TargetLayer>?
    let gradientAccumulationSteps: Int
}

private enum TrainingProjectionCoverage: String, CaseIterable, Identifiable {
    case queryValue
    case fullAttention

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .queryValue:
            return "Q/V"
        case .fullAttention:
            return "Q/K/V/O"
        }
    }

    static func infer(from config: LoRAConfig) -> TrainingProjectionCoverage {
        let patterns = Set(config.injectionPatterns)
        let hasKeyOrOutput = patterns.contains("selfAttn.kProj")
            || patterns.contains("selfAttn.oProj")
            || patterns.contains("crossAttn.kProj")
            || patterns.contains("crossAttn.oProj")
        return hasKeyOrOutput ? .fullAttention : .queryValue
    }

    func exactInjectionPatterns(for targetLayers: Set<TargetLayer>) -> [String]? {
        switch self {
        case .queryValue:
            return nil
        case .fullAttention:
            var patterns: [String] = []
            if targetLayers.contains(.selfAttention) {
                patterns.append(contentsOf: [
                    "selfAttn.qProj",
                    "selfAttn.kProj",
                    "selfAttn.vProj",
                    "selfAttn.oProj",
                ])
            }
            if targetLayers.contains(.crossAttention) {
                patterns.append(contentsOf: [
                    "crossAttn.qProj",
                    "crossAttn.kProj",
                    "crossAttn.vProj",
                    "crossAttn.oProj",
                ])
            }
            if targetLayers.contains(.mlp) {
                patterns.append(contentsOf: TargetLayer.mlp.defaultPatterns)
            }
            return patterns.isEmpty ? nil : patterns
        }
    }
}

private enum TrainingAlphaScale: Float, CaseIterable, Identifiable {
    case rank = 1
    case doubleRank = 2

    var id: Float { rawValue }

    var displayName: String {
        switch self {
        case .rank:
            return "1x"
        case .doubleRank:
            return "2x"
        }
    }

    static func infer(from alpha: Float, dimension: Int) -> TrainingAlphaScale {
        let ratio = alpha / Float(max(1, dimension))
        return ratio >= 1.5 ? .doubleRank : .rank
    }

    func alpha(for dimension: Int) -> Float {
        Float(dimension) * rawValue
    }
}

private enum TrainingDropoutPreset: Float, CaseIterable, Identifiable {
    case none = 0
    case p05 = 0.05
    case p08 = 0.08
    case p10 = 0.10

    var id: Float { rawValue }

    var value: Float { rawValue }

    var displayName: String {
        switch self {
        case .none:
            return "0"
        case .p05:
            return "0.05"
        case .p08:
            return "0.08"
        case .p10:
            return "0.10"
        }
    }

    static func closest(to value: Float) -> TrainingDropoutPreset {
        allCases.min { lhs, rhs in
            abs(lhs.value - value) < abs(rhs.value - value)
        } ?? .none
    }
}

private extension AdapterKind {
    var title: String {
        switch self {
        case .lora: return "LoRA"
        case .lokr: return "LoKR"
        }
    }
}

struct TrainingDatasetProfile: Equatable {
    let sampleCount: Int
    let knownDuration: TimeInterval
    let knownDurationSampleCount: Int

    var durationCoverage: Double {
        guard sampleCount > 0 else { return 0 }
        return Double(min(knownDurationSampleCount, sampleCount)) / Double(sampleCount)
    }

    var hasUsableDurationCoverage: Bool {
        knownDuration > 0 && durationCoverage >= 0.80
    }
}

struct TrainingPlanRecommendation: Equatable {
    let adapterKind: AdapterKind
    let profile: TrainingDatasetProfile
    let rank: Int
    let epochs: Int
    let layerCount: Int
    let gradientAccumulationSteps: Int
    let estimatedOptimizerUpdates: Int
    let rationale: String

    var summary: String {
        let dimensionName = adapterKind == .lokr ? "dim" : "rank"
        let duration = durationSummary
        return "\(profile.sampleCount) tracks\(duration): \(dimensionName) \(rank), \(epochs) epochs, last \(layerCount) layers, grad accum \(gradientAccumulationSteps), ~\(estimatedOptimizerUpdates) updates. \(rationale)"
    }

    static func make(
        adapterKind: AdapterKind,
        sampleCount: Int,
        knownDuration: TimeInterval = 0
    ) -> TrainingPlanRecommendation? {
        make(
            adapterKind: adapterKind,
            profile: TrainingDatasetProfile(
                sampleCount: sampleCount,
                knownDuration: knownDuration,
                knownDurationSampleCount: knownDuration > 0 ? sampleCount : 0
            )
        )
    }

    static func make(
        adapterKind: AdapterKind,
        profile: TrainingDatasetProfile
    ) -> TrainingPlanRecommendation? {
        guard profile.sampleCount > 0 else { return nil }

        let sampleTier = DatasetTier(sampleCount: profile.sampleCount)
        let diversityTier = DatasetTier.diversityTier(for: profile, fallback: sampleTier)
        let gradientAccumulationSteps = sampleTier.gradientAccumulationSteps
        let updatesPerEpoch = optimizerStepsPerEpoch(
            datasetSize: profile.sampleCount,
            gradientAccumulationSteps: gradientAccumulationSteps
        )
        let rawEpochs = Int(ceil(Double(diversityTier.targetOptimizerUpdates) / Double(updatesPerEpoch)))
        let epochs = nearestEpochOption(to: rawEpochs)
        let estimatedOptimizerUpdates = updatesPerEpoch * epochs

        return TrainingPlanRecommendation(
            adapterKind: adapterKind,
            profile: TrainingDatasetProfile(
                sampleCount: profile.sampleCount,
                knownDuration: max(0, profile.knownDuration),
                knownDurationSampleCount: max(0, min(profile.knownDurationSampleCount, profile.sampleCount))
            ),
            rank: diversityTier.dimension(for: adapterKind),
            epochs: epochs,
            layerCount: diversityTier.layerCount,
            gradientAccumulationSteps: gradientAccumulationSteps,
            estimatedOptimizerUpdates: estimatedOptimizerUpdates,
            rationale: diversityTier.rationale
        )
    }

    private var durationSummary: String {
        guard profile.knownDuration >= 1 else { return "" }
        let minutes = Int((profile.knownDuration / 60).rounded())
        guard minutes > 0 else { return "" }
        if profile.knownDurationSampleCount >= profile.sampleCount {
            return ", ~\(minutes)m audio"
        }
        return ", ~\(minutes)m known audio from \(profile.knownDurationSampleCount)/\(profile.sampleCount)"
    }

    private static let epochOptions = [5, 10, 20, 50, 75, 100]

    private static func optimizerStepsPerEpoch(
        datasetSize: Int,
        gradientAccumulationSteps: Int
    ) -> Int {
        let sampleCount = max(1, datasetSize)
        let requestedAccumulation = max(1, gradientAccumulationSteps)
        let accumulation = min(requestedAccumulation, sampleCount)
        return max(1, (sampleCount + accumulation - 1) / accumulation)
    }

    private static func nearestEpochOption(to rawEpochs: Int) -> Int {
        let clamped = max(epochOptions.first ?? 5, min(epochOptions.last ?? 100, rawEpochs))
        return epochOptions.min { lhs, rhs in
            let lhsDistance = abs(lhs - clamped)
            let rhsDistance = abs(rhs - clamped)
            if lhsDistance == rhsDistance {
                return lhs < rhs
            }
            return lhsDistance < rhsDistance
        } ?? 10
    }

    private enum DatasetTier: Int, Comparable {
        case tiny
        case small
        case medium
        case large
        case huge

        static func < (lhs: DatasetTier, rhs: DatasetTier) -> Bool {
            lhs.rawValue < rhs.rawValue
        }

        init(sampleCount: Int) {
            switch sampleCount {
            case ..<20:
                self = .tiny
            case ..<80:
                self = .small
            case ..<200:
                self = .medium
            case ..<500:
                self = .large
            default:
                self = .huge
            }
        }

        init(duration: TimeInterval) {
            let minutes = max(0, duration / 60)
            switch minutes {
            case ..<10:
                self = .tiny
            case ..<40:
                self = .small
            case ..<100:
                self = .medium
            case ..<240:
                self = .large
            default:
                self = .huge
            }
        }

        static func diversityTier(
            for profile: TrainingDatasetProfile,
            fallback sampleTier: DatasetTier
        ) -> DatasetTier {
            guard profile.hasUsableDurationCoverage else {
                return sampleTier
            }
            return min(sampleTier, DatasetTier(duration: profile.knownDuration))
        }

        func dimension(for adapterKind: AdapterKind) -> Int {
            switch (self, adapterKind) {
            case (.tiny, _), (.small, _):
                return 16
            case (.medium, .lora):
                return 32
            case (.medium, .lokr):
                return 16
            case (.large, .lora), (.huge, .lora):
                return 64
            case (.large, .lokr), (.huge, .lokr):
                return 32
            }
        }

        var targetOptimizerUpdates: Int {
            switch self {
            case .tiny:
                return 600
            case .small:
                return 900
            case .medium:
                return 1_400
            case .large:
                return 2_200
            case .huge:
                return 3_200
            }
        }

        var layerCount: Int {
            switch self {
            case .tiny:
                return 4
            case .small, .medium:
                return 8
            case .large:
                return 12
            case .huge:
                return 16
            }
        }

        var gradientAccumulationSteps: Int {
            switch self {
            case .tiny, .small:
                return 1
            case .medium, .large:
                return 2
            case .huge:
                return 4
            }
        }

        var rationale: String {
            switch self {
            case .tiny:
                return "Use more passes and fewer layers to avoid spreading a tiny set too thin."
            case .small:
                return "A longer run helps the adapter catch the style without widening too much."
            case .medium:
                return "Enough data for a wider adapter and moderate passes."
            case .large:
                return "Use stronger capacity and broader coverage while keeping the run bounded."
            case .huge:
                return "Favor coverage and fewer passes because the dataset already supplies many steps."
            }
        }
    }
}

// MARK: - TargetLayer Display

extension TargetLayer {
    var displayName: String {
        switch self {
        case .selfAttention: return "Self-Attn"
        case .crossAttention: return "Cross-Attn"
        case .mlp: return "MLP"
        }
    }
}
