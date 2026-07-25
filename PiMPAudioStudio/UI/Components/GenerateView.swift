import SwiftUI
import PiMPAudioCore
import UniformTypeIdentifiers

struct GenerateView: View {
    @EnvironmentObject private var appState: AppState
    @EnvironmentObject private var generationService: GenerationService
    @EnvironmentObject private var trainingService: TrainingService
    @EnvironmentObject private var playerService: AudioPlayerService
    @EnvironmentObject private var intentRouter: IntentRouter
    @EnvironmentObject private var promptEnhancer: PromptEnhancer
    @EnvironmentObject private var lyricWriterManager: LyricWriterManager
    @EnvironmentObject private var sfxModelManager: SFXModelManager
    @EnvironmentObject private var stackManager: AdapterStackManager
    @EnvironmentObject private var generationParameters: GenerationParameters
    @EnvironmentObject private var presetStore: PresetStore
    @Environment(\.openWindow) private var openWindow

    @State private var showingSavePreset: Bool = false
    @State private var presetName: String = ""
    @State private var promptText: String = ""
    @State private var lyricsText: String = ""
    @State private var method: InferMethod = .ode
    @State private var precision: Precision = .float32
    @State private var stepsOverrideText: String = ""
    @State private var shiftOverride: Double = 3.0
    @State private var advancedExpanded: Bool = false
    @AppStorage(AppUserDefaultsKeys.audioDebugVerbose) private var audioDebugVerbose: Bool = false
    @State private var audioDebugLogPath: String?
    @State private var diagnosticsPruneMessage: String?
    @State private var modeSelection: ModeSelection = .auto
    @State private var selectedAdapterID: String?
    @State private var activeDiTModelID: String?
    @State private var activeDiTModelSpec: ModelSpec?
    @AppStorage(AppUserDefaultsKeys.selectedAdapterID) private var persistedAdapterID: String = ""
    @AppStorage(AppUserDefaultsKeys.selectedAdapterStrength) private var persistedAdapterStrength: Double = 1.0
    @AppStorage(AppUserDefaultsKeys.generatePromptEditorHeight) private var promptEditorHeight: Double = 80
    @AppStorage(AppUserDefaultsKeys.generateLyricsEditorHeight) private var lyricsEditorHeight: Double = 120

    // Cover song
    @State private var sourceAudioURL: URL? = nil
    @State private var coverStrength: Double = 0.7
    @State private var showingAudioImporter: Bool = false
    @State private var isDropTargeted: Bool = false
    @State private var sourceAudioImportError: String?

    private static let promptEditorHeightRange: ClosedRange<Double> = 60...800
    private static let lyricsEditorHeightRange: ClosedRange<Double> = 80...900

    var body: some View {
        ScrollView {
            VStack(spacing: 16) {
                modePickerSection
                promptSection
                lyricsSection
                sourceAudioSection
                controlsBar
                advancedSection
                adapterPickerSection
                progressSection
                playerSection
            }
            .padding()
        }
        .onChange(of: promptText) { _, newValue in
            intentRouter.classify(prompt: newValue)
        }
        .onAppear {
            AudioDiagnostics.setLogDirectory(AppStorageLocations.diagnosticsDirectory)
            AudioDiagnostics.refreshFileLoggingState()
            updateAudioDebugLogPath()
            Task { await refreshActiveDiTAndApplyXLGuidanceRecipe() }
        }
        .onChange(of: audioDebugVerbose) { _, newValue in
            AudioDiagnostics.setVerboseLoggingEnabled(newValue)
            updateAudioDebugLogPath()
        }
        .onReceive(presetStore.$presetToLoad) { preset in
            guard let preset else { return }
            applyPreset(preset)
            presetStore.presetToLoad = nil
        }
        .onReceive(presetStore.$historyRestore) { restore in
            guard let restore else { return }
            applyHistoryRestore(restore)
            presetStore.historyRestore = nil
        }
        .onReceive(NotificationCenter.default.publisher(for: .mcpAdapterSelectionDidChange)) { notification in
            applyAdapterSelection(notification)
        }
        .onReceive(NotificationCenter.default.publisher(for: .adapterSelectionDidChange)) { notification in
            applyAdapterSelection(notification)
        }
        .onReceive(NotificationCenter.default.publisher(for: .modelRegistryDidChange)) { _ in
            Task { await refreshActiveDiTAndApplyXLGuidanceRecipe() }
        }
        .alert("Save as Preset", isPresented: $showingSavePreset) {
            TextField("Preset name", text: $presetName)
            Button("Cancel", role: .cancel) {}
            Button("Save") { saveCurrentAsPreset() }
        } message: {
            Text("Enter a name for this preset.")
        }
        .alert("Could Not Import Source Audio", isPresented: Binding(
            get: { sourceAudioImportError != nil },
            set: { if !$0 { sourceAudioImportError = nil } }
        )) {
            Button("OK", role: .cancel) { sourceAudioImportError = nil }
        } message: {
            Text(sourceAudioImportError ?? "")
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
            resizableTextEditor(
                text: $promptText,
                height: $promptEditorHeight,
                range: Self.promptEditorHeightRange,
                disabled: false
            )
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
                                duration: generationParameters.duration,
                                into: $lyricsText
                            )
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.mini)
                        .disabled(promptText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    }
                }
            }
            resizableTextEditor(
                text: $lyricsText,
                height: $lyricsEditorHeight,
                range: Self.lyricsEditorHeightRange,
                disabled: lyricWriterManager.isWriting
            )
            if let message = lyricWriterManager.lastError {
                Label(message, systemImage: "exclamationmark.triangle")
                    .font(.caption)
                    .foregroundStyle(.red)
            }
        }
    }

    /// Text field with a bottom grip for vertical-only resize (width stays full).
    private func resizableTextEditor(
        text: Binding<String>,
        height: Binding<Double>,
        range: ClosedRange<Double>,
        disabled: Bool
    ) -> some View {
        VStack(spacing: 0) {
            TextEditor(text: text)
                .font(.body)
                .frame(height: CGFloat(clampedHeight(height.wrappedValue, in: range)))
                .disabled(disabled)
                .scrollContentBackground(.hidden)

            VerticalResizeGrip(height: height, range: range)
        }
        .background(Color(nsColor: .textBackgroundColor), in: RoundedRectangle(cornerRadius: 8))
        .overlay(RoundedRectangle(cornerRadius: 8).stroke(.quaternary))
        .onAppear {
            height.wrappedValue = clampedHeight(height.wrappedValue, in: range)
        }
    }

    private func clampedHeight(_ value: Double, in range: ClosedRange<Double>) -> Double {
        min(max(value, range.lowerBound), range.upperBound)
    }

    private var sourceAudioSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Source Audio")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Text("optional — generates a cover")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                Spacer()
                if sourceAudioURL != nil {
                    Button("Clear") { clearSourceAudio() }
                        .buttonStyle(.borderless)
                        .controlSize(.mini)
                        .foregroundStyle(.secondary)
                }
            }

            if let url = sourceAudioURL {
                // Show selected file name
                HStack(spacing: 8) {
                    Image(systemName: "waveform")
                        .foregroundStyle(Color.accentColor)
                    Text(url.lastPathComponent)
                        .font(.callout)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    Spacer()
                    Button {
                        showingAudioImporter = true
                    } label: {
                        Image(systemName: "arrow.triangle.2.circlepath")
                            .font(.caption)
                    }
                    .buttonStyle(.borderless)
                    .help("Replace source audio")
                }
                .padding(.horizontal, 10)
                .padding(.vertical, 8)
                .background(.secondary.opacity(0.08), in: RoundedRectangle(cornerRadius: 8))
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(.secondary.opacity(0.2), lineWidth: 1)
                )

                // Cover strength slider (only shown when audio is set)
                HStack(spacing: 8) {
                    Text("Rework")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .frame(width: 56, alignment: .leading)
                    Slider(value: $coverStrength, in: 0.0...1.0, step: 0.05)
                        .frame(maxWidth: 200)
                        .help("Lower stays closer to the source recording; higher gives the model more freedom to reinterpret it.")
                    Text(String(format: "%.0f%%", coverStrength * 100))
                        .font(.caption.monospacedDigit())
                        .frame(width: 36, alignment: .trailing)
                    Text(coverStrengthHint)
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                    Spacer()
                }

            } else {
                // Drop zone / file picker button
                Button {
                    showingAudioImporter = true
                } label: {
                    HStack(spacing: 8) {
                        Image(systemName: "music.note")
                            .font(.body)
                            .foregroundStyle(isDropTargeted ? Color.accentColor : Color.secondary)
                        Text(isDropTargeted ? "Drop to use as source" : "Drop audio or click to browse")
                            .font(.callout)
                            .foregroundStyle(isDropTargeted ? Color.accentColor : Color.secondary)
                        Spacer()
                    }
                    .padding(.horizontal, 12)
                    .padding(.vertical, 10)
                    .frame(maxWidth: .infinity)
                    .background(
                        isDropTargeted
                            ? Color.accentColor.opacity(0.08)
                            : Color.secondary.opacity(0.04),
                        in: RoundedRectangle(cornerRadius: 8)
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: 8)
                            .stroke(
                                isDropTargeted ? Color.accentColor.opacity(0.6) : Color.secondary.opacity(0.25),
                                style: StrokeStyle(lineWidth: 1, dash: [5, 3])
                            )
                    )
                }
                .buttonStyle(.plain)
                .onDrop(of: [UTType.audio, UTType.fileURL], isTargeted: $isDropTargeted) { providers in
                    guard let provider = providers.first else { return false }
                    _ = provider.loadInPlaceFileRepresentation(
                        forTypeIdentifier: UTType.audio.identifier
                    ) { url, _, _ in
                        guard let url else { return }
                        let result = Result { try Self.copySourceAudioIntoAppContainer(from: url) }
                        Task { @MainActor in
                            switch result {
                            case .success(let importedURL):
                                setImportedSourceAudio(importedURL)
                            case .failure(let error):
                                sourceAudioImportError = error.localizedDescription
                            }
                        }
                    }
                    return true
                }
            }
        }
        .fileImporter(
            isPresented: $showingAudioImporter,
            allowedContentTypes: [.audio],
            allowsMultipleSelection: false
        ) { result in
            switch result {
            case .success(let urls):
                if let url = urls.first {
                    Task { await importSourceAudio(url) }
                }
            case .failure(let error):
                sourceAudioImportError = error.localizedDescription
            }
        }
    }

    @MainActor
    private func importSourceAudio(_ url: URL) async {
        do {
            let importedURL = try await Task.detached(priority: .userInitiated) {
                try Self.copySourceAudioIntoAppContainer(from: url)
            }.value
            setImportedSourceAudio(importedURL)
        } catch {
            sourceAudioImportError = error.localizedDescription
        }
    }

    @MainActor
    private func setImportedSourceAudio(_ importedURL: URL) {
        let previousURL = sourceAudioURL
        sourceAudioURL = importedURL
        sourceAudioImportError = nil
        removeImportedSourceAudio(previousURL)
    }

    private func clearSourceAudio() {
        removeImportedSourceAudio(sourceAudioURL)
        sourceAudioURL = nil
    }

    private func removeImportedSourceAudio(_ url: URL?) {
        guard let url, Self.isImportedSourceAudio(url) else {
            return
        }
        try? FileManager.default.removeItem(at: url)
    }

    nonisolated private static func copySourceAudioIntoAppContainer(from sourceURL: URL) throws -> URL {
        if isImportedSourceAudio(sourceURL) {
            return sourceURL
        }

        let didStartAccessing = sourceURL.startAccessingSecurityScopedResource()
        defer {
            if didStartAccessing {
                sourceURL.stopAccessingSecurityScopedResource()
            }
        }

        let fileManager = FileManager.default
        let importsDirectory = try sourceAudioImportsDirectory()
        try fileManager.createDirectory(at: importsDirectory, withIntermediateDirectories: true)

        let destinationURL = uniqueSourceAudioURL(
            for: sourceURL,
            in: importsDirectory
        )
        try fileManager.copyItem(at: sourceURL, to: destinationURL)
        return destinationURL
    }

    nonisolated private static func sourceAudioImportsDirectory() throws -> URL {
        AppStorageLocations.sourceAudioImportsDirectory
    }

    nonisolated private static func uniqueSourceAudioURL(for sourceURL: URL, in directory: URL) -> URL {
        let rawBaseName = sourceURL.deletingPathExtension().lastPathComponent
        let baseName = sanitizedFileComponent(rawBaseName, fallback: "source-audio")
        let rawExtension = sourceURL.pathExtension
        let fileExtension = sanitizedFileComponent(rawExtension, fallback: "audio")
        let shortID = UUID().uuidString.prefix(8).lowercased()
        return directory.appending(path: "\(baseName)-\(shortID).\(fileExtension)")
    }

    nonisolated private static func sanitizedFileComponent(_ value: String, fallback: String) -> String {
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: " -_"))
        let scalars = value.unicodeScalars.map { scalar in
            allowed.contains(scalar) ? Character(scalar) : "-"
        }
        let sanitized = String(scalars)
            .trimmingCharacters(in: CharacterSet(charactersIn: " -_"))
        return sanitized.isEmpty ? fallback : sanitized
    }

    nonisolated private static func isImportedSourceAudio(_ url: URL) -> Bool {
        guard let importsDirectory = try? sourceAudioImportsDirectory() else {
            return false
        }
        return url.standardizedFileURL.path(percentEncoded: false)
            .hasPrefix(importsDirectory.standardizedFileURL.path(percentEncoded: false) + "/")
    }

    private var coverStrengthHint: String {
        switch coverStrength {
        case ..<0.25: return "(closest to source)"
        case ..<0.5:  return "(light reinterpret)"
        case ..<0.75: return "(balanced)"
        default:      return "(most creative)"
        }
    }

    private var controlsBar: some View {
        HStack {
            Button(sourceAudioURL != nil ? "Generate Cover" : "Generate") {
                syncPrimaryAdapter(id: selectedAdapterID)
                let request = GenerationRequest(
                    text: promptText,
                    lyrics: lyricsText,
                    referenceAudioURL: sourceAudioURL,
                    duration: effectiveDuration,
                    seed: generationParameters.seed,
                    method: method,
                    guidanceScale: Float(generationParameters.guidanceScale),
                    guidancePreset: generationParameters.guidancePreset,
                    precision: precision,
                    stepsOverride: Int(stepsOverrideText.trimmingCharacters(in: .whitespacesAndNewlines)),
                    shiftOverride: Float(shiftOverride),
                    resolvedMode: resolvedMode,
                    pipelineOverride: generationParameters.pipelineOverride,
                    coverStrength: Float(coverStrength)
                )
                // Resolve the full adapter stack and hand it to the service so
                // LoRACoordinator.ensureStack() can apply per-adapter strengths.
                generationService.adapterStack = stackManager.allStackedAdapters
                    .map { (id: $0.adapter.id, weight: $0.strength) }
                generationService.generate(request: request)
            }
            .buttonStyle(.borderedProminent)
            .disabled(!canGenerate)
            .help(generateButtonHelp)

            if case .generating = generationService.state {
                Button("Cancel", role: .destructive) {
                    generationService.cancel()
                }
                .buttonStyle(.bordered)
            }

            if trainingService.isTrainingActive {
                Text("Training in progress…")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Spacer()

            Button {
                openWindow(id: "generationAnalyzer")
            } label: {
                Label("Analyzer", systemImage: "waveform.path.ecg")
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
            .help("Open the live generation analyzer")

            // Save as Preset
            Button {
                presetName = ""
                showingSavePreset = true
            } label: {
                Image(systemName: "square.and.arrow.down")
            }
            .buttonStyle(.bordered)
            .help("Save current settings as a preset")

            if let eta = estimatedTime {
                Text("~\(Int(eta))s remaining")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private var adapterPickerSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            AdapterPicker(
                selectedAdapterID: $selectedAdapterID,
                activeModelCategory: .aceStepDiT
            )
            .onChange(of: selectedAdapterID) { _, newValue in
                persistedAdapterID = newValue ?? ""
                generationService.selectedAdapterID = newValue
                syncPrimaryAdapter(id: newValue)
                applyXLGuidanceRecipeIfNeeded()
            }
            .onAppear {
                restorePersistedAdapterSelection()
            }

            if selectedAdapterID != nil {
                primaryAdapterStrengthRow
            }

            // Multi-adapter stacking controls
            if selectedAdapterID != nil || stackManager.isStacking {
                AdapterStackView(
                    stackManager: stackManager,
                    activeModelCategory: .aceStepDiT
                )
            }
        }
    }

    private var primaryAdapterStrengthRow: some View {
        HStack(spacing: 8) {
            Text("Adapter Gain")
                .font(.caption)
                .foregroundStyle(.secondary)
                .frame(width: 96, alignment: .leading)

            Slider(
                value: Binding(
                    get: { Double(stackManager.primaryStrength) },
                    set: { newValue in
                        let strength = Float(newValue)
                        stackManager.setPrimaryStrength(strength)
                        persistedAdapterStrength = Double(stackManager.primaryStrength)
                    }
                ),
                in: Double(AdapterStrengthLimits.minimum)...Double(AdapterStrengthLimits.maximum),
                step: 0.10
            )
            .frame(maxWidth: 240)
            .help("1.0x is normal strength. Higher values boost weak adapters at generation time.")

            Text(String(format: "%.2fx", stackManager.primaryStrength))
                .font(.caption.monospacedDigit())
                .frame(width: 48, alignment: .trailing)

            if isXLGenerationContext {
                Text("XL defaults to Constant guidance for full-schedule CFG")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }

            Spacer()
        }
    }

    private var advancedSection: some View {
        DisclosureGroup(isExpanded: $advancedExpanded) {
            VStack(alignment: .leading, spacing: 12) {
                // Precision
                HStack {
                    Text("Precision")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .frame(width: 80, alignment: .leading)
                    Picker("", selection: $precision) {
                        ForEach(Precision.allCases, id: \.self) { p in
                            Text(p.rawValue).tag(p)
                        }
                    }
                    .pickerStyle(.segmented)
                    .labelsHidden()
                    .frame(maxWidth: 220)
                    Spacer()
                    Text(precision == .float16 ? "Faster, less RAM" : "Slower, more accurate")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }

                // Steps override
                HStack {
                    Text("Steps")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .frame(width: 80, alignment: .leading)
                    TextField("auto", text: $stepsOverrideText)
                        .textFieldStyle(.roundedBorder)
                        .frame(maxWidth: 80)
                    Text("(blank = auto: 8 Turbo / 50 Base, SFT, or XL)")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                    Spacer()
                }

                // Guidance schedule
                HStack(alignment: .top) {
                    Text("Guidance")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .frame(width: 80, alignment: .leading)
                    VStack(alignment: .leading, spacing: 4) {
                        Picker("", selection: $generationParameters.guidancePreset) {
                            ForEach(GuidancePreset.allCases) { preset in
                                Text(preset.displayName).tag(preset)
                            }
                        }
                        .pickerStyle(.menu)
                        .labelsHidden()
                        .frame(maxWidth: 180)
                        Text(generationParameters.guidancePreset.detail)
                            .font(.caption2)
                            .foregroundStyle(.tertiary)
                        if generationParameters.guidancePreset.overridesGuidanceScale,
                           let override = generationParameters.guidancePreset.guidanceScaleOverride {
                            Text("Ignores sidebar Guidance field; effective scale \(String(format: "%.1f", override))")
                                .font(.caption2)
                                .foregroundStyle(.orange)
                        }
                        if isXLGenerationContext,
                           generationParameters.guidancePreset == .softLate {
                            Text("Soft Late often degrades XL quality; Constant is recommended")
                                .font(.caption2)
                                .foregroundStyle(.orange)
                        }
                    }
                    Spacer()
                }

                // Shift
                HStack {
                    Text("Shift")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .frame(width: 80, alignment: .leading)
                    Slider(value: $shiftOverride, in: 0.5...10.0, step: 0.1)
                        .frame(maxWidth: 220)
                    Text(String(format: "%.1f", shiftOverride))
                        .font(.caption.monospacedDigit())
                        .frame(width: 36, alignment: .trailing)
                    Button("Reset") { shiftOverride = 3.0 }
                        .buttonStyle(.borderless)
                        .controlSize(.mini)
                    Spacer()
                }

                // Verbose audio diagnostics
                HStack {
                    Toggle(isOn: $audioDebugVerbose) {
                        Text("Verbose audio diagnostics")
                            .font(.caption)
                    }
                    .toggleStyle(.checkbox)
                    Text(audioDebugLogPath.map { "(logs: \($0))" } ?? "(logs write to Diagnostics)")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                        .help(audioDebugLogPath ?? AppStorageLocations.diagnosticsDirectory.path)
                    Button("Clean logs") {
                        cleanDiagnosticsLogs()
                    }
                    .buttonStyle(.borderless)
                    .controlSize(.mini)
                    .help(
                        "Delete old AUDIODIAG logs (keeps last 14 days, 40 files, 64 MB; never the active log). Also runs automatically on launch."
                    )
                    Spacer()
                }
                if let diagnosticsPruneMessage {
                    Text(diagnosticsPruneMessage)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }
            .padding(.top, 8)
        } label: {
            Text("Advanced")
                .font(.caption)
                .foregroundStyle(.secondary)
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
        case .tokenizingSource:
            ProgressView("Tokenizing source audio...")
                .progressViewStyle(.linear)
        case .lmAudioCodeConditioning:
            ProgressView("Conditioning audio codes...")
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
        case .releasing:
            ProgressView("Releasing models from memory...")
                .progressViewStyle(.linear)
        case .error(let message):
            Label(message, systemImage: "exclamationmark.triangle")
                .foregroundStyle(.red)
        }
    }

    @ViewBuilder
    private var playerSection: some View {
        if let result = generationService.currentResult {
            PlayerView(fileURL: result.fileURL, precomputedPeaks: result.waveformPeaks)
        }
    }

    private var effectiveDuration: TimeInterval {
        if shouldUseStableAudio {
            return min(generationParameters.duration, 47.0)
        }
        return generationParameters.duration
    }

    private var shouldUseStableAudio: Bool {
        switch generationParameters.pipelineOverride {
        case .aceStep: return false
        case .stableAudio: return true
        case .auto: return resolvedMode == .soundFX || resolvedMode == .ambient
        }
    }

    private var canGenerate: Bool {
        // Honor train↔generate mutual exclusion from GenerationService.
        generationService.canSubmitGeneration
            && !promptText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private var generateButtonHelp: String {
        if trainingService.isTrainingActive {
            return "Cannot generate while training is running. Stop training first."
        }
        if !generationService.canSubmitGeneration {
            return "Wait for the current generation to finish or cancel it."
        }
        if promptText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return "Enter a prompt to generate."
        }
        return sourceAudioURL != nil ? "Generate a cover from the source audio" : "Generate audio"
    }

    private var estimatedTime: TimeInterval? {
        if case .generating(_, _, let eta) = generationService.state, eta > 0 {
            return eta
        }
        return nil
    }

    private func syncPrimaryAdapter(id: String?) {
        guard let id, !id.isEmpty else {
            stackManager.setPrimary(nil)
            return
        }

        let adapter = AdapterRegistry.listAdapters().first { $0.id == id }
        stackManager.setPrimary(adapter)
    }

    private func updateAudioDebugLogPath() {
        audioDebugLogPath = AudioDiagnostics.activeLogFileURL?.path
            ?? AudioDiagnostics.configuredLogDirectoryURL?.path
            ?? AppStorageLocations.diagnosticsDirectory.path
    }

    private func cleanDiagnosticsLogs() {
        let result = AudioDiagnostics.pruneLogFiles(
            in: AppStorageLocations.diagnosticsDirectory
        )
        if result.deletedCount == 0 {
            diagnosticsPruneMessage =
                "Diagnostics already clean (\(result.remainingCount) log\(result.remainingCount == 1 ? "" : "s"))."
        } else {
            let freed = ByteCountFormatter.string(
                fromByteCount: Int64(result.deletedBytes),
                countStyle: .file
            )
            diagnosticsPruneMessage =
                "Removed \(result.deletedCount) old log\(result.deletedCount == 1 ? "" : "s") (\(freed)); \(result.remainingCount) kept."
        }
    }

    private func restorePersistedAdapterSelection() {
        stackManager.setPrimaryStrength(Float(persistedAdapterStrength))

        guard !persistedAdapterID.isEmpty,
              let adapter = compatibleAdapter(id: persistedAdapterID) else {
            clearPrimaryAdapterSelection()
            return
        }

        selectedAdapterID = adapter.id
        generationService.selectedAdapterID = adapter.id
        syncPrimaryAdapter(id: adapter.id)
        applyXLGuidanceRecipeIfNeeded()
    }

    private func clearPrimaryAdapterSelection() {
        selectedAdapterID = nil
        persistedAdapterID = ""
        generationService.selectedAdapterID = nil
        syncPrimaryAdapter(id: nil)
    }

    private func compatibleAdapter(id: String?) -> LoRAAdapter? {
        guard let id, !id.isEmpty else { return nil }
        return AdapterRegistry.compatibleAdapters(for: .aceStepDiT)
            .first(where: { $0.id == id })
    }

    private func applyAdapterSelection(_ notification: Notification) {
        guard let payload = notification.userInfo?["adapters"] as? [[String: Any]] else {
            return
        }

        let installed = AdapterRegistry.listAdapters()
        stackManager.clearAll()

        guard let first = payload.first,
              let primaryID = adapterID(from: first),
              let primary = installed.first(where: { $0.id == primaryID }) else {
            selectedAdapterID = nil
            persistedAdapterID = ""
            stackManager.setPrimaryStrength(1.0)
            persistedAdapterStrength = 1.0
            generationService.selectedAdapterID = nil
            generationService.adapterStack = []
            return
        }

        let primaryStrength = adapterStrength(from: first)
        selectedAdapterID = primary.id
        persistedAdapterID = primary.id
        stackManager.setPrimary(primary)
        stackManager.setPrimaryStrength(primaryStrength)
        persistedAdapterStrength = Double(stackManager.primaryStrength)

        for entry in payload.dropFirst() {
            guard let id = adapterID(from: entry),
                  let adapter = installed.first(where: { $0.id == id }) else {
                continue
            }
            stackManager.addToStack(adapter, strength: adapterStrength(from: entry))
        }

        generationService.selectedAdapterID = selectedAdapterID
        generationService.adapterStack = stackManager.allStackedAdapters
            .map { (id: $0.adapter.id, weight: $0.strength) }
        applyXLGuidanceRecipeIfNeeded()
    }

    private func adapterID(from payload: [String: Any]) -> String? {
        payload["id"] as? String
    }

    private func adapterStrength(from payload: [String: Any]) -> Float {
        if let value = payload["strength"] as? Float {
            return value
        }
        if let value = payload["strength"] as? Double {
            return Float(value)
        }
        if let value = payload["strength"] as? NSNumber {
            return value.floatValue
        }
        if let value = payload["strength"] as? String,
           let parsed = Float(value.trimmingCharacters(in: .whitespacesAndNewlines)) {
            return parsed
        }
        return 1.0
    }

    private var isXLGenerationContext: Bool {
        if let activeDiTModelID,
           LoRAAdapter.isACEBaseModelXL(activeDiTModelID, spec: activeDiTModelSpec) {
            return true
        }
        if let selectedAdapterID,
           let adapter = AdapterRegistry.listAdapters().first(where: { $0.id == selectedAdapterID }),
           isXLAdapter(adapter) {
            return true
        }
        return stackManager.allStackedAdapters.contains { isXLAdapter($0.adapter) }
    }

    @MainActor
    private func refreshActiveDiTAndApplyXLGuidanceRecipe() async {
        let id = await appState.registry.activeModelID(for: .aceStepDiT)
        let spec = await appState.registry.activeSpec(for: .aceStepDiT)
        activeDiTModelID = id
        activeDiTModelSpec = spec
        applyXLGuidanceRecipeIfNeeded()
    }

    private func applyXLGuidanceRecipeIfNeeded() {
        let next = GuidancePreset.applyingXLRecipeIfNeeded(
            current: generationParameters.guidancePreset,
            isXLContext: isXLGenerationContext
        )
        guard next != generationParameters.guidancePreset else { return }
        generationParameters.guidancePreset = next
    }

    private func isXLAdapter(_ adapter: LoRAAdapter) -> Bool {
        guard adapter.baseModelCategory == .aceStepDiT else { return false }
        return LoRAAdapter.isACEBaseModelXL(adapter.baseModelID)
    }

    // MARK: - Presets

    private func saveCurrentAsPreset() {
        let name = presetName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty else { return }

        let resolvedModeForPreset: GenerationMode? = {
            switch modeSelection {
            case .auto: return nil
            case .manual(let mode): return mode
            }
        }()

        let preset = GenerationPreset(
            name: name,
            prompt: promptText,
            lyrics: lyricsText,
            mode: resolvedModeForPreset,
            duration: generationParameters.duration,
            seedText: generationParameters.seedText,
            guidanceScale: generationParameters.guidanceScale,
            guidancePreset: generationParameters.guidancePreset,
            pipelineOverride: generationParameters.pipelineOverride,
            method: method,
            precision: precision,
            stepsOverride: stepsOverrideText,
            shiftOverride: shiftOverride,
            adapterID: selectedAdapterID,
            adapterStrength: Double(stackManager.primaryStrength),
            coverStrength: coverStrength
        )
        presetStore.save(preset)
    }

    private func applyPreset(_ preset: GenerationPreset) {
        promptText = preset.prompt
        lyricsText = preset.lyrics
        modeSelection = preset.mode.map { .manual($0) } ?? .auto
        method = preset.method
        precision = preset.precision
        stepsOverrideText = preset.stepsOverride
        shiftOverride = preset.shiftOverride
        let adapterID = compatibleAdapter(id: preset.adapterID)?.id
        selectedAdapterID = adapterID
        persistedAdapterID = adapterID ?? ""
        generationService.selectedAdapterID = adapterID
        stackManager.setPrimaryStrength(Float(preset.adapterStrength))
        persistedAdapterStrength = preset.adapterStrength
        syncPrimaryAdapter(id: adapterID)
        coverStrength = preset.coverStrength

        // Shared parameters
        generationParameters.duration = preset.duration
        generationParameters.seedText = preset.seedText
        generationParameters.guidanceScale = preset.guidanceScale
        generationParameters.guidancePreset = preset.guidancePreset
        generationParameters.pipelineOverride = preset.pipelineOverride
    }

    /// History open: apply knobs then rebuild the full adapter stack when present.
    private func applyHistoryRestore(_ restore: HistoryGenerateRestore) {
        applyPreset(restore.preset)
        if restore.adapterStack.isEmpty {
            return
        }
        let notification = Notification(
            name: .adapterSelectionDidChange,
            object: nil,
            userInfo: ["adapters": restore.adapterStack]
        )
        applyAdapterSelection(notification)
    }
}
