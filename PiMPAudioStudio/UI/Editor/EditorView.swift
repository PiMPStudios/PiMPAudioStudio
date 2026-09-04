import AVFoundation
import OSLog
import PiMPAudioCore
import SwiftUI
import UniformTypeIdentifiers

private let editorExportLogger = Logger(
    subsystem: "com.pimpstudios.PiMPAudioStudio",
    category: "EditorExport"
)

private let editorSupportedAudioTypes: [UTType] = {
    var types: [UTType] = [.wav, .mp3, .mpeg4Audio, .aiff]
    if let flac = UTType(filenameExtension: "flac") {
        types.append(flac)
    }
    return types
}()

struct EditorView: View {
    @StateObject private var state = EditorState()
    @EnvironmentObject private var editorService: EditorService
    @EnvironmentObject private var sfxModelManager: SFXModelManager
    @State private var exportErrorMessage: String?
    /// Last `EditorState.bufferRevision` pushed to the inpaint coordinator.
    @State private var lastSyncedBufferRevision: Int = -1

    var preloadedAudioURL: URL?
    var preloadedPrompt: String?

    var body: some View {
        VStack(spacing: 0) {
            EditorToolbar(
                state: state,
                isStableAudioReady: sfxModelManager.isReady,
                onOpen: openFile,
                onExport: { exportFile(format: $0) },
                onUndo: performUndo,
                onRedo: performRedo
            )

            if state.isLoaded {
                ScrollView {
                    VStack(spacing: 16) {
                        waveformSection
                        if state.editMode == .inpaint {
                            frequencySection
                        }
                        EditorPromptArea(state: state)
                        EditorControlsBar(state: state, onExecute: executeEdit)

                        Divider()
                            .padding(.horizontal, 16)

                        DSPPaletteView(state: state) { effect, params in
                            state.applyDSPEffect(effect, parameters: params)
                        }
                    }
                    .padding(.vertical, 16)
                }
            } else {
                emptyState
            }
        }
        .onAppear {
            // Defer so History → Edit (view identity change + onAppear) does not
            // publish EditorState during the current SwiftUI update pass.
            let url = preloadedAudioURL
            let prompt = preloadedPrompt
            DispatchQueue.main.async {
                if let prompt {
                    state.originalPrompt = prompt
                }
                if let url {
                    loadFromURL(url)
                }
            }
        }
        .background {
            Group {
                Button("") {
                    if state.hasSelection, state.isLoaded {
                        state.applyDSPEffect(.trimToSelection)
                    }
                }
                .keyboardShortcut("t", modifiers: .command)

                Button("") {
                    if state.hasSelection, state.isLoaded {
                        state.applyDSPEffect(.deleteSelection)
                    }
                }
                .keyboardShortcut(.delete, modifiers: .command)

                Button("") {
                    if state.isLoaded {
                        withAnimation { state.expandedEffect = .normalize }
                    }
                }
                .keyboardShortcut("n", modifiers: [.command, .shift])

                Button("") {
                    if state.isLoaded {
                        withAnimation { state.expandedEffect = .eq }
                    }
                }
                .keyboardShortcut("e", modifiers: [.command, .shift])

                Button("") {
                    if state.isLoaded {
                        withAnimation { state.expandedEffect = .reverb }
                    }
                }
                .keyboardShortcut("r", modifiers: [.command, .shift])

                Button("") {
                    if state.isLoaded {
                        withAnimation { state.expandedEffect = .highPass }
                    }
                }
                .keyboardShortcut("h", modifiers: [.command, .shift])

                Button("") {
                    if state.isLoaded {
                        withAnimation { state.expandedEffect = .trimSilence }
                    }
                }
                .keyboardShortcut("j", modifiers: [.command, .shift])
            }
            .frame(width: 0, height: 0)
            .opacity(0)
        }
        .alert("Could Not Export Audio", isPresented: Binding(
            get: { exportErrorMessage != nil },
            set: {
                if !$0 {
                    exportErrorMessage = nil
                }
            }
        )) {
            Button("OK", role: .cancel) { exportErrorMessage = nil }
        } message: {
            Text(exportErrorMessage ?? "")
        }
        .alert("Editor Error", isPresented: Binding(
            get: { state.operationErrorMessage != nil },
            set: {
                if !$0 {
                    state.operationErrorMessage = nil
                }
            }
        )) {
            Button("OK", role: .cancel) { state.operationErrorMessage = nil }
        } message: {
            Text(state.operationErrorMessage ?? "")
        }
    }

    private var waveformSection: some View {
        VStack(spacing: 8) {
            EditorWaveformView(
                peaks: state.waveformPeaks,
                duration: state.duration,
                playbackProgress: state.playbackProgress,
                selectionStart: $state.selectionStart,
                selectionEnd: $state.selectionEnd,
                isSelectionEnabled: state.editMode == .inpaint,
                onSeek: { fraction in
                    state.seekPreview(to: fraction)
                }
            )
            .padding(.horizontal, 16)

            // Transport bar
            HStack(spacing: 10) {
                Button {
                    if state.isPlayingPreview {
                        state.pausePreview()
                    } else {
                        state.playPreview()
                    }
                } label: {
                    Image(systemName: state.isPlayingPreview
                        ? "pause.circle.fill" : "play.circle.fill")
                        .font(.system(size: 26))
                }
                .buttonStyle(.plain)

                Text(formatTime(state.previewCurrentTime))
                    .font(.caption2.monospacedDigit())
                    .foregroundStyle(.secondary)
                    .frame(minWidth: 36, alignment: .leading)

                Spacer()

                if state.hasSelection {
                    Text("⌗ \(formatTime(state.selectionStart))–\(formatTime(state.selectionEnd))")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }

                Spacer()

                Text(formatTime(state.duration))
                    .font(.caption2.monospacedDigit())
                    .foregroundStyle(.secondary)
                    .frame(minWidth: 36, alignment: .trailing)
            }
            .padding(.horizontal, 16)
        }
    }

    private var frequencySection: some View {
        let cutRange = EditorState.frequencyCutRange
        let minBandwidth = EditorState.minimumFrequencyBandwidthHz
        let lowSliderRange = cutRange.lowerBound ... (cutRange.upperBound - minBandwidth)
        let highSliderRange = (cutRange.lowerBound + minBandwidth) ... cutRange.upperBound

        return DisclosureGroup("Frequency Constraint") {
            VStack(spacing: 8) {
                Toggle("Enable frequency band", isOn: $state.frequencyBandEnabled)
                if state.frequencyBandEnabled {
                    HStack {
                        VStack(alignment: .leading) {
                            Text("Low Cut: \(Int(state.lowCutHz)) Hz").font(.caption)
                            Slider(value: state.lowCutHzBinding, in: lowSliderRange)
                                .tint(.orange)
                        }
                        VStack(alignment: .leading) {
                            Text("High Cut: \(Int(state.highCutHz)) Hz").font(.caption)
                            Slider(value: state.highCutHzBinding, in: highSliderRange)
                                .tint(.orange)
                        }
                    }
                }
            }
            .padding(.vertical, 8)
        }
        .padding(.horizontal, 16)
    }

    private var emptyState: some View {
        VStack(spacing: 16) {
            Spacer()
            Image(systemName: "waveform.badge.plus")
                .font(.system(size: 48))
                .foregroundStyle(.secondary)
            Text("Open an audio file to start editing")
                .font(.headline).foregroundStyle(.secondary)
            Button("Open File") { openFile() }
                .buttonStyle(.bordered)
            Spacer()
        }
    }

    private func openFile() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = editorSupportedAudioTypes
        panel.allowsMultipleSelection = false
        panel.begin { response in
            guard response == .OK, let url = panel.url else { return }
            loadFromURL(url)
        }
    }

    private func loadFromURL(_ url: URL) {
        let fileName = url.lastPathComponent
        // Show empty state while decoding without racing the appear pass.
        state.deferUIUpdate { editor in
            editor.fileName = fileName
            editor.operationErrorMessage = nil
            editor.isLoaded = false
        }

        // Decode the file off the main actor — long WAVs can be multi-GB.
        Task {
            let loadResult: Result<AudioBuffer, Error> = await Task.detached(priority: .userInitiated) {
                Result {
                    let audioFile = try AVAudioFile(forReading: url)
                    let format = audioFile.processingFormat
                    let frameCount = AVAudioFrameCount(audioFile.length)

                    guard let pcmBuffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
                        throw NSError(
                            domain: "PiMPAudioStudio.Editor",
                            code: 1,
                            userInfo: [
                                NSLocalizedDescriptionKey:
                                    "AVFoundation could not create an audio buffer.",
                            ]
                        )
                    }
                    try audioFile.read(into: pcmBuffer)
                    return AudioBufferConverter.fromPCMBuffer(pcmBuffer)
                }
            }.value

            switch loadResult {
            case let .success(buffer):
                let loadGeneration = await MainActor.run { () -> Int in
                    state.applyLoadedBuffer(buffer, fileName: fileName)
                    return state.documentGeneration
                }

                // Store PCM only — full-track VAE encode is deferred until first
                // Inpaint/Repaint so History → Edit / polish does not park 40–50 GiB
                // of MLX GPU cache (see Diagnostics editor/loadAudio).
                guard let coordinator = editorService.coordinator else {
                    await MainActor.run {
                        guard state.documentGeneration == loadGeneration else { return }
                        state.deferUIUpdate { editor in
                            editor.operationErrorMessage =
                                "Audio opened for playback, but the editor pipeline is not ready for AI edits."
                        }
                    }
                    return
                }
                let mlxArray = await Task.detached(priority: .userInitiated) {
                    AudioBufferConverter.toMLXArray(buffer)
                }.value
                do {
                    let stillCurrent = await MainActor.run { () -> Bool in
                        state.documentGeneration == loadGeneration
                    }
                    guard stillCurrent else { return }
                    _ = try await coordinator.loadAudio(
                        samples: mlxArray,
                        sampleRate: buffer.sampleRate,
                        precision: .float32
                    )
                    await MainActor.run {
                        guard state.documentGeneration == loadGeneration else { return }
                        lastSyncedBufferRevision = state.bufferRevision
                    }
                } catch {
                    await MainActor.run {
                        guard state.documentGeneration == loadGeneration else { return }
                        state.deferUIUpdate { editor in
                            editor.operationErrorMessage =
                                "Audio opened for playback, but editor prep failed: \(error.localizedDescription)"
                        }
                    }
                }

            case let .failure(error):
                await MainActor.run {
                    state.deferUIUpdate { editor in
                        editor.isLoaded = false
                        editor.operationErrorMessage =
                            "Could not load \(fileName): \(error.localizedDescription)"
                    }
                }
            }
        }
    }

    private func executeEdit() {
        guard let coordinator = editorService.coordinator else {
            state.progress = .failed("Editor pipeline not ready")
            return
        }
        guard !state.isProcessing else { return }

        let request: InpaintRequest
        do {
            request = try state.buildRequest()
        } catch {
            state.operationErrorMessage = error.localizedDescription
            state.progress = .failed(error.localizedDescription)
            return
        }

        state.saveUndoSnapshot() // save pre-inpaint buffer so AI edits are undoable
        state.isProcessing = true
        state.progress = .buildingMask
        let sourceRevision = state.bufferRevision
        let sourceDocument = state.documentGeneration

        Task { @MainActor in
            // If local DSP/load changed PCM since last sync, push it so AI encodes
            // the polished buffer. Skip when unchanged so iterative Inpaint reuses latents.
            if state.bufferRevision != lastSyncedBufferRevision, let buffer = state.audioBuffer {
                let mlx = await Task.detached(priority: .userInitiated) {
                    AudioBufferConverter.toMLXArray(buffer)
                }.value
                await coordinator.syncSourceAudio(samples: mlx, sampleRate: buffer.sampleRate)
                lastSyncedBufferRevision = state.bufferRevision
            }

            var succeeded = false
            for await progress in await coordinator.edit(request: request) {
                state.progress = progress
                switch progress {
                case .complete:
                    succeeded = true
                    state.isProcessing = false
                case .failed:
                    state.isProcessing = false
                default:
                    break
                }
            }
            // Only refresh the buffer after a successful edit. Pulling audio on
            // failure rewrote the waveform from stale coordinator state and
            // made it look like a no-op "edit" had been applied.
            guard succeeded,
                  state.documentGeneration == sourceDocument,
                  state.bufferRevision == sourceRevision,
                  let updatedAudio = await coordinator.getCurrentAudio() else { return }
            let newBuffer = AudioBufferConverter.fromMLXArray(
                updatedAudio,
                sampleRate: state.sampleRate
            )
            state.audioBuffer = newBuffer
            state.duration = newBuffer.duration
            state.noteBufferMutation()
            lastSyncedBufferRevision = state.bufferRevision
            state.refreshWaveformPeaks(from: newBuffer)
            state.refreshLevelMetrics()
        }
    }

    private func performUndo() {
        state.undoDSP()
    }

    private func performRedo() {
        state.redoDSP()
    }

    private func exportFile(format: ExportFormat) {
        guard let buffer = state.audioBuffer else { return }

        let panel = NSSavePanel()
        panel.allowedContentTypes = [format.contentType]
        panel.canCreateDirectories = true
        let baseName = (state.fileName as NSString).deletingPathExtension
        panel.nameFieldStringValue = "\(baseName.isEmpty ? "edited" : baseName)-edited.\(format.fileExtension)"

        panel.begin { [self] response in
            guard response == .OK, let url = panel.url else { return }
            Task {
                do {
                    _ = try await EditorAudioExporter.export(
                        buffer: buffer,
                        format: format,
                        to: url
                    )
                } catch {
                    editorExportLogger.error("Editor export failed: \(error.localizedDescription, privacy: .public)")
                    await MainActor.run {
                        exportErrorMessage = error.localizedDescription
                    }
                }
            }
        }
    }
}

enum EditorAudioExportError: LocalizedError {
    case invalidBuffer(String)
    case writeFailed(URL, Error)
    case exportSessionUnavailable
    case compressedExportFailed(URL, Error?)

    var errorDescription: String? {
        switch self {
        case let .invalidBuffer(reason):
            return "Could not export edited audio because the editor buffer is invalid: \(reason)"
        case let .writeFailed(url, error):
            return "Could not write edited audio to \(url.path(percentEncoded: false)): \(error.localizedDescription)"
        case .exportSessionUnavailable:
            return "Could not create the native M4A export session."
        case let .compressedExportFailed(url, error):
            let detail = error?.localizedDescription ?? "AVFoundation did not report a detailed reason."
            return "Could not export edited audio to \(url.path(percentEncoded: false)): \(detail)"
        }
    }
}

enum EditorAudioExporter {
    @discardableResult
    static func export(
        buffer: AudioBuffer,
        format: ExportFormat,
        to selectedURL: URL
    ) async throws -> URL {
        let destinationURL = normalizedDestinationURL(for: selectedURL, format: format)
        let didStartAccessing = destinationURL.startAccessingSecurityScopedResource()
        defer {
            if didStartAccessing {
                destinationURL.stopAccessingSecurityScopedResource()
            }
        }

        do {
            switch format {
            case .wav:
                try writeWAV(buffer: buffer, to: destinationURL)
            case .aiff:
                try writeAIFF(buffer: buffer, to: destinationURL)
            case .m4a:
                try await exportM4A(buffer: buffer, to: destinationURL)
            }
        } catch let error as EditorAudioExportError {
            throw error
        } catch {
            throw EditorAudioExportError.writeFailed(destinationURL, error)
        }

        return destinationURL
    }

    static func normalizedDestinationURL(for url: URL, format: ExportFormat) -> URL {
        let expectedExtension = format.fileExtension.lowercased()
        let currentExtension = url.pathExtension.lowercased()
        if currentExtension == expectedExtension {
            return url
        }
        if currentExtension.isEmpty {
            return url.appendingPathExtension(expectedExtension)
        }
        return url
            .deletingPathExtension()
            .appendingPathExtension(expectedExtension)
    }

    private static func writeWAV(buffer: AudioBuffer, to destinationURL: URL) throws {
        guard buffer.channels > 0 else {
            throw EditorAudioExportError.invalidBuffer("channel count is zero")
        }
        guard buffer.sampleRate > 0 else {
            throw EditorAudioExportError.invalidBuffer("sample rate is zero")
        }

        let bytesPerSample = MemoryLayout<Int16>.size
        let bytesPerFrame = buffer.channels * bytesPerSample
        guard buffer.data.count.isMultiple(of: bytesPerFrame) else {
            throw EditorAudioExportError.invalidBuffer("PCM data is not aligned to the channel count")
        }
        guard buffer.data.count <= Int(UInt32.max - 36) else {
            throw EditorAudioExportError.invalidBuffer("audio is too large for a standard WAV file")
        }

        do {
            if !FileManager.default.fileExists(atPath: destinationURL.path(percentEncoded: false)) {
                _ = FileManager.default.createFile(atPath: destinationURL.path(percentEncoded: false), contents: nil)
            }

            let handle = try FileHandle(forWritingTo: destinationURL)
            defer { try? handle.close() }
            try handle.truncate(atOffset: 0)
            try handle.write(contentsOf: wavHeader(for: buffer))
            try handle.write(contentsOf: buffer.data)
        } catch {
            throw EditorAudioExportError.writeFailed(destinationURL, error)
        }
    }

    private static func writeAIFF(buffer: AudioBuffer, to destinationURL: URL) throws {
        do {
            let pcmBuffer = try AudioBufferConverter.toPCMBuffer(buffer)
            let outputFile = try AVAudioFile(forWriting: destinationURL, settings: pcmBuffer.format.settings)
            try outputFile.write(from: pcmBuffer)
        } catch {
            throw EditorAudioExportError.writeFailed(destinationURL, error)
        }
    }

    /// Export to M4A (AAC) via a temp WAV + AVAssetExportSession.
    private static func exportM4A(buffer: AudioBuffer, to destinationURL: URL) async throws {
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("wav")
        try writeWAV(buffer: buffer, to: tmp)
        defer { try? FileManager.default.removeItem(at: tmp) }

        if FileManager.default.fileExists(atPath: destinationURL.path(percentEncoded: false)) {
            try FileManager.default.removeItem(at: destinationURL)
        }

        let asset = AVURLAsset(url: tmp)
        guard let session = AVAssetExportSession(asset: asset, presetName: AVAssetExportPresetAppleM4A) else {
            throw EditorAudioExportError.exportSessionUnavailable
        }
        do {
            try await session.export(to: destinationURL, as: .m4a)
        } catch {
            throw EditorAudioExportError.compressedExportFailed(destinationURL, error)
        }
    }

    private static func wavHeader(for buffer: AudioBuffer) -> Data {
        let bitsPerSample: UInt16 = 16
        let channelCount = UInt16(buffer.channels)
        let blockAlign = channelCount * bitsPerSample / 8
        let byteRate = UInt32(buffer.sampleRate) * UInt32(blockAlign)
        let dataSize = UInt32(buffer.data.count)
        let fileSize = UInt32(36) + dataSize

        var header = Data(capacity: 44)
        appendString("RIFF", to: &header)
        appendLittleEndian(fileSize, to: &header)
        appendString("WAVE", to: &header)
        appendString("fmt ", to: &header)
        appendLittleEndian(UInt32(16), to: &header)
        appendLittleEndian(UInt16(1), to: &header)
        appendLittleEndian(channelCount, to: &header)
        appendLittleEndian(UInt32(buffer.sampleRate), to: &header)
        appendLittleEndian(byteRate, to: &header)
        appendLittleEndian(blockAlign, to: &header)
        appendLittleEndian(bitsPerSample, to: &header)
        appendString("data", to: &header)
        appendLittleEndian(dataSize, to: &header)
        return header
    }

    private static func appendString(_ value: String, to data: inout Data) {
        data.append(contentsOf: value.utf8)
    }

    private static func appendLittleEndian(_ value: some FixedWidthInteger, to data: inout Data) {
        var littleEndian = value.littleEndian
        withUnsafeBytes(of: &littleEndian) { data.append(contentsOf: $0) }
    }
}

extension EditorView {
    private func formatTime(_ time: TimeInterval) -> String {
        let minutes = Int(time) / 60
        let seconds = Int(time) % 60
        return String(format: "%d:%02d", minutes, seconds)
    }
}
