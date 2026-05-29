import SwiftUI
import AppKit

struct DatasetReviewView: View {
    @EnvironmentObject private var playerService: AudioPlayerService

    let datasetDirectory: URL
    var onOpenInTraining: ([TrainingImportedFile]) -> Void
    var onPackageUpdated: () -> Void = {}

    @StateObject private var enrichment = LocalDatasetEnrichmentService()
    @State private var document: DatasetPackageDocument?
    @State private var selectedClipID: String?
    @State private var notice: String?
    @State private var errorMessage: String?
    @State private var isSaving = false
    @State private var isExporting = false

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            if let document {
                reviewBody(document)
            } else if let errorMessage {
                statusMessage(
                    "Could Not Load Dataset",
                    message: errorMessage,
                    systemImage: "exclamationmark.triangle",
                    color: .orange
                )
            } else {
                ProgressView()
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .onAppear(perform: loadDocument)
        .onChange(of: datasetDirectory) { _, _ in loadDocument() }
        .onChange(of: enrichment.phase) { _, phase in
            if case .complete = phase {
                loadDocument()
                onPackageUpdated()
            }
        }
    }

    private var header: some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                Text(document?.name ?? datasetDirectory.lastPathComponent)
                    .font(.headline)
                    .lineLimit(1)
                Text(datasetDirectory.path(percentEncoded: false))
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }

            Spacer()

            Button {
                NSWorkspace.shared.open(datasetDirectory)
            } label: {
                Label("Finder", systemImage: "folder")
            }
            .buttonStyle(.bordered)
            .controlSize(.small)

            Button("Reload") { loadDocument() }
                .buttonStyle(.bordered)
                .controlSize(.small)

            Button {
                importSidecars()
            } label: {
                Label("Import Sidecars", systemImage: "square.and.arrow.down.on.square")
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
            .disabled(document == nil || isBusy)

            Button {
                markAllApproved()
            } label: {
                Label("Approve All", systemImage: "checkmark.seal")
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
            .disabled(document == nil || isBusy)

            Button {
                saveDocument()
            } label: {
                Label("Save", systemImage: "square.and.arrow.down")
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.small)
            .disabled(document?.hasUnsavedChanges != true || isBusy)

            Button {
                exportPackage()
            } label: {
                Label("Export Zip", systemImage: "square.and.arrow.up")
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
            .disabled(document?.isTrainingReady != true || isBusy)
        }
        .padding()
    }

    @ViewBuilder
    private func reviewBody(_ document: DatasetPackageDocument) -> some View {
        VStack(spacing: 0) {
            readinessStrip(document)
            Divider()
            HSplitView {
                clipList(document)
                    .frame(minWidth: 260, idealWidth: 320, maxWidth: 420)
                clipEditor
                    .frame(minWidth: 460)
            }
        }
    }

    private func readinessStrip(_ document: DatasetPackageDocument) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 10) {
                Label(
                    document.isTrainingReady ? "Ready for Training" : "Needs Review",
                    systemImage: document.isTrainingReady ? "checkmark.circle.fill" : "exclamationmark.triangle.fill"
                )
                .font(.subheadline)
                .fontWeight(.semibold)
                .foregroundStyle(document.isTrainingReady ? .green : .orange)

                metric("\(document.clips.count)", "Clips")
                metric("\(document.approvedCount)", "Approved")
                metric("\(document.blockerCount)", "Blockers")
                metric("\(document.warningCount)", "Warnings")

                Spacer()

                Button {
                    startEnrichment()
                } label: {
                    Label("Enrich Dataset", systemImage: "waveform.path.ecg")
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .disabled(isBusy)

                Button {
                    openInTraining()
                } label: {
                    Label("Open in Training", systemImage: "graduationcap")
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.small)
                .disabled(!document.isTrainingReady || isBusy)
            }

            switch enrichment.phase {
            case .idle:
                if let notice {
                    Text(notice)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                } else {
                    Text(document.validation.summary)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            case .running(let step):
                HStack(spacing: 8) {
                    ProgressView()
                        .controlSize(.small)
                    Text(step.title)
                        .font(.caption)
                    Spacer()
                    Button("Stop", role: .destructive) { enrichment.cancel() }
                        .controlSize(.mini)
                }
            case .complete(let result):
                Text(result.summary)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            case .failed(let message):
                Text(message)
                    .font(.caption)
                    .foregroundStyle(.orange)
                    .textSelection(.enabled)
            }
        }
        .padding(.horizontal)
        .padding(.vertical, 10)
    }

    private func clipList(_ document: DatasetPackageDocument) -> some View {
        List(selection: $selectedClipID) {
            ForEach(document.clips) { clip in
                clipRow(clip)
                    .tag(clip.id)
            }
        }
        .listStyle(.sidebar)
    }

    private func clipRow(_ clip: DatasetClipReview) -> some View {
        let issues = TrainingDatasetPackageEditor.issues(for: clip, datasetDirectory: datasetDirectory)
        let blockers = issues.filter { $0.severity == .blocker }.count
        let warnings = issues.filter { $0.severity == .warning }.count

        return HStack(alignment: .top, spacing: 8) {
            Image(systemName: clip.isApproved ? "checkmark.circle.fill" : "waveform")
                .foregroundStyle(clip.isApproved ? .green : .secondary)
                .frame(width: 18)
            VStack(alignment: .leading, spacing: 3) {
                Text(clip.displayName)
                    .font(.callout)
                    .lineLimit(1)
                    .truncationMode(.middle)
                Text(clip.caption.isEmpty ? "No caption" : clip.caption)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
                HStack(spacing: 6) {
                    badge(clip.role.isEmpty ? "full_mix" : clip.role)
                    if let duration = clip.durationSeconds {
                        Text("\(Int(duration.rounded()))s")
                            .font(.caption2)
                            .foregroundStyle(.tertiary)
                    }
                    if blockers > 0 {
                        badge("\(blockers) blocker\(blockers == 1 ? "" : "s")", color: .red)
                    } else if warnings > 0 {
                        badge("\(warnings) warning\(warnings == 1 ? "" : "s")", color: .orange)
                    }
                }
            }
            Spacer(minLength: 0)
        }
        .padding(.vertical, 4)
    }

    @ViewBuilder
    private var clipEditor: some View {
        if let index = selectedClipIndex {
            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    editorHeader(index)
                    audioPreview(index)
                    issueSection(for: document!.clips[index])
                    metadataEditor(index)
                    assetsSection(document!.clips[index])
                }
                .padding()
            }
        } else {
            statusMessage(
                "Select a Clip",
                message: "Choose a clip to review captions, lyrics, metadata, and readiness warnings.",
                systemImage: "waveform",
                color: .secondary
            )
        }
    }

    private func editorHeader(_ index: Int) -> some View {
        let clip = document!.clips[index]
        return HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text(clip.displayName)
                    .font(.title3)
                    .fontWeight(.semibold)
                Text(clip.file)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
            Spacer()
            Toggle("Approved", isOn: approvedBinding(index))
                .toggleStyle(.checkbox)
        }
    }

    @ViewBuilder
    private func audioPreview(_ index: Int) -> some View {
        if let url = document!.clips[index].audioURL(relativeTo: datasetDirectory),
           FileManager.default.fileExists(atPath: url.path(percentEncoded: false)) {
            PlayerView(fileURL: url)
                .environmentObject(playerService)
        } else {
            HStack(spacing: 8) {
                Image(systemName: "exclamationmark.triangle")
                    .foregroundStyle(.orange)
                Text("Audio file is missing.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .padding(10)
            .background(.orange.opacity(0.08))
            .clipShape(RoundedRectangle(cornerRadius: 8))
        }
    }

    private func issueSection(for clip: DatasetClipReview) -> some View {
        let issues = TrainingDatasetPackageEditor.issues(for: clip, datasetDirectory: datasetDirectory)
        return VStack(alignment: .leading, spacing: 8) {
            Text("Readiness")
                .font(.subheadline)
                .fontWeight(.semibold)
            if issues.isEmpty {
                Label("No issues found.", systemImage: "checkmark.circle")
                    .font(.caption)
                    .foregroundStyle(.green)
            } else {
                VStack(alignment: .leading, spacing: 4) {
                    ForEach(issues) { issue in
                        Label(issue.message, systemImage: issue.severity == .blocker ? "xmark.octagon" : "exclamationmark.triangle")
                            .font(.caption)
                            .foregroundStyle(issue.severity == .blocker ? .red : .orange)
                    }
                }
            }
        }
    }

    private func metadataEditor(_ index: Int) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Training Metadata")
                .font(.subheadline)
                .fontWeight(.semibold)

            VStack(alignment: .leading, spacing: 6) {
                HStack(spacing: 6) {
                    Text("Caption")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    sourceBadge(document!.clips[index].captionSource)
                }
                TextEditor(text: captionBinding(index))
                    .font(.callout)
                    .frame(minHeight: 70)
                    .scrollContentBackground(.hidden)
                    .background(.quaternary.opacity(0.3))
                    .clipShape(RoundedRectangle(cornerRadius: 8))
            }

            VStack(alignment: .leading, spacing: 6) {
                HStack(spacing: 6) {
                    Text("Lyrics / Transcript")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    sourceBadge(document!.clips[index].lyricsSource)
                }
                TextEditor(text: lyricsBinding(index))
                    .font(.callout)
                    .frame(minHeight: 110)
                    .scrollContentBackground(.hidden)
                    .background(.quaternary.opacity(0.3))
                    .clipShape(RoundedRectangle(cornerRadius: 8))
            }

            Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 8) {
                GridRow {
                    field("BPM", text: binding(index, \.bpm))
                    field("Key", text: binding(index, \.key))
                }
                GridRow {
                    field("Language", text: binding(index, \.language))
                    field("Time Sig", text: binding(index, \.timeSignature))
                }
            }

            HStack {
                Picker("Role", selection: binding(index, \.role)) {
                    Text("Full Mix").tag("full_mix")
                    Text("Vocal").tag("vocal")
                    Text("Instrumental").tag("instrumental")
                }
                .pickerStyle(.segmented)
                .frame(maxWidth: 320)
                Spacer()
            }

            VStack(alignment: .leading, spacing: 6) {
                Text("Notes")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                TextEditor(text: binding(index, \.notes))
                    .font(.callout)
                    .frame(minHeight: 70)
                    .scrollContentBackground(.hidden)
                    .background(.quaternary.opacity(0.3))
                    .clipShape(RoundedRectangle(cornerRadius: 8))
            }
        }
    }

    private func assetsSection(_ clip: DatasetClipReview) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Artifacts")
                .font(.subheadline)
                .fontWeight(.semibold)
            Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 6) {
                GridRow {
                    Text("Audio")
                        .foregroundStyle(.secondary)
                    Text(clip.audioStatus)
                }
                GridRow {
                    Text("Separation")
                        .foregroundStyle(.secondary)
                    Text(clip.separationStatus)
                }
                GridRow {
                    Text("Transcription")
                        .foregroundStyle(.secondary)
                    Text(clip.transcriptionStatus)
                }
                GridRow {
                    Text("Review")
                        .foregroundStyle(.secondary)
                    Text(clip.reviewStatus.isEmpty ? "needs_review" : clip.reviewStatus)
                }
                GridRow {
                    Text("Caption Source")
                        .foregroundStyle(.secondary)
                    Text(clip.captionSource.isEmpty ? "unknown" : clip.captionSource)
                }
                GridRow {
                    Text("Lyrics Source")
                        .foregroundStyle(.secondary)
                    Text(clip.lyricsSource.isEmpty ? "unknown" : clip.lyricsSource)
                }
                GridRow {
                    Text("Annotation")
                        .foregroundStyle(.secondary)
                    Text(clip.annotationBackend.isEmpty ? "manual" : clip.annotationBackend)
                }
                if !clip.annotationModelID.isEmpty {
                    GridRow {
                        Text("Model")
                            .foregroundStyle(.secondary)
                        Text(clip.annotationModelID)
                            .lineLimit(1)
                            .truncationMode(.middle)
                    }
                }
                if !clip.rawCaptionOutputFile.isEmpty {
                    GridRow {
                        Text("Raw Caption")
                            .foregroundStyle(.secondary)
                        Text(clip.rawCaptionOutputFile)
                            .lineLimit(1)
                            .truncationMode(.middle)
                    }
                }
                if !clip.rawLyricsOutputFile.isEmpty {
                    GridRow {
                        Text("Raw Lyrics")
                            .foregroundStyle(.secondary)
                        Text(clip.rawLyricsOutputFile)
                            .lineLimit(1)
                            .truncationMode(.middle)
                    }
                }
                ForEach(clip.assets.keys.sorted(), id: \.self) { key in
                    GridRow {
                        Text(key)
                            .foregroundStyle(.secondary)
                        Text(clip.assets[key] ?? "")
                            .lineLimit(1)
                            .truncationMode(.middle)
                    }
                }
            }
            .font(.caption)
        }
        .padding(10)
        .background(.quaternary.opacity(0.25))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }

    private var selectedClipIndex: Int? {
        guard let selectedClipID, let document else { return nil }
        return document.clips.firstIndex { $0.id == selectedClipID }
    }

    private var isBusy: Bool {
        enrichment.phase.isRunning || isSaving || isExporting
    }

    private func binding<T>(_ index: Int, _ keyPath: WritableKeyPath<DatasetClipReview, T>) -> Binding<T> {
        Binding(
            get: { document!.clips[index][keyPath: keyPath] },
            set: { value in
                document!.clips[index][keyPath: keyPath] = value
                document!.hasUnsavedChanges = true
            }
        )
    }

    private func captionBinding(_ index: Int) -> Binding<String> {
        Binding(
            get: { document!.clips[index].caption },
            set: { value in
                document!.clips[index].caption = value
                document!.clips[index].captionSource = value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? "none" : "manual"
                document!.clips[index].annotationBackend = "manual"
                document!.clips[index].reviewStatus = document!.clips[index].isApproved ? "approved" : "needs_review"
                document!.hasUnsavedChanges = true
            }
        )
    }

    private func lyricsBinding(_ index: Int) -> Binding<String> {
        Binding(
            get: { document!.clips[index].lyrics },
            set: { value in
                document!.clips[index].lyrics = value
                document!.clips[index].lyricsSource = value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? "none" : "manual"
                document!.clips[index].annotationBackend = "manual"
                document!.clips[index].reviewStatus = document!.clips[index].isApproved ? "approved" : "needs_review"
                document!.hasUnsavedChanges = true
            }
        )
    }

    private func approvedBinding(_ index: Int) -> Binding<Bool> {
        Binding(
            get: { document!.clips[index].isApproved },
            set: { value in
                document!.clips[index].isApproved = value
                document!.clips[index].reviewStatus = value ? "approved" : "needs_review"
                document!.hasUnsavedChanges = true
            }
        )
    }

    private func field(_ label: String, text: Binding<String>) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(label)
                .font(.caption)
                .foregroundStyle(.secondary)
            TextField(label, text: text)
                .textFieldStyle(.roundedBorder)
                .frame(minWidth: 140)
        }
    }

    private func metric(_ value: String, _ label: String) -> some View {
        VStack(alignment: .leading, spacing: 0) {
            Text(value)
                .font(.callout)
                .fontWeight(.semibold)
            Text(label)
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .frame(minWidth: 58, alignment: .leading)
    }

    private func badge(_ text: String, color: Color = .secondary) -> some View {
        Text(text)
            .font(.caption2)
            .foregroundStyle(color)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(color.opacity(0.12))
            .clipShape(Capsule())
    }

    private func sourceBadge(_ text: String) -> some View {
        let source = text.isEmpty ? "unknown" : text
        let color: Color = {
            switch source {
            case "manual", "dataset_review":
                return .blue
            case "sidecar", "csv":
                return .teal
            case "native-qwen3-asr", "ace-step-transcriber", "ace-step-captioner":
                return .purple
            case "none":
                return .secondary
            default:
                return .orange
            }
        }()
        return badge(source, color: color)
    }

    private func statusMessage(_ title: String, message: String?, systemImage: String, color: Color) -> some View {
        VStack(spacing: 8) {
            Image(systemName: systemImage)
                .font(.largeTitle)
                .foregroundStyle(color)
            Text(title)
                .font(.headline)
            if let message {
                Text(message)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
            }
        }
        .padding()
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private func loadDocument() {
        do {
            var loaded = try TrainingDatasetPackageEditor.loadPackage(at: datasetDirectory)
            if selectedClipID == nil || !loaded.clips.contains(where: { $0.id == selectedClipID }) {
                selectedClipID = loaded.clips.first?.id
            }
            loaded.hasUnsavedChanges = false
            document = loaded
            errorMessage = nil
        } catch {
            document = nil
            errorMessage = error.localizedDescription
        }
    }

    @discardableResult
    private func saveDocument() -> Bool {
        guard let current = document else { return false }
        isSaving = true
        defer { isSaving = false }
        do {
            try TrainingDatasetPackageEditor.save(current)
            notice = "Dataset saved with review backup."
            loadDocument()
            onPackageUpdated()
            return true
        } catch {
            notice = nil
            errorMessage = error.localizedDescription
            return false
        }
    }

    private func importSidecars() {
        guard var current = document else { return }
        let changed = TrainingDatasetPackageEditor.importSidecars(into: &current)
        document = current
        notice = changed == 0
            ? "No matching sidecar metadata was found."
            : "Imported \(changed) sidecar field\(changed == 1 ? "" : "s"). Review, then save."
    }

    private func markAllApproved() {
        guard document != nil else { return }
        for index in document!.clips.indices {
            document!.clips[index].isApproved = true
        }
        document!.hasUnsavedChanges = true
        notice = "All clips marked approved. Save to persist."
    }

    private func startEnrichment() {
        guard saveDocument() else { return }
        enrichment.enrich(datasetDirectory: datasetDirectory)
    }

    private func openInTraining() {
        guard let current = document else { return }
        do {
            if current.hasUnsavedChanges {
                try TrainingDatasetPackageEditor.save(current)
                loadDocument()
            }
            let files = try TrainingDatasetPackageEditor.importedFiles(for: current)
            onOpenInTraining(files)
        } catch {
            notice = nil
            errorMessage = error.localizedDescription
        }
    }

    private func exportPackage() {
        guard let current = document else { return }
        if current.hasUnsavedChanges, !saveDocument() {
            return
        }

        let panel = NSSavePanel()
        panel.allowedContentTypes = [.zip]
        panel.canCreateDirectories = true
        panel.nameFieldStringValue = "\(datasetDirectory.lastPathComponent).zip"
        guard panel.runModal() == .OK, let destination = panel.url else { return }

        isExporting = true
        defer { isExporting = false }

        do {
            let archiveURL = try TrainingDatasetPackageWriter.exportPackageArchive(
                outputDirectory: datasetDirectory,
                to: destination
            )
            notice = "Package exported to \(archiveURL.lastPathComponent)."
            onPackageUpdated()
        } catch {
            notice = nil
            errorMessage = error.localizedDescription
        }
    }
}
