import SwiftUI

struct YouTubeImportSheet: View {
    @EnvironmentObject private var outputManager: OutputDirectoryManager
    @Environment(\.dismiss) private var dismiss

    /// Called when user taps "Open in Training"; parent opens TrainingSheet with these clips.
    var onOpenInTraining: ([TrainingImportedFile]) -> Void

    @StateObject private var builder = YouTubeDatasetBuilder()

    @State private var urlText = ""
    @State private var styleTemplate = ""
    @State private var config = YouTubeDatasetConfig()
    @State private var editableClips: [(url: URL, caption: String)] = []
    @State private var ytDlpVersion: String?
    @State private var captionsWriteError: String?

    private var ytDlpFound: Bool { ytDlpVersion != nil }

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    ytDlpBanner
                    urlSection
                    configSection
                    switch builder.phase {
                    case .idle:
                        EmptyView()
                    case .failed(let msg):
                        errorView(msg)
                    case .fetchingPlaylist, .downloading, .chunking:
                        progressSection
                    case .complete:
                        resultsSection
                    }
                    if let captionsWriteError {
                        errorView(captionsWriteError)
                    }
                }
                .padding(16)
            }
            Divider()
            actionBar
        }
        .frame(minWidth: 540, minHeight: 500)
        .onAppear { ytDlpVersion = YouTubeDatasetBuilder.ytDlpVersion() }
        .onChange(of: builder.phase) { _, newPhase in
            if case .complete = newPhase { editableClips = builder.resultClips }
        }
    }

    // MARK: - Header

    private var header: some View {
        HStack(spacing: 10) {
            Image(systemName: "play.rectangle.fill")
                .font(.title2)
                .foregroundStyle(.red)
            VStack(alignment: .leading, spacing: 2) {
                Text("Train from YouTube Channel")
                    .font(.headline)
                Text("Download audio from a channel or playlist and build a training dataset.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Button("Close") { dismiss() }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .disabled(builder.phase.isRunning)
        }
        .padding(16)
    }

    // MARK: - yt-dlp Banner

    private var ytDlpBanner: some View {
        Group {
            if ytDlpFound, let version = ytDlpVersion {
                HStack(spacing: 8) {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundStyle(.green)
                    Text("yt-dlp \(version) found")
                        .font(.callout)
                        .foregroundStyle(.primary)
                    Spacer()
                }
                .padding(10)
                .background(.green.opacity(0.08))
                .clipShape(RoundedRectangle(cornerRadius: 8))
            } else {
                VStack(alignment: .leading, spacing: 6) {
                    HStack(spacing: 8) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundStyle(.orange)
                        Text("yt-dlp not found")
                            .font(.callout)
                            .fontWeight(.semibold)
                    }
                    Text("Install via Homebrew: **brew install yt-dlp**")
                        .font(.caption)
                    Text("If already installed, PiMPAudioStudio may need Full Disk Access in System Settings \u{2192} Privacy & Security.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .padding(10)
                .background(.orange.opacity(0.08))
                .clipShape(RoundedRectangle(cornerRadius: 8))
            }
        }
    }

    // MARK: - URL Section

    private var urlSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Channel or Playlist URL")
                .font(.subheadline)
                .fontWeight(.semibold)

            TextField("https://www.youtube.com/@channel", text: $urlText)
                .textFieldStyle(.roundedBorder)
                .disabled(builder.phase.isRunning)

            Text("Accepted: channel URL (@handle or /channel/), playlist URL (/playlist?list=), or any YouTube URL yt-dlp can parse as a playlist.")
                .font(.caption)
                .foregroundStyle(.secondary)

            VStack(alignment: .leading, spacing: 4) {
                Text("Style Description")
                    .font(.subheadline)
                    .fontWeight(.semibold)
                TextField("dark blues, slow tempo, minor key, atmospheric guitar", text: $styleTemplate)
                    .textFieldStyle(.roundedBorder)
                    .disabled(builder.phase.isRunning)
                Text("Applied to every clip as its training caption. Leave blank to use video titles.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }

    // MARK: - Config Section

    private var configSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Import Settings")
                .font(.subheadline)
                .fontWeight(.semibold)

            VStack(spacing: 6) {
                LabeledContent("Clip Length") {
                    HStack(spacing: 8) {
                        Slider(value: $config.clipLength, in: 15...60, step: 5)
                            .frame(maxWidth: 160)
                            .disabled(builder.phase.isRunning)
                        Text("\(Int(config.clipLength))s")
                            .monospacedDigit()
                            .frame(width: 34, alignment: .trailing)
                            .foregroundStyle(.secondary)
                    }
                }
                LabeledContent("Clips per Track") {
                    Stepper("\(config.clipsPerTrack)",
                            value: $config.clipsPerTrack,
                            in: 1...20)
                        .disabled(builder.phase.isRunning)
                        .frame(maxWidth: 130)
                }
                LabeledContent("Max Total Clips") {
                    Stepper("\(config.maxTotalClips)",
                            value: $config.maxTotalClips,
                            in: 5...200,
                            step: 5)
                        .disabled(builder.phase.isRunning)
                        .frame(maxWidth: 130)
                }
            }
            .padding(10)
            .background(.quaternary.opacity(0.3))
            .clipShape(RoundedRectangle(cornerRadius: 8))

            if let dir = outputManager.directory {
                HStack(spacing: 4) {
                    Image(systemName: "folder")
                        .foregroundStyle(.secondary)
                    Text("Output: \(dir.path)/TrainingDatasets/…")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
            } else {
                HStack(spacing: 4) {
                    Image(systemName: "exclamationmark.triangle")
                        .foregroundStyle(.orange)
                    Text("No output directory set. Configure one in Settings before starting.")
                        .font(.caption)
                        .foregroundStyle(.orange)
                }
            }
        }
    }

    // MARK: - Progress Section

    private var progressSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Importing")
                .font(.subheadline)
                .fontWeight(.semibold)

            progressCard
        }
    }

    private var progressCard: some View {
        HStack(spacing: 12) {
            progressIcon
            VStack(alignment: .leading, spacing: 4) {
                Text(progressLabel)
                    .font(.callout)
                    .fontWeight(.medium)
                Text(progressDetail)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                progressIndicator
            }
        }
        .padding(12)
        .background(.quaternary.opacity(0.3))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }

    private var progressIcon: some View {
        Group {
            switch builder.phase {
            case .fetchingPlaylist:
                Image(systemName: "list.bullet.rectangle")
                    .font(.title2)
                    .foregroundStyle(.blue)
            case .downloading:
                Image(systemName: "arrow.down.circle")
                    .font(.title2)
                    .foregroundStyle(.blue)
            case .chunking:
                Image(systemName: "scissors")
                    .font(.title2)
                    .foregroundStyle(.orange)
            default:
                EmptyView()
            }
        }
    }

    private var progressLabel: String {
        switch builder.phase {
        case .fetchingPlaylist: return "Fetching playlist…"
        case .downloading(let idx, let total, _): return "Downloading \(idx) of \(total)"
        case .chunking(let idx, let total): return "Chunking audio \(idx) of \(total)"
        default: return ""
        }
    }

    private var progressDetail: String {
        switch builder.phase {
        case .fetchingPlaylist: return "Querying video list with yt-dlp"
        case .downloading(_, _, let title): return title
        case .chunking(let idx, _): return "Slicing video \(idx) into \(Int(config.clipLength))s clips"
        default: return ""
        }
    }

    @ViewBuilder
    private var progressIndicator: some View {
        switch builder.phase {
        case .fetchingPlaylist:
            ProgressView()
                .progressViewStyle(.linear)
                .frame(maxWidth: .infinity)
        case .downloading(let idx, let total, _):
            ProgressView(value: Double(idx - 1), total: Double(total))
                .frame(maxWidth: .infinity)
        case .chunking(let idx, let total):
            ProgressView(value: Double(idx - 1), total: Double(total))
                .frame(maxWidth: .infinity)
        default:
            EmptyView()
        }
    }

    // MARK: - Results Section

    private var resultsSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Summary row
            HStack(spacing: 8) {
                Image(systemName: "checkmark.circle.fill")
                    .foregroundStyle(.green)
                Text("\(editableClips.count) clips ready for training")
                    .font(.callout)
                    .fontWeight(.medium)
                Spacer()
                if case .complete(let dir, _) = builder.phase {
                    Button {
                        NSWorkspace.shared.open(dir)
                    } label: {
                        Label("Show in Finder", systemImage: "folder")
                            .font(.caption)
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                }
            }
            .padding(10)
            .background(.green.opacity(0.08))
            .clipShape(RoundedRectangle(cornerRadius: 8))

            // Caption hint
            Text("Review and edit captions below. These become the training descriptions for each clip.")
                .font(.caption)
                .foregroundStyle(.secondary)

            // Editable clip list
            VStack(spacing: 4) {
                ForEach(editableClips.indices, id: \.self) { idx in
                    HStack(spacing: 8) {
                        Image(systemName: "waveform")
                            .foregroundStyle(.secondary)
                            .frame(width: 16)
                        TextField("Caption", text: Binding(
                            get: { editableClips[idx].caption },
                            set: { editableClips[idx].caption = $0 }
                        ))
                        .textFieldStyle(.roundedBorder)
                        .font(.callout)
                    }
                    .padding(.vertical, 2)
                }
            }
            .padding(10)
            .background(.quaternary.opacity(0.3))
            .clipShape(RoundedRectangle(cornerRadius: 8))
        }
    }

    // MARK: - Error View

    private func errorView(_ message: String) -> some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: "xmark.circle.fill")
                .foregroundStyle(.red)
                .padding(.top, 1)
            VStack(alignment: .leading, spacing: 4) {
                Text("Import failed")
                    .font(.callout)
                    .fontWeight(.semibold)
                Text(message)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
        }
        .padding(12)
        .background(.red.opacity(0.08))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }

    // MARK: - Action Bar

    private var actionBar: some View {
        HStack {
            Button("Cancel") { dismiss() }
                .buttonStyle(.bordered)
                .disabled(builder.phase.isRunning)

            Spacer()

            switch builder.phase {
            case .idle, .failed:
                Button("Start") { startImport() }
                    .buttonStyle(.borderedProminent)
                    .disabled(!canStart)
            case .fetchingPlaylist, .downloading, .chunking:
                Button("Stop", role: .destructive) { builder.cancel() }
                    .buttonStyle(.bordered)
            case .complete:
                Button("Open in Training") {
                    captionsWriteError = nil
                    if case .complete(let dir, _) = builder.phase {
                        do {
                            try YouTubeDatasetBuilder.writeCaptionsFile(clips: editableClips, to: dir)
                        } catch {
                            captionsWriteError =
                                "Could not write captions file: \(error.localizedDescription)"
                            return
                        }
                    }
                    let files = editableClips.map {
                        TrainingImportedFile(url: $0.url, description: $0.caption)
                    }
                    onOpenInTraining(files)
                }
                .buttonStyle(.borderedProminent)
            }
        }
        .padding(16)
    }

    // MARK: - Helpers

    private var canStart: Bool {
        let trimmed = urlText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let url = URL(string: trimmed), YouTubeURLPolicy.isAllowed(url) else { return false }
        return outputManager.directory != nil && ytDlpFound
    }

    private func startImport() {
        let trimmed = urlText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let url = URL(string: trimmed), let outputBase = outputManager.directory else { return }
        guard YouTubeURLPolicy.isAllowed(url) else {
            builder.phase = .failed(YouTubeDatasetError.unsupportedURL.localizedDescription ?? "Unsupported URL")
            return
        }
        let template = styleTemplate.trimmingCharacters(in: .whitespacesAndNewlines)
        builder.start(channelURL: url, config: config, outputBase: outputBase, styleTemplate: template)
    }
}
