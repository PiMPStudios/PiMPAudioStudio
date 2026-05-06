import SwiftUI
import AnvilCore

struct GenerateView: View {
    @EnvironmentObject private var generationService: GenerationService
    @EnvironmentObject private var playerService: AudioPlayerService

    @State private var promptText: String = ""
    @State private var lyricsText: String = ""
    @State private var duration: Double = 30
    @State private var seed: String = ""
    @State private var guidanceScale: Double = 1.0
    @State private var method: InferMethod = .ode
    @State private var precision: Precision = .float16

    var body: some View {
        ScrollView {
            VStack(spacing: 16) {
                promptSection
                lyricsSection
                controlsBar
                progressSection
                playerSection
            }
            .padding()
        }
    }

    private var promptSection: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Prompt")
                .font(.caption)
                .foregroundStyle(.secondary)
            TextEditor(text: $promptText)
                .font(.body)
                .frame(minHeight: 60, maxHeight: 100)
                .overlay(RoundedRectangle(cornerRadius: 8).stroke(.quaternary))
        }
    }

    private var lyricsSection: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Lyrics")
                .font(.caption)
                .foregroundStyle(.secondary)
            TextEditor(text: $lyricsText)
                .font(.body)
                .frame(minHeight: 80, maxHeight: 120)
                .overlay(RoundedRectangle(cornerRadius: 8).stroke(.quaternary))
        }
    }

    private var controlsBar: some View {
        HStack {
            Button("Generate") {
                let request = GenerationRequest(
                    text: promptText,
                    lyrics: lyricsText,
                    duration: duration,
                    seed: Int(seed),
                    method: method,
                    guidanceScale: Float(guidanceScale),
                    precision: precision
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

    @ViewBuilder
    private var progressSection: some View {
        switch generationService.state {
        case .idle, .complete, .cancelled:
            EmptyView()
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
