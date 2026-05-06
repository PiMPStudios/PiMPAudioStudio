import SwiftUI

struct BackgroundDownloadIndicator: View {
    let phase: BackgroundDownloadPhase
    var onRetry: (() -> Void)?

    var body: some View {
        switch phase {
        case .idle:
            EmptyView()

        case .downloading(let modelName, let progress):
            HStack(spacing: 8) {
                ProgressView()
                    .controlSize(.small)
                Text("Downloading \(modelName)... \(Int(progress * 100))%")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .background(.regularMaterial, in: Capsule())

        case .complete:
            HStack(spacing: 6) {
                Image(systemName: "checkmark.circle.fill")
                    .foregroundStyle(.green)
                Text("All models ready")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .background(.regularMaterial, in: Capsule())

        case .failed(let message):
            Button(action: { onRetry?() }) {
                HStack(spacing: 6) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundStyle(.yellow)
                    Text(message)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 6)
                .background(.regularMaterial, in: Capsule())
            }
            .buttonStyle(.plain)
        }
    }
}
