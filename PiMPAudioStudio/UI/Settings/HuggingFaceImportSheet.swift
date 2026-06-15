import SwiftUI
import AppKit
import PiMPAudioCore

struct HuggingFaceImportSheet: View {
    @EnvironmentObject private var service: CommunityService
    @Environment(\.dismiss) private var dismiss

    @State private var repoURLText = ""
    @State private var fetchedAdapter: LoRAAdapter?
    @State private var isFetching = false
    @State private var fetchError: String?
    @State private var downloadComplete = false

    // The currently active installed base model category (nil if unknown).
    // In a real app this would come from AppState; we default to nil for
    // compatibility checking so the warning only shows on a real mismatch.
    var installedCategory: ModelCategory? = nil

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            header
            Divider()
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    urlSection
                    Divider()
                    localSection
                    if let error = fetchError {
                        errorView(error)
                    }
                    if let adapter = fetchedAdapter {
                        previewSection(adapter)
                    }
                }
                .padding(16)
            }
            Divider()
            actionBar
        }
    }

    // MARK: - Header

    private var header: some View {
        HStack {
            Image(systemName: "arrow.down.circle")
                .font(.title2)
            VStack(alignment: .leading, spacing: 2) {
                Text("Import Adapter")
                    .font(.headline)
                Text("Install a PiMP Audio Studio or PEFT LoRA adapter from HuggingFace or a local folder.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
        }
        .padding(16)
    }

    // MARK: - URL Section

    private var urlSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Repository URL")
                .font(.subheadline)
                .fontWeight(.medium)
            HStack(spacing: 8) {
                TextField(
                    "https://huggingface.co/username/repo-name",
                    text: $repoURLText
                )
                .textFieldStyle(.roundedBorder)
                .onSubmit { fetchMetadata() }
                Button("Fetch") {
                    fetchMetadata()
                }
                .buttonStyle(.borderedProminent)
                .disabled(isFetching || repoURLText.trimmingCharacters(in: .whitespaces).isEmpty)
                if isFetching {
                    ProgressView().controlSize(.small)
                }
            }
            Text("Accepts PEFT repos (adapter_config.json + adapter_model.safetensors) and native PiMP Audio Studio repos (adapter.json + adapter.safetensors).")
                .font(.caption2)
                .foregroundStyle(.tertiary)
        }
    }

    private var localSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Local Adapter Folder")
                .font(.subheadline)
                .fontWeight(.medium)
            HStack(spacing: 8) {
                Button {
                    importLocalFolder()
                } label: {
                    Label("Choose Folder", systemImage: "folder")
                }
                .buttonStyle(.bordered)
                .disabled(isFetching)

                Text("Select a native, PEFT, raw LoKR, or Side-Step LoRA/LoKR run folder. Side-Step imports prefer best/ before final/.")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
        }
    }

    // MARK: - Preview Section

    private func previewSection(_ adapter: LoRAAdapter) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Adapter Preview")
                .font(.subheadline)
                .fontWeight(.medium)

            VStack(alignment: .leading, spacing: 6) {
                LabeledRow(label: "Name", value: adapter.name)
                LabeledRow(label: "Category", value: adapter.baseModelCategory.rawValue)
                LabeledRow(label: "Rank", value: "r\(adapter.config.rank)")
                LabeledRow(label: "Base Model", value: adapter.baseModelID)
                LabeledRow(label: "Created",
                           value: adapter.createdAt.formatted(date: .abbreviated, time: .omitted))
            }
            .padding(10)
            .background(.quaternary.opacity(0.4))
            .clipShape(RoundedRectangle(cornerRadius: 8))

            // Compatibility warning when category doesn't match installed model
            if let installed = installedCategory,
               adapter.baseModelCategory != installed {
                HStack(alignment: .top, spacing: 8) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundStyle(.yellow)
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Compatibility Warning")
                            .font(.caption)
                            .fontWeight(.semibold)
                        Text(
                            "This adapter targets \u{201c}\(adapter.baseModelCategory.rawValue)\u{201d} " +
                            "but your installed model is \u{201c}\(installed.rawValue)\u{201d}. " +
                            "You can still install it, but it may not work correctly."
                        )
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    }
                }
                .padding(10)
                .background(.yellow.opacity(0.08))
                .clipShape(RoundedRectangle(cornerRadius: 8))
            }
        }
    }

    // MARK: - Error View

    private func errorView(_ message: String) -> some View {
        HStack(spacing: 8) {
            Image(systemName: "exclamationmark.circle.fill")
                .foregroundStyle(.red)
            Text(message)
                .font(.caption)
                .foregroundStyle(.red)
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.red.opacity(0.06))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }

    // MARK: - Action Bar

    private var actionBar: some View {
        HStack {
            Button("Cancel") { dismiss() }
                .buttonStyle(.bordered)

            Spacer()

            if downloadComplete {
                Label("Installed", systemImage: "checkmark.circle.fill")
                    .foregroundStyle(.green)
                    .font(.subheadline)
                Button("Done") { dismiss() }
                    .buttonStyle(.borderedProminent)
            } else {
                if isFetching {
                    ProgressView().controlSize(.small)
                }
                Button("Import") {
                    fetchMetadata()
                }
                .buttonStyle(.borderedProminent)
                .disabled(repoURLText.trimmingCharacters(in: .whitespaces).isEmpty || isFetching)
            }
        }
        .padding(16)
    }

    // MARK: - Actions

    private func fetchMetadata() {
        let raw = repoURLText.trimmingCharacters(in: .whitespaces)
        guard !raw.isEmpty, let url = URL(string: raw) else {
            fetchError = "Enter a valid URL."
            return
        }
        fetchError = nil
        fetchedAdapter = nil
        downloadComplete = false
        isFetching = true

        Task {
            do {
                let adapter = try await service.importFromHuggingFace(repoURL: url)
                fetchedAdapter = adapter
                downloadComplete = true   // importFromHuggingFace also downloads
            } catch {
                fetchError = error.localizedDescription
            }
            isFetching = false
        }
    }

    private func importLocalFolder() {
        let panel = NSOpenPanel()
        panel.title = "Import Adapter Folder"
        panel.prompt = "Import"
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false

        guard panel.runModal() == .OK, let url = panel.url else { return }

        fetchError = nil
        fetchedAdapter = nil
        downloadComplete = false
        isFetching = true

        Task {
            do {
                let adapter = try await service.importLocalAdapter(directoryURL: url)
                fetchedAdapter = adapter
                downloadComplete = true
            } catch {
                fetchError = error.localizedDescription
            }
            isFetching = false
        }
    }

}

// MARK: - Helpers

private struct LabeledRow: View {
    let label: String
    let value: String

    var body: some View {
        HStack {
            Text(label)
                .font(.caption)
                .foregroundStyle(.secondary)
                .frame(width: 80, alignment: .leading)
            Text(value)
                .font(.caption)
        }
    }
}
