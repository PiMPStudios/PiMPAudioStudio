import PiMPAudioCore
import SwiftUI

struct AddModelView: View {
    let registry: ModelRegistry
    let onModelAdded: () -> Void
    let onModelStatusChanged: (String, ModelStatus) -> Void

    @State private var repoID: String = ""
    @State private var selectedCategory: ModelCategory = .aceStepDiT
    @State private var isValidating: Bool = false
    @State private var isDownloading: Bool = false
    @State private var downloadProgress: Double = 0
    @State private var downloadingModelName: String?
    @State private var validationError: String?
    @State private var validatedSpec: ModelSpec?
    @State private var showConfirmation: Bool = false

    private var categoryOptions: [(ModelCategory, String)] {
        [
            (.aceStepDiT, "ACE-Step DiT (Music)"),
            (.aceStepLM, "ACE-Step LM (Audio Codes)"),
            (.stableAudio, "Stable Audio (SFX)"),
        ]
    }

    var body: some View {
        Section("Add Custom Model") {
            HStack {
                TextField("HuggingFace Repo ID (e.g., username/model-name)", text: $repoID)
                    .textFieldStyle(.roundedBorder)
                    .disabled(isValidating)

                Picker("Category", selection: $selectedCategory) {
                    ForEach(categoryOptions, id: \.0) { category, label in
                        Text(label).tag(category)
                    }
                }
                .frame(width: 200)
                .disabled(isValidating)

                Button(action: validate) {
                    if isValidating {
                        ProgressView()
                            .controlSize(.small)
                    } else {
                        Text("Validate & Add")
                    }
                }
                .disabled(repoID.trimmingCharacters(in: .whitespacesAndNewlines)
                    .isEmpty || isValidating || isDownloading)
            }

            if isDownloading {
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text("Downloading \(downloadingModelName ?? "model")")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        Spacer()
                        Text("\(Int(downloadProgress * 100))%")
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(.secondary)
                    }
                    ProgressView(value: downloadProgress)
                }
            }

            if let error = validationError {
                Label(error, systemImage: "exclamationmark.triangle.fill")
                    .foregroundStyle(.red)
                    .font(.caption)
            }
        }
        .alert("Add Model", isPresented: $showConfirmation) {
            Button("Add & Download") { confirmAdd() }
            Button("Cancel", role: .cancel) { validatedSpec = nil }
        } message: {
            if let spec = validatedSpec {
                Text("Add \(spec.displayName)?\nSize: \(spec.totalSizeFormatted)")
            }
        }
    }

    private func validate() {
        let trimmed = repoID.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }

        isValidating = true
        validationError = nil

        Task {
            do {
                // Check for duplicate
                let existingSpecs = await registry.allSpecs
                if existingSpecs.contains(where: { $0.huggingFaceRepo == trimmed }) {
                    validationError = "This model is already added."
                    isValidating = false
                    return
                }

                let modelFiles: CustomModelFiles
                do {
                    modelFiles = try await resolveFiles(repoID: trimmed, category: selectedCategory)
                } catch let error as ValidationFailure {
                    validationError = error.message
                    isValidating = false
                    return
                }

                let id = trimmed.replacingOccurrences(of: "/", with: "-").lowercased()
                let spec = ModelSpec(
                    id: id,
                    displayName: trimmed,
                    huggingFaceRepo: trimmed,
                    files: modelFiles.files,
                    category: selectedCategory,
                    variantFamily: modelFiles.variantFamily,
                    isRunnable: modelFiles.isRunnable,
                    runtimeNote: modelFiles.runtimeNote
                )

                validatedSpec = spec
                showConfirmation = true
                isValidating = false
            } catch {
                validationError = "Could not reach HuggingFace. Check your connection."
                isValidating = false
            }
        }
    }

    private func confirmAdd() {
        guard let spec = validatedSpec else { return }
        isDownloading = true
        downloadProgress = 0
        downloadingModelName = spec.displayName
        validationError = nil
        validatedSpec = nil

        Task {
            do {
                try await registry.addModel(spec)
                onModelAdded()
                onModelStatusChanged(spec.id, .downloading(progress: 0))

                try await registry.downloadModels([spec]) { _, progress in
                    Task { @MainActor in
                        if progress < downloadProgress {
                            return
                        }
                        downloadProgress = progress
                        onModelStatusChanged(spec.id, .downloading(progress: progress))
                    }
                }

                repoID = ""
                downloadProgress = 1
                isDownloading = false
                downloadingModelName = nil
                onModelStatusChanged(spec.id, .ready)
                onModelAdded()
            } catch {
                isDownloading = false
                downloadingModelName = nil
                onModelStatusChanged(spec.id, .failed(error.localizedDescription))
                validationError = error.localizedDescription
            }
        }
    }

    private struct CustomModelFiles {
        let files: [ModelFileSpec]
        let variantFamily: String?
        let isRunnable: Bool
        let runtimeNote: String?
    }

    private struct ValidationFailure: Error {
        let message: String
    }

    private func resolveFiles(repoID: String, category: ModelCategory) async throws -> CustomModelFiles {
        switch category {
        case .aceStepDiT:
            var files = try await resolveSafetensorsFiles(repoID: repoID)
            let configFile = try await requireFile(repoID: repoID, path: "config.json")
            files.append(configFile)
            if let silence = try await optionalFile(repoID: repoID, path: "silence_latent.pt") {
                files.append(silence)
            }

            let config = try await loadACEConfig(repoID: repoID)
            let runtimeNote = config.runtimeSupportNote
            return CustomModelFiles(
                files: files,
                variantFamily: config.variantFamily,
                isRunnable: runtimeNote == nil,
                runtimeNote: runtimeNote
            )
        default:
            var files: [ModelFileSpec] = []
            for file in expectedFiles(for: category) {
                try await files.append(requireFile(repoID: repoID, path: file))
            }
            return CustomModelFiles(files: files, variantFamily: nil, isRunnable: true, runtimeNote: nil)
        }
    }

    private func expectedFiles(for category: ModelCategory) -> [String] {
        switch category {
        case .aceStepDiT: ["model.safetensors", "config.json"]
        case .aceStepVAE: ["vae.safetensors", "config.json"]
        case .aceStepLM: ["model.safetensors", "tokenizer.json", "config.json"]
        case .stableAudio: ["model.safetensors", "config.json"]
        }
    }

    private func resolveSafetensorsFiles(repoID: String) async throws -> [ModelFileSpec] {
        if let single = try await optionalFile(repoID: repoID, path: "model.safetensors") {
            return [single]
        }

        guard let index = try await optionalFile(repoID: repoID, path: "model.safetensors.index.json") else {
            throw ValidationFailure(message: "Missing required file: model.safetensors or model.safetensors.index.json")
        }

        let indexURL = try huggingFaceURL(repoID: repoID, path: "model.safetensors.index.json")
        let (data, _) = try await URLSession.shared.data(from: indexURL)
        struct SafetensorsIndex: Decodable {
            let weight_map: [String: String]
        }
        let parsed = try JSONDecoder().decode(SafetensorsIndex.self, from: data)
        let shardNames = Array(Set(parsed.weight_map.values)).sorted()
        guard !shardNames.isEmpty else {
            throw ValidationFailure(message: "model.safetensors.index.json did not list any shard files.")
        }

        var result = [index]
        for shard in shardNames {
            try await result.append(requireFile(repoID: repoID, path: shard))
        }
        return result
    }

    private func loadACEConfig(repoID: String) async throws -> DiTConfig {
        let url = try huggingFaceURL(repoID: repoID, path: "config.json")
        let (data, response) = try await URLSession.shared.data(from: url)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            throw ValidationFailure(message: "Could not read config.json.")
        }

        do {
            return try JSONDecoder().decode(DiTConfig.self, from: data)
        } catch let error as ValidationFailure {
            throw error
        } catch {
            throw ValidationFailure(message: "config.json is not a readable ACE-Step DiT config.")
        }
    }

    private func optionalFile(repoID: String, path: String) async throws -> ModelFileSpec? {
        let response = try await head(repoID: repoID, path: path)
        switch response.statusCode {
        case 200:
            return ModelFileSpec(remotePath: path, sizeBytes: contentLength(response))
        case 404:
            return nil
        case 401, 403:
            throw ValidationFailure(
                message: "This repository is private or gated. Only public repositories are supported for now."
            )
        default:
            throw ValidationFailure(message: "Unexpected error (HTTP \(response.statusCode)) checking \(path).")
        }
    }

    private func requireFile(repoID: String, path: String) async throws -> ModelFileSpec {
        if let spec = try await optionalFile(repoID: repoID, path: path) {
            return spec
        }
        throw ValidationFailure(message: "Missing required file: \(path)")
    }

    private func head(repoID: String, path: String) async throws -> HTTPURLResponse {
        let url = try huggingFaceURL(repoID: repoID, path: path)
        var request = URLRequest(url: url)
        request.httpMethod = "HEAD"
        let (_, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw ValidationFailure(message: "Could not reach HuggingFace. Check your connection.")
        }
        return http
    }

    private func huggingFaceURL(repoID: String, path: String) throws -> URL {
        guard let url = URL(string: "https://huggingface.co/\(repoID)/resolve/main/\(path)") else {
            throw ValidationFailure(message: "Invalid repo ID format.")
        }
        return url
    }

    private func contentLength(_ response: HTTPURLResponse) -> Int64 {
        guard let value = response.value(forHTTPHeaderField: "Content-Length"),
              let size = Int64(value) else {
            return 0
        }
        return size
    }
}
