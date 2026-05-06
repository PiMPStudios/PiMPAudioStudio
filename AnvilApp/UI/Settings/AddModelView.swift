import SwiftUI
import AnvilCore

struct AddModelView: View {
    let registry: ModelRegistry
    let onModelAdded: () -> Void

    @State private var repoID: String = ""
    @State private var selectedCategory: ModelCategory = .aceStepDiT
    @State private var isValidating: Bool = false
    @State private var validationError: String?
    @State private var validatedSpec: ModelSpec?
    @State private var showConfirmation: Bool = false

    private var categoryOptions: [(ModelCategory, String)] {
        [
            (.aceStepDiT, "ACE-Step DiT (Music)"),
            (.stableAudio, "Stable Audio (SFX)")
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
                .disabled(repoID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || isValidating)
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

                let expectedFiles = expectedFiles(for: selectedCategory)
                var totalSize: Int64 = 0

                for file in expectedFiles {
                    let urlString = "https://huggingface.co/\(trimmed)/resolve/main/\(file)"
                    guard let url = URL(string: urlString) else {
                        validationError = "Invalid repo ID format."
                        isValidating = false
                        return
                    }

                    var request = URLRequest(url: url)
                    request.httpMethod = "HEAD"

                    let (_, response) = try await URLSession.shared.data(for: request)
                    guard let http = response as? HTTPURLResponse else {
                        validationError = "Could not reach HuggingFace. Check your connection."
                        isValidating = false
                        return
                    }

                    switch http.statusCode {
                    case 200:
                        if let contentLength = http.value(forHTTPHeaderField: "Content-Length"),
                           let size = Int64(contentLength) {
                            totalSize += size
                        }
                    case 401, 403:
                        validationError = "This repository is private or gated. Only public repositories are supported for now."
                        isValidating = false
                        return
                    case 404:
                        validationError = "Missing required file: \(file)"
                        isValidating = false
                        return
                    default:
                        validationError = "Unexpected error (HTTP \(http.statusCode)) checking \(file)."
                        isValidating = false
                        return
                    }
                }

                let id = trimmed.replacingOccurrences(of: "/", with: "-").lowercased()
                let spec = ModelSpec(
                    id: id,
                    huggingFaceRepo: trimmed,
                    files: expectedFiles,
                    sizeBytes: totalSize > 0 ? totalSize : 1_000_000_000,
                    category: selectedCategory
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
        Task {
            do {
                try await registry.addModel(spec)
                // Start download
                try? await registry.downloadModels([spec]) { _, _ in }
                repoID = ""
                validatedSpec = nil
                onModelAdded()
            } catch {
                validationError = error.localizedDescription
            }
        }
    }

    private func expectedFiles(for category: ModelCategory) -> [String] {
        switch category {
        case .aceStepDiT: return ["model.safetensors", "config.json"]
        case .aceStepVAE: return ["vae.safetensors", "config.json"]
        case .stableAudio: return ["model.safetensors", "config.json"]
        case .lyricLLM: return ["model.safetensors", "tokenizer.json", "config.json"]
        }
    }

}
