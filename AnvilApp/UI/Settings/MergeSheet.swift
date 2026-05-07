import SwiftUI
import AnvilCore

struct MergeSheet: View {
    @Environment(\.dismiss) private var dismiss

    let onComplete: () -> Void

    @State private var adapters: [LoRAAdapter] = []
    @State private var adapterA: LoRAAdapter?
    @State private var adapterB: LoRAAdapter?
    @State private var alpha: Float = 0.5
    @State private var newName: String = ""
    @State private var isMerging = false
    @State private var errorMessage: String?

    var body: some View {
        VStack(spacing: 16) {
            Text("Merge Two Adapters")
                .font(.headline)

            Text("Creates a new adapter by blending weights: \u{03B1} \u{00D7} A + (1-\u{03B1}) \u{00D7} B")
                .font(.caption)
                .foregroundStyle(.secondary)

            // Adapter A picker
            LabeledContent("Adapter A") {
                Picker("Adapter A", selection: $adapterA) {
                    Text("Select...").tag(nil as LoRAAdapter?)
                    ForEach(adapters) { adapter in
                        adapterLabel(adapter).tag(adapter as LoRAAdapter?)
                    }
                }
                .frame(maxWidth: 220)
            }

            // Adapter B picker (filtered to same base model + rank)
            LabeledContent("Adapter B") {
                Picker("Adapter B", selection: $adapterB) {
                    Text("Select...").tag(nil as LoRAAdapter?)
                    ForEach(compatibleWithA) { adapter in
                        adapterLabel(adapter).tag(adapter as LoRAAdapter?)
                    }
                }
                .frame(maxWidth: 220)
                .disabled(adapterA == nil)
            }

            // Alpha slider
            VStack(alignment: .leading, spacing: 4) {
                LabeledContent("Blend (\u{03B1} = \(String(format: "%.2f", alpha)))") {
                    Slider(
                        value: Binding(
                            get: { Double(alpha) },
                            set: { alpha = Float($0) }
                        ),
                        in: 0...1,
                        step: 0.05
                    )
                    .frame(maxWidth: 200)
                }
                HStack {
                    Text("\u{2190} More B")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Text("More A \u{2192}")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: 200)
                .padding(.leading, 120)
            }

            // Name
            LabeledContent("New Name") {
                TextField("e.g. Jazz+Blues Blend", text: $newName)
                    .textFieldStyle(.roundedBorder)
                    .frame(maxWidth: 200)
            }

            if let error = errorMessage {
                Label(error, systemImage: "exclamationmark.triangle")
                    .foregroundStyle(.red)
                    .font(.caption)
            }

            // Buttons
            HStack {
                Button("Cancel") { dismiss() }
                    .buttonStyle(.bordered)
                Spacer()
                Button("Merge") { performMerge() }
                    .buttonStyle(.borderedProminent)
                    .disabled(!canMerge)
            }
        }
        .padding()
        .onAppear { adapters = AdapterRegistry.listAdapters() }
        .overlay {
            if isMerging {
                Color.black.opacity(0.3)
                    .overlay {
                        VStack {
                            ProgressView()
                            Text("Merging...")
                                .font(.caption)
                        }
                    }
            }
        }
    }

    // MARK: - Computed

    private var compatibleWithA: [LoRAAdapter] {
        guard let a = adapterA else { return [] }
        return adapters.filter { adapter in
            adapter.id != a.id
                && adapter.baseModelID == a.baseModelID
                && adapter.config.rank == a.config.rank
        }
    }

    private var canMerge: Bool {
        adapterA != nil
            && adapterB != nil
            && !newName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && !isMerging
    }

    // MARK: - Actions

    private func performMerge() {
        guard let a = adapterA, let b = adapterB else { return }
        isMerging = true
        errorMessage = nil

        Task {
            do {
                try AdapterRegistry.merge(
                    adapterA: a.id,
                    adapterB: b.id,
                    alpha: alpha,
                    name: newName
                )
                await MainActor.run {
                    isMerging = false
                    onComplete()
                    dismiss()
                }
            } catch {
                await MainActor.run {
                    isMerging = false
                    errorMessage = error.localizedDescription
                }
            }
        }
    }

    private func adapterLabel(_ adapter: LoRAAdapter) -> some View {
        HStack {
            Text(adapter.name)
            Text("(r\(adapter.config.rank))")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }
}
