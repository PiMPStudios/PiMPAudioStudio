import SwiftUI
import AnvilCore

/// Dropdown picker for selecting a LoRA adapter in the Generate view.
struct AdapterPicker: View {
    @Binding var selectedAdapterID: String?
    let activeModelCategory: ModelCategory

    @State private var adapters: [LoRAAdapter] = []

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("LoRA Adapter")
                .font(.caption)
                .foregroundStyle(.secondary)

            Picker("LoRA Adapter", selection: $selectedAdapterID) {
                Text("None").tag(String?.none)
                ForEach(adapters) { adapter in
                    HStack {
                        Text(adapter.name)
                        Text("rank \(adapter.config.rank) \u{2022} \(adapter.sampleCount) tracks")
                            .foregroundStyle(.secondary)
                    }
                    .tag(Optional(adapter.id))
                }
            }
            .pickerStyle(.menu)
        }
        .onAppear { refreshAdapters() }
        .onChange(of: activeModelCategory) { _, _ in
            refreshAdapters()
            if let selected = selectedAdapterID,
               !adapters.contains(where: { $0.id == selected }) {
                selectedAdapterID = nil
            }
        }
    }

    private func refreshAdapters() {
        adapters = AdapterRegistry.compatibleAdapters(for: activeModelCategory)
    }
}
