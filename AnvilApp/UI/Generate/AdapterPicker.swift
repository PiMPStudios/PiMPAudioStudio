import SwiftUI
import AnvilCore

/// Dropdown picker for selecting a LoRA adapter in the Generate view.
struct AdapterPicker: View {
    @Binding var selectedAdapterID: String?
    let activeModelCategory: ModelCategory

    @EnvironmentObject private var trainingService: TrainingService

    /// Incrementing this causes SwiftUI to re-evaluate `adapters` on the next render.
    @State private var adapterListVersion = 0

    /// Always reads live from the registry — no onAppear timing gap that would
    /// produce "selection is invalid" warnings when the view first renders with a
    /// previously-saved adapter ID.
    private var adapters: [LoRAAdapter] {
        _ = adapterListVersion
        return AdapterRegistry.compatibleAdapters(for: activeModelCategory)
    }

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
        .onChange(of: activeModelCategory) { _, _ in
            adapterListVersion += 1
            if let selected = selectedAdapterID,
               !adapters.contains(where: { $0.id == selected }) {
                selectedAdapterID = nil
            }
        }
        .onReceive(trainingService.$state) { newState in
            if case .complete = newState { adapterListVersion += 1 }
        }
    }
}
