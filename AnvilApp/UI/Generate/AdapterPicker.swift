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

    private var normalizedSelection: Binding<String?> {
        Binding(
            get: {
                guard let selectedAdapterID else { return nil }
                return adapters.contains(where: { $0.id == selectedAdapterID }) ? selectedAdapterID : nil
            },
            set: { selectedAdapterID = $0 }
        )
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Adapter")
                .font(.caption)
                .foregroundStyle(.secondary)

            Picker("Adapter", selection: normalizedSelection) {
                Text("None").tag(String?.none)
                ForEach(adapters) { adapter in
                    HStack {
                        Text(adapter.name)
                        Text("\(adapter.kind.rawValue.uppercased()) \(adapterDimensionLabel(adapter)) \u{2022} \(adapter.sampleCount) tracks")
                            .foregroundStyle(.secondary)
                    }
                    .tag(Optional(adapter.id))
                }
            }
            .pickerStyle(.menu)
            .id(adapterListVersion)
        }
        .onChange(of: activeModelCategory) { _, _ in
            adapterListVersion += 1
            normalizeSelection()
        }
        .onAppear {
            adapterListVersion += 1
            normalizeSelection()
        }
        .onReceive(NotificationCenter.default.publisher(for: .adapterRegistryDidChange)) { _ in
            adapterListVersion += 1
            normalizeSelection()
        }
        .onReceive(trainingService.$state) { newState in
            if case .complete = newState { adapterListVersion += 1 }
            normalizeSelection()
        }
    }

    private func normalizeSelection() {
        if let selected = selectedAdapterID,
           !adapters.contains(where: { $0.id == selected }) {
            selectedAdapterID = nil
        }
    }

    private func adapterDimensionLabel(_ adapter: LoRAAdapter) -> String {
        switch adapter.kind {
        case .lora:
            return "rank \(adapter.config.rank)"
        case .lokr:
            return "dim \(adapter.lokrConfig?.linearDim ?? adapter.config.rank)"
        }
    }
}
