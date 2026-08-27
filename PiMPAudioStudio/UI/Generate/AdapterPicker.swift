import PiMPAudioCore
import SwiftUI

/// Dropdown picker for selecting a LoRA adapter in the Generate view.
struct AdapterPicker: View {
    @Binding var selectedAdapterID: String?
    let activeModelCategory: ModelCategory

    @EnvironmentObject private var appState: AppState
    @EnvironmentObject private var trainingService: TrainingService

    /// Incrementing this causes SwiftUI to re-evaluate `adapters` on the next render.
    @State private var adapterListVersion = 0
    @State private var activeModelID: String?
    @State private var activeModelSpec: ModelSpec?

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

    private var selectedAdapter: LoRAAdapter? {
        guard let selectedAdapterID else { return nil }
        return adapters.first { $0.id == selectedAdapterID }
    }

    private var selectedAdapterWarning: String? {
        guard let selectedAdapter,
              !selectedAdapter.isCompatible(with: activeModelID, activeSpec: activeModelSpec) else {
            return nil
        }
        let activeName = activeModelSpec?.displayName ?? activeModelID ?? "the active model"
        return "\(selectedAdapter.name) was trained for \(selectedAdapter.baseModelID). Active model: \(activeName). It will be skipped until you switch models or pick a matching adapter."
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
                        Text(
                            "\(adapter.kind.rawValue.uppercased()) \(adapterDimensionLabel(adapter)) \u{2022} \(adapter.sampleCount) tracks"
                        )
                        .foregroundStyle(.secondary)
                    }
                    .tag(Optional(adapter.id))
                }
            }
            .pickerStyle(.menu)
            .id(adapterListVersion)

            if let selectedAdapterWarning {
                Label(selectedAdapterWarning, systemImage: "exclamationmark.triangle.fill")
                    .font(.caption)
                    .foregroundStyle(.orange)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .onChange(of: activeModelCategory) { _, _ in
            adapterListVersion += 1
            Task { await refreshActiveModel() }
            normalizeSelection()
        }
        .onAppear {
            adapterListVersion += 1
            Task { await refreshActiveModel() }
            normalizeSelection()
        }
        .onReceive(NotificationCenter.default.publisher(for: .adapterRegistryDidChange)) { _ in
            adapterListVersion += 1
            normalizeSelection()
        }
        .onReceive(NotificationCenter.default.publisher(for: .modelRegistryDidChange)) { _ in
            Task { await refreshActiveModel() }
        }
        .onReceive(trainingService.$state) { newState in
            if case .complete = newState {
                adapterListVersion += 1
            }
            normalizeSelection()
        }
    }

    private func normalizeSelection() {
        if let selected = selectedAdapterID,
           !adapters.contains(where: { $0.id == selected }) {
            selectedAdapterID = nil
        }
    }

    @MainActor
    private func refreshActiveModel() async {
        let id = await appState.registry.activeModelID(for: activeModelCategory)
        let spec = await appState.registry.activeSpec(for: activeModelCategory)
        activeModelID = id
        activeModelSpec = spec
    }

    private func adapterDimensionLabel(_ adapter: LoRAAdapter) -> String {
        switch adapter.kind {
        case .lora:
            "rank \(adapter.config.rank)"
        case .lokr:
            "dim \(adapter.lokrConfig?.linearDim ?? adapter.config.rank)"
        }
    }
}
