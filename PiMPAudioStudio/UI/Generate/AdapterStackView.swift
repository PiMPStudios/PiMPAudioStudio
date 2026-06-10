import SwiftUI
import PiMPAudioCore

/// Stacking controls for additional LoRA adapters in the generation view.
/// Placed below the primary AdapterPicker.
struct AdapterStackView: View {
    @ObservedObject var stackManager: AdapterStackManager
    let activeModelCategory: ModelCategory

    @EnvironmentObject private var appState: AppState
    @State private var showingAdapterPicker = false
    @State private var adapterListVersion = 0
    @State private var activeModelID: String?
    @State private var activeModelSpec: ModelSpec?

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            // Section header
            HStack {
                Text("Additional Adapters")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
                if !stackManager.additionalAdapters.isEmpty {
                    Text("Total: \(String(format: "%.2fx", stackManager.totalStrength))")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }
            }

            // Stacked adapter rows
            ForEach(Array(stackManager.additionalAdapters.enumerated()), id: \.element.adapter.id) { idx, entry in
                stackedAdapterRow(entry.adapter, strength: entry.strength)
            }

            // Add button
            Button {
                showingAdapterPicker = true
            } label: {
                Label("Add adapter", systemImage: "plus")
                    .font(.caption)
            }
            .disabled(stackManager.isAtCapacity)
            .help(stackManager.isAtCapacity ? "Maximum \(AdapterStackManager.maxStackSize) adapters active" : "Add another adapter to the stack")
        }
        .sheet(isPresented: $showingAdapterPicker) {
            adapterPickerSheet
        }
        .onAppear {
            Task { await refreshActiveModel() }
        }
        .onReceive(NotificationCenter.default.publisher(for: .adapterRegistryDidChange)) { _ in
            adapterListVersion += 1
        }
        .onReceive(NotificationCenter.default.publisher(for: .modelRegistryDidChange)) { _ in
            Task { await refreshActiveModel() }
        }
    }

    private func stackedAdapterRow(_ adapter: LoRAAdapter, strength: Float) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 8) {
                // Name + source badge
                VStack(alignment: .leading, spacing: 2) {
                    Text(adapter.name)
                        .font(.caption)
                        .fontWeight(.medium)
                        .lineLimit(1)
                    if adapter.source != .local {
                        Text(adapter.source.rawValue.capitalized)
                            .font(.caption2)
                            .foregroundStyle(.blue)
                    }
                }
                .frame(width: 100, alignment: .leading)

                Slider(
                    value: Binding(
                        get: { strength },
                        set: { stackManager.setStrength(adapterID: adapter.id, strength: $0) }
                    ),
                    in: AdapterStrengthLimits.minimum...AdapterStrengthLimits.maximum,
                    step: 0.10
                )
                .help("1.0x is normal strength. Higher values boost weak adapters at generation time.")

                Text(String(format: "%.2fx", strength))
                    .font(.caption)
                    .monospacedDigit()
                    .frame(width: 48)

                // Remove button
                Button {
                    stackManager.removeFromStack(adapterID: adapter.id)
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
            }

            if let warning = compatibilityWarning(for: adapter) {
                Label(warning, systemImage: "exclamationmark.triangle.fill")
                    .font(.caption2)
                    .foregroundStyle(.orange)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private var adapterPickerSheet: some View {
        VStack(spacing: 12) {
            Text("Select Adapter to Stack")
                .font(.headline)

            let available = availableAdapters
            if available.isEmpty {
                Text("No compatible adapters available")
                    .foregroundStyle(.secondary)
                    .padding()
            } else {
                List(available) { adapter in
                    Button {
                        stackManager.addToStack(adapter)
                        showingAdapterPicker = false
                    } label: {
                        HStack {
                            VStack(alignment: .leading) {
                                Text(adapter.name)
                                    .font(.body)
                                HStack(spacing: 4) {
                                    Text(adapter.kind == .lokr
                                        ? "LoKR dim \(adapter.lokrConfig?.linearDim ?? adapter.config.rank)"
                                        : "LoRA rank \(adapter.config.rank)")
                                    if adapter.source != .local {
                                        Text("(\(adapter.source.rawValue))")
                                    }
                                }
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            }
                            Spacer()
                        }
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                }
            }

            Button("Cancel") { showingAdapterPicker = false }
                .keyboardShortcut(.cancelAction)
        }
        .padding()
        .frame(minWidth: 300, minHeight: 200)
    }

    /// Adapters compatible with current model that aren't already in the stack.
    private var availableAdapters: [LoRAAdapter] {
        _ = adapterListVersion
        let allCompatible = AdapterRegistry.compatibleAdapters(for: activeModelCategory)
        let stackedIDs = Set(stackManager.allStackedAdapters.map(\.adapter.id))
        return allCompatible.filter { adapter in
            guard !stackedIDs.contains(adapter.id) else { return false }
            guard adapter.isCompatible(with: activeModelID, activeSpec: activeModelSpec) else {
                return false
            }
            if let primary = stackManager.primaryAdapter {
                return adapter.kind == primary.kind
            }
            return true
        }
    }

    @MainActor
    private func refreshActiveModel() async {
        let id = await appState.registry.activeModelID(for: activeModelCategory)
        let spec = await appState.registry.activeSpec(for: activeModelCategory)
        activeModelID = id
        activeModelSpec = spec
        adapterListVersion += 1
    }

    private func compatibilityWarning(for adapter: LoRAAdapter) -> String? {
        guard !adapter.isCompatible(with: activeModelID, activeSpec: activeModelSpec) else {
            return nil
        }
        let activeName = activeModelSpec?.displayName ?? activeModelID ?? "active model"
        return "Trained for \(adapter.baseModelID); active model is \(activeName)."
    }
}
