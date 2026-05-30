import Foundation
import AnvilCore
import MLX

/// Manages the runtime state of multiple stacked LoRA adapters.
///
/// Owns which adapters are active, at what strength, and coordinates
/// with LoRAInjector to apply/update/clear contributions on the model.
@MainActor
final class AdapterStackManager: ObservableObject {

    // MARK: - Published State

    /// The primary adapter shown in the main AdapterPicker.
    @Published var primaryAdapter: LoRAAdapter?

    /// Strength for the primary adapter. Upstream PEFT LoRA exposes this as
    /// lora_scale, and XL adapters often need A/B tuning below 1.0.
    @Published var primaryStrength: Float = 1.0

    /// Additional stacked adapters with their individual strengths.
    @Published var additionalAdapters: [(adapter: LoRAAdapter, strength: Float)] = []

    // MARK: - Constants

    /// Maximum total adapters (primary + additional).
    static let maxStackSize = 3

    /// Maximum additional adapters that can be stacked.
    var maxAdditional: Int { Self.maxStackSize - (primaryAdapter != nil ? 1 : 0) }

    /// Whether the stack is at capacity.
    var isAtCapacity: Bool {
        let total = (primaryAdapter != nil ? 1 : 0) + additionalAdapters.count
        return total >= Self.maxStackSize
    }

    /// Total combined strength of all active adapters.
    var totalStrength: Float {
        var total: Float = primaryAdapter != nil ? primaryStrength : 0.0
        for entry in additionalAdapters {
            total += entry.strength
        }
        return total
    }

    // MARK: - Stack Management

    /// Set the primary adapter. Replaces any existing primary.
    func setPrimary(_ adapter: LoRAAdapter?) {
        primaryAdapter = adapter
        if let adapter {
            additionalAdapters.removeAll { $0.adapter.id == adapter.id }
        }
    }

    /// Update the primary adapter strength.
    func setPrimaryStrength(_ strength: Float) {
        primaryStrength = min(max(strength, 0.0), 1.0)
    }

    /// Add an adapter to the additional stack.
    /// - Parameters:
    ///   - adapter: The adapter to add.
    ///   - strength: Initial strength (default 0.5).
    func addToStack(_ adapter: LoRAAdapter, strength: Float = 0.5) {
        guard !isAtCapacity else { return }
        guard !additionalAdapters.contains(where: { $0.adapter.id == adapter.id }) else { return }
        guard adapter.id != primaryAdapter?.id else { return }
        if let primaryAdapter {
            guard adapter.kind == primaryAdapter.kind else { return }
        }
        additionalAdapters.append((adapter: adapter, strength: strength))
    }

    /// Remove an adapter from the additional stack by ID.
    func removeFromStack(adapterID: String) {
        additionalAdapters.removeAll { $0.adapter.id == adapterID }
    }

    /// Update strength for a stacked adapter.
    func setStrength(adapterID: String, strength: Float) {
        guard let idx = additionalAdapters.firstIndex(where: { $0.adapter.id == adapterID }) else {
            return
        }
        additionalAdapters[idx].strength = strength
    }

    /// Clear all additional adapters (keep primary).
    func clearStack() {
        additionalAdapters = []
    }

    /// Clear everything (primary + additional).
    func clearAll() {
        primaryAdapter = nil
        additionalAdapters = []
    }

    /// All adapters currently in the stack (for display purposes).
    var allStackedAdapters: [(adapter: LoRAAdapter, strength: Float)] {
        var result: [(adapter: LoRAAdapter, strength: Float)] = []
        if let primary = primaryAdapter {
            result.append((adapter: primary, strength: primaryStrength))
        }
        result.append(contentsOf: additionalAdapters)
        return result
    }

    /// Whether stacking is active (more than just a primary adapter).
    var isStacking: Bool {
        !additionalAdapters.isEmpty
    }

    /// Handle adapter deletion — remove from stack if present.
    func handleAdapterDeleted(adapterID: String) {
        if primaryAdapter?.id == adapterID {
            // Promote first additional to primary, or clear
            if let first = additionalAdapters.first {
                primaryAdapter = first.adapter
                additionalAdapters.removeFirst()
            } else {
                primaryAdapter = nil
            }
        } else {
            removeFromStack(adapterID: adapterID)
        }
    }
}
