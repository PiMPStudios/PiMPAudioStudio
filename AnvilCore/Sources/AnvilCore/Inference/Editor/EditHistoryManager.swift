import Foundation

// MARK: - EditHistoryManager

/// Manages a multi-level undo/redo stack of ``EditSnapshot`` instances.
public class EditHistoryManager: @unchecked Sendable {
    private var stack: [EditSnapshot] = []
    private var currentIndex: Int = -1
    private let maxSnapshotsLimit: Int

    public init(maxSnapshots: Int = 20) {
        self.maxSnapshotsLimit = maxSnapshots
    }

    /// Whether there is a previous snapshot to revert to.
    public var canUndo: Bool { currentIndex > 0 }

    /// Whether there is a next snapshot to advance to.
    public var canRedo: Bool { currentIndex < stack.count - 1 }

    /// Total number of snapshots currently stored in the stack.
    public var stackCount: Int { stack.count }

    /// Returns the most recent snapshots (up to 5) up to and including the current position.
    public var recentEdits: [EditSnapshot] {
        guard !stack.isEmpty else { return [] }
        let end = currentIndex + 1
        let start = max(0, end - 5)
        return Array(stack[start..<end])
    }

    /// The snapshot at the current position, or `nil` if the stack is empty.
    public var currentSnapshot: EditSnapshot? {
        guard currentIndex >= 0, currentIndex < stack.count else { return nil }
        return stack[currentIndex]
    }

    /// Pushes a new snapshot onto the stack, discarding any redo history beyond the current position.
    ///
    /// If the stack exceeds `maxSnapshots`, the oldest entry is removed.
    public func push(_ snapshot: EditSnapshot) {
        // Discard redo history beyond current position.
        if currentIndex < stack.count - 1 {
            stack = Array(stack[...currentIndex])
        }
        stack.append(snapshot)
        currentIndex = stack.count - 1
        // Cap memory.
        if stack.count > maxSnapshotsLimit {
            stack.removeFirst()
            currentIndex -= 1
        }
    }

    /// Moves the current position back one step and returns that snapshot.
    ///
    /// Returns `nil` if ``canUndo`` is `false`.
    @discardableResult
    public func undo() -> EditSnapshot? {
        guard canUndo else { return nil }
        currentIndex -= 1
        return stack[currentIndex]
    }

    /// Moves the current position forward one step and returns that snapshot.
    ///
    /// Returns `nil` if ``canRedo`` is `false`.
    @discardableResult
    public func redo() -> EditSnapshot? {
        guard canRedo else { return nil }
        currentIndex += 1
        return stack[currentIndex]
    }

    /// Clears the entire stack and resets the current position.
    public func reset() {
        stack.removeAll()
        currentIndex = -1
    }
}
