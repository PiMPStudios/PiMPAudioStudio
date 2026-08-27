import Foundation
@testable import PiMPAudioCore
import Testing

// MARK: - EditHistoryManager Tests

@Suite("EditHistoryManager")
struct EditHistoryManagerTests {
    // MARK: Helpers

    private func makeSnapshot(description: String) -> EditSnapshot {
        EditSnapshot(description: description, audioData: Data([0x01]), latents: Data([0x02]))
    }

    // MARK: 1 — Initial state

    @Test("Initially cannot undo or redo")
    func initialState() {
        let manager = EditHistoryManager()
        #expect(manager.canUndo == false)
        #expect(manager.canRedo == false)
        #expect(manager.stackCount == 0)
        #expect(manager.currentSnapshot == nil)
        #expect(manager.recentEdits.isEmpty)
    }

    // MARK: 2 — Single push vs. two pushes

    @Test("One push cannot undo; two pushes can undo")
    func pushEnablesUndo() {
        let manager = EditHistoryManager()
        manager.push(makeSnapshot(description: "first"))
        #expect(manager.canUndo == false)
        #expect(manager.stackCount == 1)

        manager.push(makeSnapshot(description: "second"))
        #expect(manager.canUndo == true)
        #expect(manager.stackCount == 2)
    }

    // MARK: 3 — Undo returns the previous snapshot

    @Test("Undo returns the previous snapshot description")
    func undoReturnsPreviousSnapshot() {
        let manager = EditHistoryManager()
        manager.push(makeSnapshot(description: "alpha"))
        manager.push(makeSnapshot(description: "beta"))

        let result = manager.undo()
        #expect(result?.description == "alpha")
        #expect(manager.currentSnapshot?.description == "alpha")
    }

    // MARK: 4 — Redo returns the next snapshot

    @Test("Redo returns the next snapshot after undo")
    func redoReturnsNextSnapshot() {
        let manager = EditHistoryManager()
        manager.push(makeSnapshot(description: "v1"))
        manager.push(makeSnapshot(description: "v2"))
        manager.push(makeSnapshot(description: "v3"))

        manager.undo() // → v2
        manager.undo() // → v1

        let result = manager.redo()
        #expect(result?.description == "v2")
        #expect(manager.currentSnapshot?.description == "v2")
    }

    // MARK: 5 — Push after undo discards redo history

    @Test("Push after undo discards redo history")
    func pushAfterUndoDiscardsRedo() {
        let manager = EditHistoryManager()
        manager.push(makeSnapshot(description: "a"))
        manager.push(makeSnapshot(description: "b"))
        manager.push(makeSnapshot(description: "c"))

        manager.undo() // back to "b"
        #expect(manager.canRedo == true)

        manager.push(makeSnapshot(description: "d"))
        #expect(manager.canRedo == false)
        #expect(manager.currentSnapshot?.description == "d")
        #expect(manager.stackCount == 3) // a, b, d
    }

    // MARK: 6 — Max snapshots cap

    @Test("Stack count is capped at maxSnapshots")
    func maxSnapshotsCap() {
        let manager = EditHistoryManager(maxSnapshots: 5)
        for i in 1 ... 10 {
            manager.push(makeSnapshot(description: "snap\(i)"))
        }
        #expect(manager.stackCount == 5)
        // The most recent snapshot should be "snap10"
        #expect(manager.currentSnapshot?.description == "snap10")
    }

    // MARK: 7 — recentEdits returns up to last 5 entries

    @Test("recentEdits returns up to 5 most recent snapshots")
    func recentEditsReturnsLastFive() {
        let manager = EditHistoryManager()
        for i in 1 ... 8 {
            manager.push(makeSnapshot(description: "edit\(i)"))
        }
        let recent = manager.recentEdits
        #expect(recent.count == 5)
        #expect(recent.first?.description == "edit4")
        #expect(recent.last?.description == "edit8")
    }
}
