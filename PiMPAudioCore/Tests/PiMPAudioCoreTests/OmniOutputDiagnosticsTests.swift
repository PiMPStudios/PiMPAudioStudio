import XCTest
@testable import PiMPAudioCore

final class OmniOutputDiagnosticsTests: XCTestCase {
    func testEmptyOutputIsReportedWithoutRepetition() {
        let diagnostics = OmniOutputDiagnostics(text: " \n ")

        XCTAssertTrue(diagnostics.isEmpty)
        XCTAssertEqual(diagnostics.lineCount, 0)
        XCTAssertFalse(diagnostics.isPathologicalRepetition)
    }

    func testOrdinaryRepeatedChorusIsNotPathological() {
        let text = """
        [Verse]
        A new line
        [Chorus]
        The same refrain
        [Verse 2]
        Another line
        [Chorus]
        The same refrain
        """
        let diagnostics = OmniOutputDiagnostics(text: text)

        XCTAssertFalse(diagnostics.isEmpty)
        XCTAssertEqual(diagnostics.maximumRepeatedLineCount, 2)
        XCTAssertFalse(diagnostics.isPathologicalRepetition)
    }

    func testInstrumentalWallIsPathological() {
        let text = Array(
            repeating: "[Instrumental Section]",
            count: 8
        ).joined(separator: "\n")
        let diagnostics = OmniOutputDiagnostics(text: text)

        XCTAssertEqual(diagnostics.instrumentalLineCount, 8)
        XCTAssertTrue(diagnostics.isPathologicalRepetition)
    }
}
