import Foundation
import XCTest
@testable import PiMPAudioCore

final class DatasetAnnotationBackendTests: XCTestCase {
    private final class LockedCounter: @unchecked Sendable {
        private let lock = NSLock()
        private var count = 0

        func increment() -> Int {
            lock.lock()
            defer { lock.unlock() }
            count += 1
            return count
        }

        var value: Int {
            lock.lock()
            defer { lock.unlock() }
            return count
        }
    }

    func testHardwareTiersMatchSupportedMemoryPolicy() {
        let gibibyte = UInt64(1_073_741_824)

        XCTAssertEqual(
            DatasetAnnotationHardwareTier.classify(
                physicalMemoryBytes: 16 * gibibyte
            ),
            .unsupported
        )
        XCTAssertEqual(
            DatasetAnnotationHardwareTier.classify(
                physicalMemoryBytes: 24 * gibibyte
            ),
            .provisional
        )
        XCTAssertEqual(
            DatasetAnnotationHardwareTier.classify(
                physicalMemoryBytes: 32 * gibibyte
            ),
            .recommended
        )
    }

    func testMissingModelPacksAreReportedWithoutLoadingMLX() async {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        let backend = NativeACEDatasetAnnotationBackend(
            modelsDirectory: directory,
            physicalMemoryBytes: 32 * 1_073_741_824
        )

        let availability = await backend.availability()

        XCTAssertEqual(availability.hardwareTier, .recommended)
        XCTAssertFalse(availability.isFullyInstalled)
        XCTAssertEqual(availability.models.count, 2)
        XCTAssertTrue(availability.models.allSatisfy { !$0.isInstalled })
    }

    func testPackHashingChecksCancellationBetweenChunks() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        try Data(repeating: 0x5A, count: 3 * 1_048_576).write(to: url)
        addTeardownBlock {
            try? FileManager.default.removeItem(at: url)
        }

        let checks = LockedCounter()
        XCTAssertThrowsError(
            try NativeModelPackValidator.sha256(
                of: url,
                cancellationCheck: {
                    if checks.increment() == 2 {
                        throw CancellationError()
                    }
                }
            )
        ) { error in
            XCTAssertTrue(error is CancellationError)
        }
        XCTAssertEqual(checks.value, 2)
    }

    func testInstalledNativePacksAnnotateRealAudioWithoutPython() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard environment["PIMP_RUN_NATIVE_ACE_INTEGRATION"] == "1" else {
            throw XCTSkip(
                "Set PIMP_RUN_NATIVE_ACE_INTEGRATION=1 and the native pack/audio paths to run this integration test."
            )
        }
        guard let captionerPath = environment["PIMP_NATIVE_ACE_CAPTIONER_PACK"],
              let transcriberPath = environment["PIMP_NATIVE_ACE_TRANSCRIBER_PACK"],
              let audioPath = environment["PIMP_NATIVE_ACE_AUDIO"]
        else {
            XCTFail("Native ACE integration paths are incomplete.")
            return
        }

        let backend = NativeACEDatasetAnnotationBackend(
            captionerDirectory: URL(fileURLWithPath: captionerPath),
            transcriberDirectory: URL(fileURLWithPath: transcriberPath),
            physicalMemoryBytes: 32 * 1_073_741_824
        )
        let availability = await backend.availability()
        XCTAssertTrue(availability.isFullyInstalled)

        let request = DatasetAnnotationRequest(
            id: "native-integration",
            audioURL: URL(fileURLWithPath: audioPath)
        )
        for task in [DatasetAnnotationTask.caption, .transcribe] {
            let results = try await backend.annotate(
                requests: [request],
                task: task,
                options: .greedy
            )
            let result = try XCTUnwrap(results.first)
            XCTAssertEqual(result.requestID, request.id)
            XCTAssertEqual(result.task, task)
            XCTAssertFalse(result.diagnostics.isEmpty)
            XCTAssertFalse(result.diagnostics.isPathologicalRepetition)
            XCTAssertEqual(result.descriptor.id, backend.backendID)
            XCTAssertEqual(result.descriptor.quantization, "q4-affine-g64")
        }
    }
}
