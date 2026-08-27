import Foundation
@testable import PiMPAudioCore
import XCTest

final class NativeModelPackManifestTests: XCTestCase {
    func testValidManifestPassesAndCorruptionFailsClosed() throws {
        let directory = try makePackDirectory()
        let manifest = try makeManifest(in: directory)
        try write(manifest, to: directory)

        XCTAssertEqual(
            try NativeModelPackValidator.loadAndValidate(at: directory),
            manifest
        )

        try Data("corrupt".utf8).write(
            to: directory.appendingPathComponent("thinker.safetensors"),
            options: .atomic
        )
        XCTAssertThrowsError(
            try NativeModelPackValidator.loadAndValidate(at: directory)
        ) { error in
            guard case NativeModelPackError.sizeMismatch = error else {
                return XCTFail("Unexpected error: \(error)")
            }
        }
    }

    func testWrongRevisionFailsClosed() throws {
        let directory = try makePackDirectory()
        let valid = try makeManifest(in: directory)
        let manifest = NativeModelPackManifest(
            packID: valid.packID,
            task: valid.task,
            upstreamModelID: valid.upstreamModelID,
            upstreamRevision: String(repeating: "0", count: 40),
            upstreamLicense: valid.upstreamLicense,
            quantization: valid.quantization,
            tensorCounts: valid.tensorCounts,
            expectedSizeBytes: valid.expectedSizeBytes,
            excludedComponents: valid.excludedComponents,
            files: valid.files
        )
        try write(manifest, to: directory)

        XCTAssertThrowsError(
            try NativeModelPackValidator.loadAndValidate(at: directory)
        ) { error in
            guard case NativeModelPackError.invalidManifest = error else {
                return XCTFail("Unexpected error: \(error)")
            }
        }
    }

    func testUnsafeManifestPathFailsClosed() throws {
        let directory = try makePackDirectory()
        let valid = try makeManifest(in: directory)
        var files = valid.files
        let first = files.removeFirst()
        files.append(
            NativeModelPackManifest.FileRecord(
                role: first.role,
                path: "../outside",
                byteCount: first.byteCount,
                sha256: first.sha256
            )
        )
        let manifest = NativeModelPackManifest(
            packID: valid.packID,
            task: valid.task,
            upstreamModelID: valid.upstreamModelID,
            upstreamRevision: valid.upstreamRevision,
            upstreamLicense: valid.upstreamLicense,
            quantization: valid.quantization,
            tensorCounts: valid.tensorCounts,
            expectedSizeBytes: valid.expectedSizeBytes,
            excludedComponents: valid.excludedComponents,
            files: files
        )
        try write(manifest, to: directory)

        XCTAssertThrowsError(
            try NativeModelPackValidator.loadAndValidate(at: directory)
        ) { error in
            guard case NativeModelPackError.unsafePath = error else {
                return XCTFail("Unexpected error: \(error)")
            }
        }
    }

    private func makePackDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
        addTeardownBlock {
            try? FileManager.default.removeItem(at: directory)
        }

        let index: [String: Any] = [
            "weight_map": [
                "thinker.model.embed_tokens.weight": "thinker.safetensors",
                "thinker.audio_tower.conv1.weight": "audio-tower.safetensors",
            ],
        ]
        let indexData = try JSONSerialization.data(
            withJSONObject: index,
            options: [.prettyPrinted, .sortedKeys]
        )
        try indexData.write(
            to: directory.appendingPathComponent(
                "model.safetensors.index.json"
            ),
            options: .atomic
        )
        for path in NativeModelPackValidator.requiredFiles.values
            where path != "model.safetensors.index.json" {
            let data: Data = if path == "annotation-prompts.json" {
                try JSONSerialization.data(
                    withJSONObject: [
                        "schemaVersion": 1,
                        "caption": OmniAnnotationTask.caption.prompt,
                        "transcribe": OmniAnnotationTask.transcribe.prompt,
                    ],
                    options: [.prettyPrinted, .sortedKeys]
                )
            } else {
                Data(path.utf8)
            }
            try data.write(
                to: directory.appendingPathComponent(path),
                options: .atomic
            )
        }
        return directory
    }

    private func makeManifest(
        in directory: URL
    ) throws -> NativeModelPackManifest {
        let files = try NativeModelPackValidator.requiredFiles
            .map { role, path in
                try NativeModelPackValidator.record(
                    role: role,
                    path: path,
                    in: directory
                )
            }
            .sorted { $0.role < $1.role }
        return NativeModelPackManifest(
            packID: "ace-captioner-4bit-v1",
            task: .caption,
            upstreamModelID: "ACE-Step/acestep-captioner",
            upstreamRevision: OmniWeightLoader.captionerRevision,
            upstreamLicense: "MIT",
            quantization: .init(bits: 4, groupSize: 64),
            tensorCounts: .init(thinker: 1, audioTower: 1),
            expectedSizeBytes: files.reduce(0) { $0 + $1.byteCount },
            excludedComponents: ["talker", "token2wav", "vision"],
            files: files
        )
    }

    private func write(
        _ manifest: NativeModelPackManifest,
        to directory: URL
    ) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(manifest).write(
            to: directory.appendingPathComponent(
                NativeModelPackManifest.filename
            ),
            options: .atomic
        )
    }
}
