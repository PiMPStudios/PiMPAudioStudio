import Foundation
import Testing
@testable import PiMPAudioCore

@Test func datasetPackageV2RoundTripsAndProjectsDefaultFullMix() throws {
    let package = makeWholeTrackPackage()

    let validation = DatasetPackageV2Validator.validate(package)
    #expect(validation.isValid)
    #expect(validation.warnings.isEmpty)

    let data = try DatasetPackageV2Codec.encodeValidated(package)
    let decoded = try DatasetPackageV2Codec.decodeValidated(data)
    #expect(decoded == package)

    let object = try JSONSerialization.jsonObject(with: data) as? [String: Any]
    #expect(object?["pimp_audio_dataset_version"] as? String == "2.0")
    #expect((object?["source_tracks"] as? [[String: Any]])?.count == 1)
    #expect((object?["training_plans"] as? [[String: Any]])?.count == 1)

    let bundle = try DatasetTrainingBundleV1Projection.makeBundle(from: package)
    #expect(bundle.pimpAudioTrainingBundleVersion == "1.0")
    #expect(bundle.sourcePackageVersion == "2.0")
    #expect(bundle.backendID == "ace-step-local")
    #expect(bundle.include == ["full_mix"])
    #expect(bundle.clipCount == 1)
    #expect(bundle.clips[0].file == "sources/track-one.wav")
    #expect(bundle.clips[0].lyrics == "[Verse]\nKeep moving through the night")
    #expect(bundle.clips[0].trainingRange == DatasetTimeRangeV2(startSeconds: 0, endSeconds: 305))
}

@Test func datasetPackageV2KeepsBackendLimitAndSampleWindowExplicit() throws {
    var package = makeWholeTrackPackage()
    package.trainingPlans[0] = DatasetTrainingPlanV2(
        id: "remote-side-step",
        backendID: "side-step-cuda",
        name: "Remote Side-Step",
        isDefault: true,
        selections: [
            DatasetTrainingSelectionV2(
                assetID: "asset-track-one-full-mix",
                includeInTraining: true,
                trainingRange: DatasetTimeRangeV2(startSeconds: 0, endSeconds: 240),
                rangePolicy: .backendLimit,
                sampleWindowSeconds: 60
            ),
            DatasetTrainingSelectionV2(
                assetID: "asset-track-one-vocals",
                includeInTraining: false,
                trainingRange: DatasetTimeRangeV2(startSeconds: 0, endSeconds: 240),
                rangePolicy: .backendLimit,
                sampleWindowSeconds: 60
            ),
        ]
    )

    let validation = DatasetPackageV2Validator.validate(package)
    #expect(validation.isValid)
    #expect(validation.warnings.isEmpty)

    let bundle = try DatasetTrainingBundleV1Projection.makeBundle(from: package)
    #expect(bundle.clipCount == 1)
    #expect(bundle.clips[0].trainingRange.endSeconds == 240)
    #expect(bundle.clips[0].sampleWindowSeconds == 60)
    #expect(bundle.clips[0].rangePolicy == .backendLimit)
}

@Test func datasetPackageV2WarnsWhenRepresentationsMultiplyTrackWeight() {
    var package = makeWholeTrackPackage()
    package.trainingPlans[0].selections[1].includeInTraining = true

    let validation = DatasetPackageV2Validator.validate(package)

    #expect(validation.isValid)
    #expect(validation.warnings.contains {
        $0.message.contains("increases its effective training weight")
    })
}

@Test func datasetPackageV2RejectsUnsafePathsBrokenReferencesAndInvalidRanges() {
    var package = makeWholeTrackPackage()
    package.assets[0].file.path = "../escape.wav"
    package.assets[1].sourceTrackID = "missing-track"
    package.sourceTracks[0].sourceFile?.path = "sources\\track-one.wav"
    package.trainingPlans[0].selections[0].trainingRange = DatasetTimeRangeV2(
        startSeconds: 0,
        endSeconds: 400
    )

    let validation = DatasetPackageV2Validator.validate(package)

    #expect(!validation.isValid)
    #expect(validation.errors.contains { $0.path == "assets[0].file.path" })
    #expect(validation.errors.contains { $0.path == "assets[1].source_track_id" })
    #expect(validation.errors.contains { $0.path == "source_tracks[0].source_file.path" })
    #expect(validation.errors.contains {
        $0.path == "training_plans[0].selections[0].training_range"
    })
}

@Test func datasetPackageV2RejectsDerivationCyclesWithoutCrashingOnDuplicateIDs() {
    var package = makeWholeTrackPackage()
    package.assets[0].derivedFrom = package.assets[1].id
    package.assets[1].derivedFrom = package.assets[0].id
    package.assets.append(package.assets[0])

    let validation = DatasetPackageV2Validator.validate(package)

    #expect(!validation.isValid)
    #expect(validation.errors.contains { $0.message.contains("duplicated") })
    #expect(validation.errors.contains { $0.message.contains("cycles") })
}

@Test func datasetPackageV1AdapterPreservesLegacyClipsStemsAndAnnotations() throws {
    let manifest = Data("""
    {
      "pimp_audio_dataset_version": "1.0",
      "name": "Legacy Dark Dataset",
      "created_at": "2026-07-01T00:00:00Z",
      "clip_length_seconds": 30,
      "source": {
        "kind": "local_files",
        "files": [
          {
            "file_name": "long-source.wav",
            "duration_seconds": 305,
            "sample_rate": 48000,
            "channels": 2
          }
        ]
      }
    }
    """.utf8)
    let captions = Data("""
    [
      {
        "id": "clip_0001",
        "file": "clips/f001_clip_001.wav",
        "assets": {
          "full_mix": "clips/f001_clip_001.wav",
          "vocals": "stems/f001_clip_001/vocals.wav",
          "instrumental": "stems/f001_clip_001/instrumental.wav"
        },
        "caption": "dark blues with a low vocal",
        "caption_source": "manual",
        "lyrics": "I keep moving",
        "lyrics_source": "native-qwen3-asr",
        "source": {
          "file_name": "long-source.wav",
          "start_seconds": 10
        },
        "audio": {
          "format": "wav",
          "duration_seconds": 30,
          "sample_rate": 48000,
          "channels": 2
        },
        "annotation": {
          "annotation_backend": "native-qwen3-asr",
          "model_id": "qwen3-asr",
          "model_revision": "local",
          "review_status": "approved",
          "raw_output_files": {
            "lyrics": "stems/f001_clip_001/lyrics.txt"
          }
        },
        "training_review": {
          "approved": true,
          "status": "approved"
        },
        "separation": {
          "backend": "demucs-mlx",
          "model": "htdemucs"
        }
      },
      {
        "id": "clip_0002",
        "file": "clips/f001_clip_002.wav",
        "caption": "dark blues guitar solo",
        "source": {
          "file_name": "long-source.wav",
          "start_seconds": 100
        },
        "audio": {
          "duration_seconds": 30
        }
      }
    ]
    """.utf8)

    let package = try DatasetPackageV1Adapter.adapt(
        manifestData: manifest,
        captionsData: captions,
        adaptedAt: "2026-07-18T00:00:00Z"
    )
    let validation = DatasetPackageV2Validator.validate(package)

    #expect(validation.isValid)
    #expect(package.sourceTracks.count == 1)
    #expect(package.sourceTracks[0].originalFileName == "long-source.wav")
    #expect(package.sourceTracks[0].durationSeconds == 305)
    #expect(package.assets.count == 4)
    #expect(package.assets.filter { $0.role == .clip }.count == 2)
    #expect(package.assets.filter { $0.role == .vocals }.count == 1)
    #expect(package.assets.filter { $0.role == .instrumental }.count == 1)
    #expect(package.assets.first { $0.role == .vocals }?.derivedFrom != nil)
    #expect(package.assets.first { $0.role == .instrumental }?.annotation?.lyrics == nil)
    #expect(package.assets.first { $0.role == .clip }?.annotation?.caption?.text == "dark blues with a low vocal")
    #expect(package.migration?.sourceVersion == "1.0")
    #expect(package.migration?.adapterVersion == "1.0")

    let bundle = try DatasetTrainingBundleV1Projection.makeBundle(from: package)
    #expect(bundle.clipCount == 2)
    #expect(bundle.include == ["clip"])
    #expect(bundle.clips.allSatisfy { $0.assetRole == .clip })
    #expect(bundle.clips[0].sourceRange == DatasetTimeRangeV2(startSeconds: 10, endSeconds: 40))
}

@Test func datasetPackageV1AdapterRejectsUnsafeLegacyPaths() {
    let manifest = Data("""
    {
      "pimp_audio_dataset_version": "1.0",
      "name": "Unsafe Legacy Dataset",
      "created_at": "2026-07-01T00:00:00Z"
    }
    """.utf8)
    let captions = Data("""
    [
      {
        "id": "clip_0001",
        "file": "../outside.wav",
        "caption": "unsafe legacy clip",
        "audio": {
          "duration_seconds": 30
        }
      }
    ]
    """.utf8)

    #expect(throws: DatasetPackageV1AdapterError.self) {
        try DatasetPackageV1Adapter.adapt(
            manifestData: manifest,
            captionsData: captions
        )
    }
}

private func makeWholeTrackPackage() -> DatasetPackageV2 {
    let captionProvenance = DatasetAnnotationProvenanceV2(
        backendID: "ace-captioner-native",
        backendVersion: "1.0",
        modelID: "ACE-Step/acestep-captioner",
        modelRevision: "7109e9",
        promptVersion: "ace-caption-v1",
        quantization: "4bit-group64",
        generatedAt: "2026-07-18T00:00:00Z",
        rawOutputFiles: ["caption": "annotations/track-one/caption.txt"]
    )
    let lyricsProvenance = DatasetAnnotationProvenanceV2(
        backendID: "ace-transcriber-native",
        backendVersion: "1.0",
        modelID: "ACE-Step/acestep-transcriber",
        modelRevision: "53062c",
        promptVersion: "ace-transcribe-v1",
        quantization: "4bit-group64",
        generatedAt: "2026-07-18T00:01:00Z",
        rawOutputFiles: ["lyrics": "annotations/track-one/lyrics.txt"]
    )
    let annotation = DatasetAssetAnnotationV2(
        caption: DatasetTextAnnotationV2(
            text: "dark blues, slow tempo, atmospheric guitar",
            source: "ace-step-captioner",
            language: "en",
            provenance: captionProvenance
        ),
        lyrics: DatasetTextAnnotationV2(
            text: "[Verse]\nKeep moving through the night",
            source: "ace-step-transcriber",
            language: "en",
            provenance: lyricsProvenance
        ),
        reviewStatus: "approved",
        approved: true,
        reviewedAt: "2026-07-18T00:02:00Z"
    )
    let sourceRange = DatasetTimeRangeV2(startSeconds: 0, endSeconds: 305)
    return DatasetPackageV2(
        id: "dark-gentlemen",
        name: "Dark Gentlemen",
        createdAt: "2026-07-18T00:00:00Z",
        sourceTracks: [
            DatasetSourceTrackV2(
                id: "track-one",
                originalFileName: "track-one.wav",
                durationSeconds: 305,
                sourceFile: DatasetAudioFileV2(
                    path: "sources/track-one.wav",
                    format: "wav",
                    byteSize: 10_000,
                    durationSeconds: 305,
                    sampleRate: 48_000,
                    channels: 2
                )
            ),
        ],
        assets: [
            DatasetAssetV2(
                id: "asset-track-one-full-mix",
                sourceTrackID: "track-one",
                role: .fullMix,
                file: DatasetAudioFileV2(
                    path: "sources/track-one.wav",
                    format: "wav",
                    byteSize: 10_000,
                    durationSeconds: 305,
                    sampleRate: 48_000,
                    channels: 2
                ),
                sourceRange: sourceRange,
                derivation: DatasetAssetDerivationV2(kind: .imported),
                annotation: annotation
            ),
            DatasetAssetV2(
                id: "asset-track-one-vocals",
                sourceTrackID: "track-one",
                role: .vocals,
                file: DatasetAudioFileV2(
                    path: "stems/track-one/vocals.wav",
                    format: "wav",
                    durationSeconds: 305
                ),
                derivedFrom: "asset-track-one-full-mix",
                sourceRange: sourceRange,
                derivation: DatasetAssetDerivationV2(
                    kind: .stemSeparation,
                    backendID: "demucs-mlx",
                    modelID: "htdemucs"
                ),
                annotation: annotation
            ),
        ],
        trainingPlans: [
            DatasetTrainingPlanV2(
                id: "local-full-mix",
                backendID: "ace-step-local",
                name: "Local full mix",
                isDefault: true,
                selections: [
                    DatasetTrainingSelectionV2(
                        assetID: "asset-track-one-full-mix",
                        includeInTraining: true,
                        trainingRange: sourceRange,
                        rangePolicy: .fullSource
                    ),
                    DatasetTrainingSelectionV2(
                        assetID: "asset-track-one-vocals",
                        includeInTraining: false,
                        trainingRange: sourceRange,
                        rangePolicy: .fullSource
                    ),
                ]
            ),
        ]
    )
}
