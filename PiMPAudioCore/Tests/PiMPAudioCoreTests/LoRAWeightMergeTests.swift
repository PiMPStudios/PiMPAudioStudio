import Foundation
import MLX
import Testing
@testable import PiMPAudioCore

private func floatArray(_ values: [Float], _ shape: [Int]) -> MLXArray {
    MLXArray(values, shape)
}

@Test func loRAWeightMergeMatchesScaledDeltaBlend() throws {
    // rank=1, in=2, out=2 — easy closed-form deltas.
    // A1 = [[1, 0]], B1 = [[2], [0]]  => B1@A1 = [[2, 0], [0, 0]]
    // A2 = [[0, 1]], B2 = [[0], [3]]  => B2@A2 = [[0, 0], [0, 3]]
    let a1 = floatArray([1, 0], [1, 2])
    let b1 = floatArray([2, 0], [2, 1])
    let a2 = floatArray([0, 1], [1, 2])
    let b2 = floatArray([0, 3], [2, 1])

    let path = "layers.0.selfAttn.qProj"
    let weightsA = ["\(path).loraA": a1, "\(path).loraB": b1]
    let weightsB = ["\(path).loraA": a2, "\(path).loraB": b2]

    let blend: Float = 0.25
    let scaleA: Float = 2
    let scaleB: Float = 4
    let result = try LoRAWeightMerge.blendDeltas(
        weightsA: weightsA,
        weightsB: weightsB,
        weightOnA: blend,
        scalingA: scaleA,
        scalingB: scaleB
    )

    #expect(result.rank == 2)
    #expect(result.alpha == 2)

    let dict = Dictionary(uniqueKeysWithValues: result.weights.map { ($0.0, $0.1) })
    let mergedA = try #require(dict["\(path).loraA"])
    let mergedB = try #require(dict["\(path).loraB"])
    #expect(mergedA.shape == [2, 2])
    #expect(mergedB.shape == [2, 2])

    // With alpha == rank, runtime scaling is 1, so delta == B@A.
    let mergedDelta = matmul(mergedB, mergedA)
    eval(mergedDelta)

    // Desired: c1*(B1@A1) + c2*(B2@A2)
    // c1 = 0.25*2 = 0.5, c2 = 0.75*4 = 3
    // delta = 0.5*[[2,0],[0,0]] + 3*[[0,0],[0,3]] = [[1,0],[0,9]]
    let expected = floatArray([1, 0, 0, 9], [2, 2])
    eval(expected)

    let mergedFlat = mergedDelta.asArray(Float.self)
    let expectedFlat = expected.asArray(Float.self)
    #expect(mergedFlat.count == expectedFlat.count)
    for (lhs, rhs) in zip(mergedFlat, expectedFlat) {
        #expect(abs(lhs - rhs) < 1e-5)
    }
}

@Test func loRAWeightMergeRejectsFactorInterpolationRegression() throws {
    // If someone interpolated A and B independently at w=0.5 with unit scale:
    // A' = 0.5*(A1+A2), B' = 0.5*(B1+B2)
    // B'@A' has cross terms and is NOT 0.5*(B1@A1 + B2@A2).
    let a1 = floatArray([1, 0], [1, 2])
    let b1 = floatArray([2, 0], [2, 1])
    let a2 = floatArray([0, 1], [1, 2])
    let b2 = floatArray([0, 4], [2, 1])

    let path = "layers.0.selfAttn.qProj"
    let result = try LoRAWeightMerge.blendDeltas(
        weightsA: ["\(path).loraA": a1, "\(path).loraB": b1],
        weightsB: ["\(path).loraA": a2, "\(path).loraB": b2],
        weightOnA: 0.5,
        scalingA: 1,
        scalingB: 1
    )

    let dict = Dictionary(uniqueKeysWithValues: result.weights.map { ($0.0, $0.1) })
    let mergedDelta = matmul(dict["\(path).loraB"]!, dict["\(path).loraA"]!)
    eval(mergedDelta)

    // True blend: 0.5*[[2,0],[0,0]] + 0.5*[[0,0],[0,4]] = [[1,0],[0,2]]
    let trueBlend: [Float] = [1, 0, 0, 2]
    // Wrong independent-factor result would be [[0.5, 0.5], [1, 1]]
    let wrongFactorInterp: [Float] = [0.5, 0.5, 1, 1]

    let flat = mergedDelta.asArray(Float.self)
    for (lhs, rhs) in zip(flat, trueBlend) {
        #expect(abs(lhs - rhs) < 1e-5)
    }
    let matchesWrong = zip(flat, wrongFactorInterp).allSatisfy { abs($0 - $1) < 1e-5 }
    #expect(!matchesWrong)
}

@Test func loRAWeightMergeRejectsUnpairedKeys() {
    #expect(throws: LoRAWeightMergeError.self) {
        try LoRAWeightMerge.blendDeltas(
            weightsA: ["layers.0.selfAttn.qProj.loraA": floatArray([1, 0], [1, 2])],
            weightsB: [
                "layers.0.selfAttn.qProj.loraA": floatArray([0, 1], [1, 2]),
                "layers.0.selfAttn.qProj.loraB": floatArray([0, 1], [2, 1]),
            ],
            weightOnA: 0.5,
            scalingA: 1,
            scalingB: 1
        )
    }
}
