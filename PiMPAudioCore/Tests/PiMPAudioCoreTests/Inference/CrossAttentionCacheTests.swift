import Testing
import MLX
@testable import PiMPAudioCore

@Suite("CrossAttentionCache")
struct CrossAttentionCacheTests {
    @Test("Empty cache reports not updated")
    func emptyCache() {
        let cache = CrossAttentionCache()
        #expect(!cache.isUpdated(layerIdx: 0))
        #expect(cache.get(layerIdx: 0) == nil)
    }

    @Test("Update and retrieve")
    func updateAndRetrieve() {
        let cache = CrossAttentionCache()
        let key = MLXArray.zeros([1, 4, 10, 128])
        let value = MLXArray.ones([1, 4, 10, 128])
        cache.update(key: key, value: value, layerIdx: 5)

        #expect(cache.isUpdated(layerIdx: 5))
        #expect(!cache.isUpdated(layerIdx: 0))

        let result = cache.get(layerIdx: 5)
        #expect(result != nil)
        #expect(result!.key.shape == [1, 4, 10, 128])
        #expect(result!.value.shape == [1, 4, 10, 128])
    }

    @Test("Reset clears all entries")
    func resetClears() {
        let cache = CrossAttentionCache()
        let key = MLXArray.zeros([1, 4, 10, 128])
        let value = MLXArray.ones([1, 4, 10, 128])
        cache.update(key: key, value: value, layerIdx: 0)
        cache.update(key: key, value: value, layerIdx: 1)

        cache.reset()

        #expect(!cache.isUpdated(layerIdx: 0))
        #expect(!cache.isUpdated(layerIdx: 1))
        #expect(cache.get(layerIdx: 0) == nil)
    }

    @Test("Multiple layers independent")
    func multipleLayers() {
        let cache = CrossAttentionCache()
        let k0 = MLXArray.zeros([1, 4, 10, 128])
        let v0 = MLXArray.zeros([1, 4, 10, 128])
        let k1 = MLXArray.ones([1, 4, 20, 128])
        let v1 = MLXArray.ones([1, 4, 20, 128])

        cache.update(key: k0, value: v0, layerIdx: 0)
        cache.update(key: k1, value: v1, layerIdx: 1)

        let r0 = cache.get(layerIdx: 0)!
        let r1 = cache.get(layerIdx: 1)!
        #expect(r0.key.shape[2] == 10)
        #expect(r1.key.shape[2] == 20)
    }
}
