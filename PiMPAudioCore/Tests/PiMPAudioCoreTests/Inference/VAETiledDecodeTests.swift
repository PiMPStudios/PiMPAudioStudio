import MLX
@testable import PiMPAudioCore
import Testing

@Suite("VAE Tiled Decode")
struct VAETiledDecodeTests {
    @Test("Short sequence does not tile")
    func shortSequenceNoTile() {
        let vae = AutoEncoderOobleck()
        let latent = MLXArray.zeros([1, 100, 64])
        let audio = VAETiledDecode.decode(latent, using: vae)
        eval(audio)
        #expect(audio.shape[0] == 1)
        #expect(audio.shape[2] == 2)
    }

    @Test("Tile count calculation")
    func tileCount() {
        let tiles = VAETiledDecode.computeTileCount(latentFrames: 3000)
        #expect(tiles == 2)
    }

    @Test("Exactly at threshold does not tile")
    func atThresholdNoTile() {
        let tiles = VAETiledDecode.computeTileCount(latentFrames: 2048)
        #expect(tiles == 1)
    }
}
