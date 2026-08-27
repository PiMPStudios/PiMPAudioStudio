import MLX
import MLXNN
import MLXRandom
@testable import PiMPAudioCore
import Testing

@Suite("VAE Model")
struct VAEModelTests {
    @Test("OobleckEncoder output shape")
    func encoderShape() {
        let encoder = OobleckEncoder(
            encoderHiddenSize: 128,
            audioChannels: 2,
            downsamplingRatios: [2, 4, 4, 8, 8],
            channelMultiples: [1, 2, 4, 8, 16]
        )
        let audio = MLXArray.zeros([1, 2048, 2])
        let latent = encoder(audio)
        eval(latent)
        #expect(latent.shape[0] == 1)
        #expect(latent.shape[2] == 128)
        #expect(latent.shape[1] == 1)
    }

    @Test("OobleckDecoder output shape")
    func decoderShape() {
        let decoder = OobleckDecoder(
            channels: 128,
            inputChannels: 64,
            audioChannels: 2,
            upsamplingRatios: [8, 8, 4, 4, 2],
            channelMultiples: [1, 2, 4, 8, 16]
        )
        let latent = MLXArray.zeros([1, 4, 64])
        let audio = decoder(latent)
        eval(audio)
        #expect(audio.shape[0] == 1)
        #expect(audio.shape[2] == 2)
        #expect(audio.shape[1] >= 8100 && audio.shape[1] <= 8300)
    }

    @Test("AutoEncoderOobleck encode_and_sample shape")
    func vaeEncodeSampleShape() {
        let vae = AutoEncoderOobleck()
        let audio = MLXArray.zeros([1, 4096, 2])
        let z = vae.encodeSample(audio)
        eval(z)
        #expect(z.shape[0] == 1)
        #expect(z.shape[2] == 64)
        #expect(z.shape[1] == 2)
    }

    @Test("AutoEncoderOobleck encode_and_sample is deterministic with a key")
    func vaeEncodeSampleKeyed() {
        let vae = AutoEncoderOobleck()
        let audio = MLXRandom.uniform(-1.0 ..< 1.0, [1, 4096, 2], key: MLXRandom.key(7))

        let a = vae.encodeSample(audio, key: MLXRandom.key(42))
        let b = vae.encodeSample(audio, key: MLXRandom.key(42))
        let c = vae.encodeSample(audio, key: MLXRandom.key(43))
        eval(a, b, c)

        let sameKeyDelta = abs(a - b).max().item(Float.self)
        let differentKeyDelta = abs(a - c).max().item(Float.self)
        #expect(sameKeyDelta == 0)
        #expect(differentKeyDelta > 0)
    }

    @Test("AutoEncoderOobleck decode shape")
    func vaeDecodeShape() {
        let vae = AutoEncoderOobleck()
        let latent = MLXArray.zeros([1, 2, 64])
        let audio = vae.decode(latent)
        eval(audio)
        #expect(audio.shape[0] == 1)
        #expect(audio.shape[2] == 2)
        #expect(audio.shape[1] == 3840)
    }
}
