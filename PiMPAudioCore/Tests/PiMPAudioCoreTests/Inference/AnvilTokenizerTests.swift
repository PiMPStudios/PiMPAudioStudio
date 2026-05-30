import Testing
import MLX
@testable import PiMPAudioCore

@Suite("Anvil Tokenizer")
struct AnvilTokenizerTests {
    @Test("AnvilTokenizer encodeText returns capped ids and mask")
    func encodeTextShape() throws {
        let tokenizer = try AnvilTokenizer.forTesting()
        let (ids, mask) = tokenizer.encodeText("Hello world", maxLength: 256)
        #expect(ids.shape[0] == 1)
        #expect(ids.shape[1] > 0)
        #expect(ids.shape[1] <= 256)
        #expect(mask.shape == ids.shape)
    }

    @Test("AnvilTokenizer encodeLyrics respects max length")
    func encodeLyricsShape() throws {
        let tokenizer = try AnvilTokenizer.forTesting()
        let (ids, mask) = tokenizer.encodeLyrics("La la la", maxLength: 2048)
        #expect(ids.shape[0] == 1)
        #expect(ids.shape[1] > 0)
        #expect(ids.shape[1] <= 2048)
        #expect(mask.shape == ids.shape)
    }

    @Test("AnvilTokenizer mask has 1s for real tokens")
    func maskValues() throws {
        let tokenizer = try AnvilTokenizer.forTesting()
        let (_, mask) = tokenizer.encodeText("Hi", maxLength: 10)
        let maskArray = mask.asArray(Int32.self)
        let realTokenCount = maskArray.filter { $0 == 1 }.count
        #expect(realTokenCount == maskArray.count)
    }

    @Test("AnvilTokenizer emits shaped placeholder for empty input")
    func emptyInputPlaceholder() throws {
        let tokenizer = try AnvilTokenizer.forTesting()
        let (ids, mask) = tokenizer.encodeText("", maxLength: 10)
        #expect(ids.shape == [1, 1])
        #expect(mask.shape == [1, 1])
        #expect(mask.asArray(Int32.self) == [0])
    }

    @Test("AnvilTokenizer truncates to max length")
    func truncatesToMaxLength() throws {
        let tokenizer = try AnvilTokenizer.forTesting()
        let (ids, mask) = tokenizer.encodeText("abcdefghijklmnop", maxLength: 10)
        #expect(ids.shape == [1, 10])
        #expect(mask.shape == [1, 10])
    }
}
