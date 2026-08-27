import Foundation

struct XAILyricWriter: LyricWriter {
    let id = "xai"
    let displayName = "Grok (xAI)"
    let requiresAPIKey = true

    private let writer: OpenAILyricWriter

    init(apiKey: String) {
        writer = OpenAILyricWriter(
            apiKey: apiKey,
            baseURL: "https://api.x.ai/v1",
            model: "grok-3-mini-fast",
            id: "xai",
            displayName: "Grok (xAI)"
        )
    }

    func write(prompt: String, style: String, duration: TimeInterval) -> AsyncThrowingStream<String, Error> {
        writer.write(prompt: prompt, style: style, duration: duration)
    }
}
