import Foundation

struct GeminiLyricWriter: LyricWriter {
    let id = "google"
    let displayName = "Gemini (Google)"
    let requiresAPIKey = true

    private let apiKey: String

    init(apiKey: String) {
        self.apiKey = apiKey
    }

    func write(prompt: String, style: String, duration: TimeInterval) -> AsyncStream<String> {
        AsyncStream { continuation in
            let task = Task {
                let urlString = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:streamGenerateContent?alt=sse&key=\(apiKey)"
                let url = URL(string: urlString)!
                var request = URLRequest(url: url)
                request.httpMethod = "POST"
                request.setValue("application/json", forHTTPHeaderField: "Content-Type")

                let linesPerSection = Int(duration / 15)
                let systemPrompt = """
                    You are a songwriter. Write lyrics for a song based on the user's description. \
                    Use section markers: [Verse], [Chorus], [Bridge], [Outro]. Match the mood, genre, \
                    and energy of the description. Target duration: \(Int(duration)) seconds (roughly \
                    \(linesPerSection) lines per section). Output only the lyrics with section markers. \
                    No commentary.
                    """

                let body: [String: Any] = [
                    "system_instruction": [
                        "parts": [["text": systemPrompt]]
                    ],
                    "contents": [
                        ["parts": [["text": "Style: \(style)\n\n\(prompt)"]]]
                    ]
                ]

                request.httpBody = try? JSONSerialization.data(withJSONObject: body)

                guard let (bytes, _) = try? await URLSession.shared.bytes(for: request) else {
                    continuation.finish()
                    return
                }

                for try await line in bytes.lines {
                    if Task.isCancelled { break }
                    guard line.hasPrefix("data: ") else { continue }
                    let json = String(line.dropFirst(6))
                    guard let data = json.data(using: .utf8),
                          let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                          let candidates = obj["candidates"] as? [[String: Any]],
                          let content = candidates.first?["content"] as? [String: Any],
                          let parts = content["parts"] as? [[String: Any]],
                          let text = parts.first?["text"] as? String
                    else { continue }
                    continuation.yield(text)
                }

                continuation.finish()
            }

            continuation.onTermination = { _ in
                task.cancel()
            }
        }
    }
}
