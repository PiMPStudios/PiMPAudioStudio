import Foundation

struct GeminiLyricWriter: LyricWriter {
    let id = "google"
    let displayName = "Gemini (Google)"
    let requiresAPIKey = true

    private let apiKey: String

    init(apiKey: String) {
        self.apiKey = apiKey
    }

    func write(prompt: String, style: String, duration: TimeInterval) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let urlString = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:streamGenerateContent?alt=sse&key=\(apiKey)"
                    let url = URL(string: urlString)!
                    var request = URLRequest(url: url)
                    request.httpMethod = "POST"
                    request.cachePolicy = .reloadIgnoringLocalAndRemoteCacheData
                    request.setValue("application/json", forHTTPHeaderField: "Content-Type")

                    let plan = LyricWritingPlan.make(duration: duration)

                    let body: [String: Any] = [
                        "system_instruction": [
                            "parts": [["text": plan.systemPrompt]]
                        ],
                        "generationConfig": [
                            "maxOutputTokens": plan.maxTokens
                        ],
                        "contents": [
                            ["parts": [["text": plan.userPrompt(prompt: prompt, style: style)]]]
                        ]
                    ]

                    request.httpBody = try LyricWriterRequest.jsonBody(body, provider: displayName)

                    let (bytes, response) = try await URLSession.shared.bytes(for: request)
                    try LyricWriterRequest.validate(response: response, provider: displayName)

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
                } catch {
                    continuation.finish(throwing: error)
                }
            }

            continuation.onTermination = { _ in
                task.cancel()
            }
        }
    }
}
