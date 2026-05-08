import Foundation

struct OpenAILyricWriter: LyricWriter {
    let id: String
    let displayName: String
    let requiresAPIKey = true

    private let apiKey: String
    private let baseURL: String
    private let model: String

    init(
        apiKey: String,
        baseURL: String = "https://api.openai.com/v1",
        model: String = "gpt-4o-mini",
        id: String = "openai",
        displayName: String = "OpenAI"
    ) {
        self.apiKey = apiKey
        self.baseURL = baseURL
        self.model = model
        self.id = id
        self.displayName = displayName
    }

    func write(prompt: String, style: String, duration: TimeInterval) -> AsyncStream<String> {
        AsyncStream { continuation in
            let task = Task {
                let url = URL(string: "\(baseURL)/chat/completions")!
                var request = URLRequest(url: url)
                request.httpMethod = "POST"
                request.setValue("application/json", forHTTPHeaderField: "Content-Type")
                request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")

                let plan = LyricWritingPlan.make(duration: duration)

                let body: [String: Any] = [
                    "model": model,
                    "stream": true,
                    "max_tokens": plan.maxTokens,
                    "messages": [
                        ["role": "system", "content": plan.systemPrompt],
                        ["role": "user", "content": plan.userPrompt(prompt: prompt, style: style)]
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
                    guard json != "[DONE]",
                          let data = json.data(using: .utf8),
                          let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                          let choices = obj["choices"] as? [[String: Any]],
                          let delta = choices.first?["delta"] as? [String: Any],
                          let content = delta["content"] as? String
                    else { continue }
                    continuation.yield(content)
                }

                continuation.finish()
            }

            continuation.onTermination = { _ in
                task.cancel()
            }
        }
    }
}
