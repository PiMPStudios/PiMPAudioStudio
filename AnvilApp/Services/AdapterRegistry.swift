import Foundation
import AnvilCore
import MLX

struct AdapterRegistry {
    static var adaptersDirectory: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appending(path: "Anvil/models/adapters")
    }

    static func listAdapters() -> [LoRAAdapter] {
        let dir = adaptersDirectory
        let fm = FileManager.default

        guard let contents = try? fm.contentsOfDirectory(
            at: dir,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ) else {
            return []
        }

        var adapters: [LoRAAdapter] = []
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601

        for itemURL in contents {
            let isDir = (try? itemURL.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory ?? false
            guard isDir else { continue }

            let metadataURL = itemURL.appending(path: "adapter.json")
            let weightsURL = itemURL.appending(path: "adapter.safetensors")

            guard fm.fileExists(atPath: metadataURL.path(percentEncoded: false)),
                  fm.fileExists(atPath: weightsURL.path(percentEncoded: false)) else {
                continue
            }

            guard let data = try? Data(contentsOf: metadataURL),
                  let adapter = try? decoder.decode(LoRAAdapter.self, from: data) else {
                continue
            }

            adapters.append(adapter)
        }

        return adapters.sorted { $0.createdAt > $1.createdAt }
    }

    static func compatibleAdapters(for category: ModelCategory) -> [LoRAAdapter] {
        listAdapters().filter { $0.baseModelCategory == category }
    }

    static func adapterDirectory(id: String) -> URL {
        adaptersDirectory.appending(path: id)
    }

    static func save(adapter: LoRAAdapter, weights: [(String, MLXArray)]) throws {
        let dir = adapterDirectory(id: adapter.id)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

        // Save metadata as JSON
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(adapter)
        try data.write(to: dir.appending(path: "adapter.json"))

        // Save weights as safetensors
        let dict = Dictionary(weights, uniquingKeysWith: { _, last in last })
        try MLX.save(arrays: dict, url: dir.appending(path: "adapter.safetensors"))
    }

    static func loadWeights(adapterID: String) throws -> [(String, MLXArray)] {
        let url = adapterDirectory(id: adapterID).appending(path: "adapter.safetensors")
        let dict = try MLX.loadArrays(url: url)
        return dict.map { ($0.key, $0.value) }
    }
}
