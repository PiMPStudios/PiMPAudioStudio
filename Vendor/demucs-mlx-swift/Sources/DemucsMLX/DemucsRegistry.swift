import Foundation

public struct DemucsModelDescriptor: Sendable {
    public let name: String
    public let sourceNames: [String]
    public let sampleRate: Int
    public let audioChannels: Int
    public let defaultSegmentSeconds: Double

    public init(
        name: String,
        sourceNames: [String],
        sampleRate: Int,
        audioChannels: Int,
        defaultSegmentSeconds: Double
    ) {
        self.name = name
        self.sourceNames = sourceNames
        self.sampleRate = sampleRate
        self.audioChannels = audioChannels
        self.defaultSegmentSeconds = defaultSegmentSeconds
    }
}

public enum DemucsModelRegistry {
    private static let fourSources = ["drums", "bass", "other", "vocals"]
    private static let sixSources = ["drums", "bass", "other", "vocals", "guitar", "piano"]

    public static let models: [String: DemucsModelDescriptor] = [
        "htdemucs": DemucsModelDescriptor(
            name: "htdemucs",
            sourceNames: fourSources,
            sampleRate: 44_100,
            audioChannels: 2,
            defaultSegmentSeconds: 8.0
        ),
        "htdemucs_ft": DemucsModelDescriptor(
            name: "htdemucs_ft",
            sourceNames: fourSources,
            sampleRate: 44_100,
            audioChannels: 2,
            defaultSegmentSeconds: 8.0
        ),
        "htdemucs_6s": DemucsModelDescriptor(
            name: "htdemucs_6s",
            sourceNames: sixSources,
            sampleRate: 44_100,
            audioChannels: 2,
            defaultSegmentSeconds: 8.0
        ),
        "hdemucs_mmi": DemucsModelDescriptor(
            name: "hdemucs_mmi",
            sourceNames: fourSources,
            sampleRate: 44_100,
            audioChannels: 2,
            defaultSegmentSeconds: 40.0
        ),
        "mdx": DemucsModelDescriptor(
            name: "mdx",
            sourceNames: fourSources,
            sampleRate: 44_100,
            audioChannels: 2,
            defaultSegmentSeconds: 40.0
        ),
        "mdx_extra": DemucsModelDescriptor(
            name: "mdx_extra",
            sourceNames: fourSources,
            sampleRate: 44_100,
            audioChannels: 2,
            defaultSegmentSeconds: 40.0
        ),
        "mdx_q": DemucsModelDescriptor(
            name: "mdx_q",
            sourceNames: fourSources,
            sampleRate: 44_100,
            audioChannels: 2,
            defaultSegmentSeconds: 40.0
        ),
        "mdx_extra_q": DemucsModelDescriptor(
            name: "mdx_extra_q",
            sourceNames: fourSources,
            sampleRate: 44_100,
            audioChannels: 2,
            defaultSegmentSeconds: 40.0
        ),
    ]

    public static var allModelNames: [String] {
        models.keys.sorted()
    }

    public static func descriptor(for name: String) throws -> DemucsModelDescriptor {
        guard let descriptor = models[name] else {
            throw DemucsError.unknownModel(name)
        }
        return descriptor
    }
}
