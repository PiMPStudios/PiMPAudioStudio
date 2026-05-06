// AnvilApp/Models/PipelineChoice.swift
import Foundation

enum PipelineChoice: String, CaseIterable, Sendable, Identifiable {
    case auto
    case aceStep
    case stableAudio

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .auto: return "Auto"
        case .aceStep: return "ACE-Step (Music)"
        case .stableAudio: return "Stable Audio (SFX)"
        }
    }
}
