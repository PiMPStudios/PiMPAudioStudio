import AVFoundation
import OSLog
import SwiftUI

struct WaveformData: Sendable {
    let peaks: [Float]
    let sampleRate: Int
    let channelCount: Int
}

@MainActor
final class AudioPlayerService: ObservableObject {
    private let logger = Logger(subsystem: "com.pimp.AnvilApp", category: "AudioPlayer")

    @Published var isPlaying: Bool = false
    @Published var currentTime: TimeInterval = 0
    @Published var duration: TimeInterval = 0
    @Published var volume: Float = 0.7
    @Published var waveform: WaveformData?

    private(set) var loadedURL: URL?
    private var player: AVAudioPlayer?
    private var timer: Timer?

    func load(url: URL, precomputedPeaks: [Float]? = nil) {
        if url == loadedURL {
            if let peaks = precomputedPeaks, !peaks.isEmpty {
                waveform = WaveformData(peaks: peaks, sampleRate: 44100, channelCount: 2)
            } else if waveform == nil {
                waveform = computeWaveform(url: url)
            }

            if player != nil, duration > 0 {
                return
            }
        }

        stop()
        player = nil
        waveform = nil
        duration = 0
        loadedURL = url
        do {
            player = try AVAudioPlayer(contentsOf: url)
            player?.prepareToPlay()
            player?.volume = volume
            duration = player?.duration ?? 0
            currentTime = 0
            if let peaks = precomputedPeaks, !peaks.isEmpty {
                // Use peaks computed at generation time — avoids re-reading the file.
                // sampleRate/channelCount are metadata only; WaveformView only uses peaks.
                waveform = WaveformData(peaks: peaks, sampleRate: 44100, channelCount: 2)
            } else {
                waveform = computeWaveform(url: url)
            }
        } catch {
            loadedURL = nil
            player = nil
            waveform = nil
            duration = 0
            currentTime = 0
            logger.error("Failed to load audio: \(error, privacy: .public)")
        }
    }

    func play() {
        guard let player else { return }
        player.play()
        isPlaying = true
        startTimer()
    }

    func pause() {
        player?.pause()
        isPlaying = false
        stopTimer()
    }

    func togglePlayback() {
        if isPlaying { pause() } else { play() }
    }

    func seek(to time: TimeInterval) {
        player?.currentTime = time
        currentTime = time
    }

    func setVolume(_ newVolume: Float) {
        volume = newVolume
        player?.volume = newVolume
    }

    func stop() {
        player?.stop()
        isPlaying = false
        currentTime = 0
        stopTimer()
    }

    private func startTimer() {
        timer = Timer.scheduledTimer(withTimeInterval: 1.0 / 30.0, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in
                guard let self, let player = self.player else { return }
                self.currentTime = player.currentTime
                if !player.isPlaying {
                    self.isPlaying = false
                    self.stopTimer()
                }
            }
        }
    }

    private func stopTimer() {
        timer?.invalidate()
        timer = nil
    }

    private func computeWaveform(url: URL, bucketCount: Int = 400) -> WaveformData? {
        guard let file = try? AVAudioFile(forReading: url) else { return nil }
        let format = file.processingFormat
        let frameCount = AVAudioFrameCount(file.length)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
            return nil
        }
        do { try file.read(into: buffer) } catch { return nil }

        guard let channelData = buffer.floatChannelData else { return nil }
        let channelCount = Int(format.channelCount)
        let totalFrames = Int(buffer.frameLength)
        let framesPerBucket = max(totalFrames / bucketCount, 1)

        var peaks = [Float](repeating: 0, count: bucketCount)
        for bucket in 0..<bucketCount {
            let start = bucket * framesPerBucket
            let end = min(start + framesPerBucket, totalFrames)
            var maxVal: Float = 0
            for frame in start..<end {
                for ch in 0..<channelCount {
                    let val = abs(channelData[ch][frame])
                    if val > maxVal { maxVal = val }
                }
            }
            peaks[bucket] = maxVal
        }

        let maxPeak = peaks.max() ?? 1.0
        if maxPeak > 0 {
            peaks = peaks.map { $0 / maxPeak }
        }

        return WaveformData(
            peaks: peaks,
            sampleRate: Int(format.sampleRate),
            channelCount: channelCount
        )
    }
}
