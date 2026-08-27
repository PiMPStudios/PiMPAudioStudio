import AVFoundation
import Darwin

enum AVAudioEngineDSP {
    enum EngineError: LocalizedError {
        case renderFailed(String)
        case bufferCreationFailed

        var errorDescription: String? {
            switch self {
            case let .renderFailed(message):
                message
            case .bufferCreationFailed:
                "Could not create an audio buffer for the DSP engine."
            }
        }
    }

    // MARK: - EQ

    static func applyEQ(to buffer: AudioBuffer, parameters: EQParameters, target: ApplyTarget) throws -> AudioBuffer {
        try applyEQ(to: FloatAudio(buffer), parameters: parameters, target: target).toAudioBuffer()
    }

    static func applyEQ(to audio: FloatAudio, parameters: EQParameters, target: ApplyTarget) throws -> FloatAudio {
        switch target {
        case .wholeTrack:
            try applyEQToBuffer(audio, parameters: parameters)
        case let .selection(start, end):
            try applyToRegion(audio, start: start, end: end) { region in
                try applyEQToBuffer(region, parameters: parameters)
            }
        }
    }

    static func bandwidthOctaves(forQ q: Float) -> Float {
        let safeQ = max(Double(q), 0.001)
        let bandwidth = 2.0 * asinh(1.0 / (2.0 * safeQ)) / log(2.0)
        return min(max(Float(bandwidth), 0.05), 5.0)
    }

    private static func applyEQToBuffer(_ audio: FloatAudio, parameters: EQParameters) throws -> FloatAudio {
        let pcmBuffer = try AudioBufferConverter.toPCMBuffer(audio)
        let format = pcmBuffer.format

        let engine = AVAudioEngine()
        let player = AVAudioPlayerNode()
        let eq = AVAudioUnitEQ(numberOfBands: 3)

        for (i, band) in parameters.bands.enumerated() {
            eq.bands[i].filterType = .parametric
            eq.bands[i].frequency = band.frequency
            eq.bands[i].gain = band.gain
            eq.bands[i].bandwidth = bandwidthOctaves(forQ: band.q)
            eq.bands[i].bypass = false
        }

        engine.attach(player)
        engine.attach(eq)
        engine.connect(player, to: eq, format: format)
        engine.connect(eq, to: engine.mainMixerNode, format: format)

        try engine.enableManualRenderingMode(.offline, format: format, maximumFrameCount: 4096)
        try engine.start()
        player.play()
        player.scheduleBuffer(pcmBuffer, completionHandler: nil)

        guard let outputBuffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: pcmBuffer.frameLength) else {
            throw EngineError.bufferCreationFailed
        }

        var framesRendered: AVAudioFrameCount = 0
        let totalFrames = pcmBuffer.frameLength

        while framesRendered < totalFrames {
            let framesToRender = min(4096, totalFrames - framesRendered)
            guard let chunk = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: framesToRender) else {
                throw EngineError.bufferCreationFailed
            }
            let status = try engine.renderOffline(framesToRender, to: chunk)
            if status == .error {
                throw EngineError.renderFailed("EQ render error at frame \(framesRendered)")
            }

            let destOffset = Int(framesRendered)
            let chunkFrames = Int(chunk.frameLength)
            for ch in 0 ..< Int(format.channelCount) {
                let src = chunk.floatChannelData![ch]
                let dst = outputBuffer.floatChannelData![ch]
                for i in 0 ..< chunkFrames {
                    dst[destOffset + i] = src[i]
                }
            }
            framesRendered += chunk.frameLength

            if status == .insufficientDataFromInputNode || status == .cannotDoInCurrentContext {
                break
            }
        }
        outputBuffer.frameLength = framesRendered

        engine.stop()
        return AudioBufferConverter.floatAudio(from: outputBuffer)
    }

    // MARK: - Reverb

    static func applyReverb(
        to buffer: AudioBuffer,
        parameters: ReverbParameters,
        target: ApplyTarget
    ) throws -> AudioBuffer {
        try applyReverb(to: FloatAudio(buffer), parameters: parameters, target: target).toAudioBuffer()
    }

    static func applyReverb(
        to audio: FloatAudio,
        parameters: ReverbParameters,
        target: ApplyTarget
    ) throws -> FloatAudio {
        switch target {
        case .wholeTrack:
            try applyReverbToBuffer(audio, parameters: parameters)
        case let .selection(start, end):
            try applyToRegion(audio, start: start, end: end) { region in
                try applyReverbToBuffer(region, parameters: parameters)
            }
        }
    }

    static func reverbTailFrameCount(parameters: ReverbParameters, sampleRate: Int) -> Int {
        guard parameters.wetDryMix > 0, sampleRate > 0 else { return 0 }

        let seconds = Double(max(0, parameters.decayTime)) + Double(max(0, parameters.preDelay)) / 1000.0
        return max(0, Int((seconds * Double(sampleRate)).rounded(.up)))
    }

    static func reverbPreDelayFrameCount(parameters: ReverbParameters, sampleRate: Int) -> Int {
        guard parameters.wetDryMix > 0, sampleRate > 0 else { return 0 }

        let seconds = Double(max(0, parameters.preDelay)) / 1000.0
        return max(0, Int((seconds * Double(sampleRate)).rounded(.up)))
    }

    static func reverbDampingCutoffHz(forDamping damping: Float, sampleRate: Int) -> Float {
        let clampedDamping = min(max(damping, 0), 1)
        let maxCutoff = min(max(Float(sampleRate) / 2.0 - 100.0, 2000.0), 20000.0)
        let minCutoff: Float = 2000.0
        return maxCutoff * pow(minCutoff / maxCutoff, clampedDamping)
    }

    private static func applyReverbToBuffer(_ audio: FloatAudio, parameters: ReverbParameters) throws -> FloatAudio {
        let pcmBuffer = try AudioBufferConverter.toPCMBuffer(audio)
        let format = pcmBuffer.format
        let wetDryMix = min(max(parameters.wetDryMix, 0), 1)
        guard wetDryMix > 0 else { return audio }

        let tailFrames = AVAudioFrameCount(reverbTailFrameCount(parameters: parameters, sampleRate: audio.sampleRate))
        let preDelayFrames = AVAudioFrameCount(reverbPreDelayFrameCount(
            parameters: parameters,
            sampleRate: audio.sampleRate
        ))
        let totalFrames = pcmBuffer.frameLength + tailFrames
        let wetInput = try delayedWetInputBuffer(
            source: pcmBuffer,
            preDelayFrames: preDelayFrames,
            totalFrames: totalFrames
        )
        let wetBuffer = try renderWetReverbBuffer(wetInput, parameters: parameters)
        applyDamping(to: wetBuffer, parameters: parameters)

        guard let outputBuffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: totalFrames) else {
            throw EngineError.bufferCreationFailed
        }
        outputBuffer.frameLength = totalFrames

        let dryGain = 1.0 - wetDryMix
        for channel in 0 ..< Int(format.channelCount) {
            let dry = pcmBuffer.floatChannelData![channel]
            let wet = wetBuffer.floatChannelData![channel]
            let output = outputBuffer.floatChannelData![channel]
            for frame in 0 ..< Int(totalFrames) {
                let drySample = frame < Int(pcmBuffer.frameLength) ? dry[frame] : 0
                let wetSample = frame < Int(wetBuffer.frameLength) ? wet[frame] : 0
                output[frame] = drySample * dryGain + wetSample * wetDryMix
            }
        }

        return AudioBufferConverter.floatAudio(from: outputBuffer)
    }

    private static func renderWetReverbBuffer(
        _ pcmBuffer: AVAudioPCMBuffer,
        parameters: ReverbParameters
    ) throws -> AVAudioPCMBuffer {
        let renderInput = try reverbCompatibleInputBuffer(pcmBuffer)
        let format = renderInput.format
        let engine = AVAudioEngine()
        let player = AVAudioPlayerNode()
        let reverb = AVAudioUnitReverb()

        let preset: AVAudioUnitReverbPreset = if parameters.decayTime <= 1.0 {
            .mediumRoom
        } else if parameters.decayTime <= 2.0 {
            .largeChamber
        } else {
            .largeHall
        }
        reverb.loadFactoryPreset(preset)
        reverb.wetDryMix = 100

        engine.attach(player)
        engine.attach(reverb)
        engine.connect(player, to: reverb, format: format)
        engine.connect(reverb, to: engine.mainMixerNode, format: format)

        try engine.enableManualRenderingMode(.offline, format: format, maximumFrameCount: 4096)
        try engine.start()
        player.scheduleBuffer(renderInput, completionHandler: nil)
        player.play()

        guard let outputBuffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: renderInput.frameLength) else {
            throw EngineError.bufferCreationFailed
        }

        var framesRendered: AVAudioFrameCount = 0
        while framesRendered < renderInput.frameLength {
            let framesToRender = min(4096, renderInput.frameLength - framesRendered)
            guard let chunk = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: framesToRender) else {
                throw EngineError.bufferCreationFailed
            }
            let status = try engine.renderOffline(framesToRender, to: chunk)
            if status == .error {
                throw EngineError.renderFailed("Reverb render error at frame \(framesRendered)")
            }

            let destOffset = Int(framesRendered)
            let chunkFrames = Int(chunk.frameLength)
            for ch in 0 ..< Int(format.channelCount) {
                let src = chunk.floatChannelData![ch]
                let dst = outputBuffer.floatChannelData![ch]
                for i in 0 ..< chunkFrames {
                    dst[destOffset + i] = src[i]
                }
            }
            framesRendered += chunk.frameLength

            if status == .insufficientDataFromInputNode || status == .cannotDoInCurrentContext {
                break
            }
        }
        outputBuffer.frameLength = framesRendered

        engine.stop()

        return try restoreOriginalChannelLayout(outputBuffer, originalFormat: pcmBuffer.format)
    }

    private static func delayedWetInputBuffer(
        source: AVAudioPCMBuffer,
        preDelayFrames: AVAudioFrameCount,
        totalFrames: AVAudioFrameCount
    ) throws -> AVAudioPCMBuffer {
        guard let buffer = AVAudioPCMBuffer(pcmFormat: source.format, frameCapacity: totalFrames) else {
            throw EngineError.bufferCreationFailed
        }
        buffer.frameLength = totalFrames
        for channel in 0 ..< Int(source.format.channelCount) {
            let data = buffer.floatChannelData![channel]
            for frame in 0 ..< Int(totalFrames) {
                data[frame] = 0
            }
            let sourceData = source.floatChannelData![channel]
            let offset = Int(preDelayFrames)
            for frame in 0 ..< Int(source.frameLength) where offset + frame < Int(totalFrames) {
                data[offset + frame] = sourceData[frame]
            }
        }
        return buffer
    }

    private static func reverbCompatibleInputBuffer(_ source: AVAudioPCMBuffer) throws -> AVAudioPCMBuffer {
        guard source.format.channelCount == 1 else { return source }
        guard let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: source.format.sampleRate,
            channels: 2,
            interleaved: false
        ),
            let stereo = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: source.frameLength) else {
            throw EngineError.bufferCreationFailed
        }

        stereo.frameLength = source.frameLength
        let mono = source.floatChannelData![0]
        for channel in 0 ..< 2 {
            let target = stereo.floatChannelData![channel]
            for frame in 0 ..< Int(source.frameLength) {
                target[frame] = mono[frame]
            }
        }
        return stereo
    }

    private static func restoreOriginalChannelLayout(
        _ rendered: AVAudioPCMBuffer,
        originalFormat: AVAudioFormat
    ) throws -> AVAudioPCMBuffer {
        guard originalFormat.channelCount == 1, rendered.format.channelCount == 2 else { return rendered }
        guard let mono = AVAudioPCMBuffer(pcmFormat: originalFormat, frameCapacity: rendered.frameLength) else {
            throw EngineError.bufferCreationFailed
        }

        mono.frameLength = rendered.frameLength
        let left = rendered.floatChannelData![0]
        let right = rendered.floatChannelData![1]
        let output = mono.floatChannelData![0]
        for frame in 0 ..< Int(rendered.frameLength) {
            output[frame] = (left[frame] + right[frame]) * 0.5
        }
        return mono
    }

    private static func applyDamping(to buffer: AVAudioPCMBuffer, parameters: ReverbParameters) {
        guard parameters.damping > 0.001 else { return }

        let cutoff = reverbDampingCutoffHz(
            forDamping: parameters.damping,
            sampleRate: Int(buffer.format.sampleRate)
        )
        let sampleRate = Float(buffer.format.sampleRate)
        let rc = 1.0 / (2.0 * Float.pi * cutoff)
        let dt = 1.0 / sampleRate
        let alpha = dt / (rc + dt)

        for channel in 0 ..< Int(buffer.format.channelCount) {
            let data = buffer.floatChannelData![channel]
            var last: Float = 0
            for frame in 0 ..< Int(buffer.frameLength) {
                last += alpha * (data[frame] - last)
                data[frame] = last
            }
        }
    }

    // MARK: - Region Splicing

    private static func applyToRegion(
        _ audio: FloatAudio,
        start: TimeInterval,
        end: TimeInterval,
        transform: (FloatAudio) throws -> FloatAudio
    ) throws -> FloatAudio {
        let startSample = min(
            SimpleDSP.sampleIndex(time: start, sampleRate: audio.sampleRate, channels: audio.channels),
            audio.samples.count
        )
        let endSample = min(
            max(
                SimpleDSP.sampleIndex(time: end, sampleRate: audio.sampleRate, channels: audio.channels),
                startSample
            ),
            audio.samples.count
        )

        let region = audio.replacingSamples(Array(audio.samples[startSample ..< endSample]))
        let processed = try transform(region)

        var result = [Float]()
        result.reserveCapacity(startSample + processed.samples.count + (audio.samples.count - endSample))
        result.append(contentsOf: audio.samples[0 ..< startSample])
        result.append(contentsOf: processed.samples)
        if endSample < audio.samples.count {
            result.append(contentsOf: audio.samples[endSample...])
        }

        return audio.replacingSamples(result)
    }
}
