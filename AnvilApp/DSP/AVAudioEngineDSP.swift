import AVFoundation

struct AVAudioEngineDSP {
    enum EngineError: Error {
        case renderFailed(String)
        case bufferCreationFailed
    }

    // MARK: - EQ

    static func applyEQ(to buffer: AudioBuffer, parameters: EQParameters, target: ApplyTarget) throws -> AudioBuffer {
        switch target {
        case .wholeTrack:
            return try applyEQToBuffer(buffer, parameters: parameters)
        case .selection(let start, let end):
            return try applyToRegion(buffer, start: start, end: end) { region in
                try applyEQToBuffer(region, parameters: parameters)
            }
        }
    }

    private static func applyEQToBuffer(_ buffer: AudioBuffer, parameters: EQParameters) throws -> AudioBuffer {
        let pcmBuffer = try AudioBufferConverter.toPCMBuffer(buffer)
        let format = pcmBuffer.format

        let engine = AVAudioEngine()
        let player = AVAudioPlayerNode()
        let eq = AVAudioUnitEQ(numberOfBands: 3)

        for (i, band) in parameters.bands.enumerated() {
            eq.bands[i].filterType = .parametric
            eq.bands[i].frequency = band.frequency
            eq.bands[i].gain = band.gain
            eq.bands[i].bandwidth = band.q
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
            for ch in 0..<Int(format.channelCount) {
                let src = chunk.floatChannelData![ch]
                let dst = outputBuffer.floatChannelData![ch]
                for i in 0..<chunkFrames {
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
        return AudioBufferConverter.fromPCMBuffer(outputBuffer)
    }

    // MARK: - Reverb

    static func applyReverb(to buffer: AudioBuffer, parameters: ReverbParameters, target: ApplyTarget) throws -> AudioBuffer {
        switch target {
        case .wholeTrack:
            return try applyReverbToBuffer(buffer, parameters: parameters)
        case .selection(let start, let end):
            return try applyToRegion(buffer, start: start, end: end) { region in
                try applyReverbToBuffer(region, parameters: parameters)
            }
        }
    }

    private static func applyReverbToBuffer(_ buffer: AudioBuffer, parameters: ReverbParameters) throws -> AudioBuffer {
        let pcmBuffer = try AudioBufferConverter.toPCMBuffer(buffer)
        let format = pcmBuffer.format

        let engine = AVAudioEngine()
        let player = AVAudioPlayerNode()
        let reverb = AVAudioUnitReverb()

        let preset: AVAudioUnitReverbPreset
        if parameters.decayTime <= 1.0 {
            preset = .mediumRoom
        } else if parameters.decayTime <= 2.0 {
            preset = .largeChamber
        } else {
            preset = .largeHall
        }
        reverb.loadFactoryPreset(preset)
        reverb.wetDryMix = parameters.wetDryMix * 100.0

        engine.attach(player)
        engine.attach(reverb)
        engine.connect(player, to: reverb, format: format)
        engine.connect(reverb, to: engine.mainMixerNode, format: format)

        let tailFrames = AVAudioFrameCount(parameters.decayTime * Float(buffer.sampleRate))
        let totalFrames = pcmBuffer.frameLength + tailFrames

        try engine.enableManualRenderingMode(.offline, format: format, maximumFrameCount: 4096)
        try engine.start()
        player.play()
        player.scheduleBuffer(pcmBuffer, completionHandler: nil)

        guard let outputBuffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: totalFrames) else {
            throw EngineError.bufferCreationFailed
        }

        var framesRendered: AVAudioFrameCount = 0
        while framesRendered < totalFrames {
            let framesToRender = min(4096, totalFrames - framesRendered)
            guard let chunk = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: framesToRender) else {
                throw EngineError.bufferCreationFailed
            }
            let status = try engine.renderOffline(framesToRender, to: chunk)
            if status == .error {
                throw EngineError.renderFailed("Reverb render error at frame \(framesRendered)")
            }

            let destOffset = Int(framesRendered)
            let chunkFrames = Int(chunk.frameLength)
            for ch in 0..<Int(format.channelCount) {
                let src = chunk.floatChannelData![ch]
                let dst = outputBuffer.floatChannelData![ch]
                for i in 0..<chunkFrames {
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

        // Trim to original length (don't extend the track)
        let originalFrames = pcmBuffer.frameLength
        let finalFrames = min(framesRendered, originalFrames)
        guard let trimmedOutput = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: finalFrames) else {
            throw EngineError.bufferCreationFailed
        }
        trimmedOutput.frameLength = finalFrames
        for ch in 0..<Int(format.channelCount) {
            let src = outputBuffer.floatChannelData![ch]
            let dst = trimmedOutput.floatChannelData![ch]
            for i in 0..<Int(finalFrames) {
                dst[i] = src[i]
            }
        }

        return AudioBufferConverter.fromPCMBuffer(trimmedOutput)
    }

    // MARK: - Time Stretch

    static func applyTimeStretch(to buffer: AudioBuffer, parameters: TimeStretchParameters, target: ApplyTarget) throws -> AudioBuffer {
        switch target {
        case .wholeTrack:
            return try applyTimeStretchToBuffer(buffer, parameters: parameters)
        case .selection(let start, let end):
            return try applyToRegion(buffer, start: start, end: end) { region in
                try applyTimeStretchToBuffer(region, parameters: parameters)
            }
        }
    }

    private static func applyTimeStretchToBuffer(_ buffer: AudioBuffer, parameters: TimeStretchParameters) throws -> AudioBuffer {
        let pcmBuffer = try AudioBufferConverter.toPCMBuffer(buffer)
        let format = pcmBuffer.format

        let engine = AVAudioEngine()
        let player = AVAudioPlayerNode()
        let timePitch = AVAudioUnitTimePitch()

        timePitch.rate = parameters.rate
        timePitch.pitch = 0

        engine.attach(player)
        engine.attach(timePitch)
        engine.connect(player, to: timePitch, format: format)
        engine.connect(timePitch, to: engine.mainMixerNode, format: format)

        let inputFrames = pcmBuffer.frameLength
        let outputFrameCount = AVAudioFrameCount(Double(inputFrames) / Double(parameters.rate))

        try engine.enableManualRenderingMode(.offline, format: format, maximumFrameCount: 4096)
        try engine.start()
        player.play()
        player.scheduleBuffer(pcmBuffer, completionHandler: nil)

        guard let outputBuffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: outputFrameCount) else {
            throw EngineError.bufferCreationFailed
        }

        var framesRendered: AVAudioFrameCount = 0
        while framesRendered < outputFrameCount {
            let framesToRender = min(4096, outputFrameCount - framesRendered)
            guard let chunk = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: framesToRender) else {
                throw EngineError.bufferCreationFailed
            }
            let status = try engine.renderOffline(framesToRender, to: chunk)
            if status == .error {
                throw EngineError.renderFailed("Time stretch render error at frame \(framesRendered)")
            }

            let destOffset = Int(framesRendered)
            let chunkFrames = Int(chunk.frameLength)
            for ch in 0..<Int(format.channelCount) {
                let src = chunk.floatChannelData![ch]
                let dst = outputBuffer.floatChannelData![ch]
                for i in 0..<chunkFrames {
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
        return AudioBufferConverter.fromPCMBuffer(outputBuffer)
    }

    // MARK: - Pitch Shift

    static func applyPitchShift(to buffer: AudioBuffer, parameters: PitchShiftParameters, target: ApplyTarget) throws -> AudioBuffer {
        switch target {
        case .wholeTrack:
            return try applyPitchShiftToBuffer(buffer, parameters: parameters)
        case .selection(let start, let end):
            return try applyToRegion(buffer, start: start, end: end) { region in
                try applyPitchShiftToBuffer(region, parameters: parameters)
            }
        }
    }

    private static func applyPitchShiftToBuffer(_ buffer: AudioBuffer, parameters: PitchShiftParameters) throws -> AudioBuffer {
        let pcmBuffer = try AudioBufferConverter.toPCMBuffer(buffer)
        let format = pcmBuffer.format

        let engine = AVAudioEngine()
        let player = AVAudioPlayerNode()
        let timePitch = AVAudioUnitTimePitch()

        timePitch.rate = 1.0
        timePitch.pitch = parameters.pitch

        engine.attach(player)
        engine.attach(timePitch)
        engine.connect(player, to: timePitch, format: format)
        engine.connect(timePitch, to: engine.mainMixerNode, format: format)

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
                throw EngineError.renderFailed("Pitch shift render error at frame \(framesRendered)")
            }

            let destOffset = Int(framesRendered)
            let chunkFrames = Int(chunk.frameLength)
            for ch in 0..<Int(format.channelCount) {
                let src = chunk.floatChannelData![ch]
                let dst = outputBuffer.floatChannelData![ch]
                for i in 0..<chunkFrames {
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
        return AudioBufferConverter.fromPCMBuffer(outputBuffer)
    }

    // MARK: - Region Splicing

    private static func applyToRegion(
        _ buffer: AudioBuffer,
        start: TimeInterval,
        end: TimeInterval,
        transform: (AudioBuffer) throws -> AudioBuffer
    ) throws -> AudioBuffer {
        let bytesPerSample = MemoryLayout<Int16>.size
        let startSample = SimpleDSP.sampleIndex(time: start, sampleRate: buffer.sampleRate, channels: buffer.channels)
        let endSample = min(
            SimpleDSP.sampleIndex(time: end, sampleRate: buffer.sampleRate, channels: buffer.channels),
            buffer.data.count / bytesPerSample
        )

        let startByte = startSample * bytesPerSample
        let endByte = endSample * bytesPerSample

        let regionData = Data(buffer.data[startByte..<endByte])
        let regionBuffer = AudioBuffer(data: regionData, sampleRate: buffer.sampleRate, channels: buffer.channels)

        let processed = try transform(regionBuffer)

        var result = Data()
        result.append(buffer.data[0..<startByte])
        result.append(processed.data)
        if endByte < buffer.data.count {
            result.append(buffer.data[endByte...])
        }

        return AudioBuffer(data: result, sampleRate: buffer.sampleRate, channels: buffer.channels)
    }
}
