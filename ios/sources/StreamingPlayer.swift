import Foundation
import AVFoundation

/// 24kHz 单声道流式播放：AVAudioEngine + AVAudioPlayerNode，边生成边 scheduleBuffer。
public final class StreamingPlayer: AudioSink {
    private let engine = AVAudioEngine()
    private let player = AVAudioPlayerNode()
    private var started = false
    private let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 24000, channels: 1, interleaved: false)!
    private let queue = DispatchQueue(label: "m5.player")
    private var underflows = 0
    private var scheduledFrames: AVAudioFramePosition = 0

    public init() throws {
        engine.attach(player)
        engine.connect(player, to: engine.mainMixerNode, format: format)
        engine.prepare()
        try engine.start()
        player.play()
    }

    public func write(_ samples: UnsafePointer<Float>, count: Int) {
        guard let cb = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(count)) else { return }
        cb.frameLength = AVAudioFrameCount(count)
        samples.withMemoryRebound(to: Float.self, capacity: count) { src in
            cb.floatChannelData![0].update(from: src, count: count)
        }
        queue.sync {
            scheduledFrames += AVAudioFramePosition(count)
            player.scheduleBuffer(cb)
            if !started && scheduledFrames > 2400 { started = true }   // ~100ms 起播
        }
    }

    /// 播放完缓冲后停机（合成结束时调用）
    public func drainAndStop() {
        queue.sync {
            player.stop()
            engine.stop()
        }
    }

    public func abort() {
        queue.sync {
            player.stop()
            engine.stop()
        }
    }

    /// 供性能面板：播放头落后计划帧时记一次欠载（粗粒度）
    public var underrunCount: Int { underflows }
    public func pollUnderrun() {
        queue.sync {
            guard started, player.isPlaying, let t = player.playerTime(forNodeTime: player.lastRenderTime!) else { return }
            if t.sampleTime < scheduledFrames - 24000 { underflows += 1 }
        }
    }
}
