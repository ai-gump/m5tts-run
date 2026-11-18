import Foundation
#if canImport(OnnxRuntime)
import OnnxRuntime
#endif

/// OnnxRuntime 缺失时的占位实现：让工程在拿到 xcframework 之前也能编译运行并给出可诊断提示。
public final class UnavailableRunner: ORTRunner {
    public init() {}
    public var codecStateBytes: Int { 141472 * 4 }
    public func runBackbone(hidden: UnsafePointer<Float>, hiddenCount: Int, past: UnsafePointer<Float>?, pastFrames: Int,
                            positions: [Int64], outLastHidden: UnsafeMutablePointer<Float>,
                            outPresent: UnsafeMutablePointer<Float>) throws -> Int { throw M5RunnerError.frameworkMissing }
    public func runDepth(lastHidden: UnsafePointer<Float>, noise: UnsafePointer<Float>, allowEos: Bool,
                         outCodes: UnsafeMutablePointer<Int64>, outNextHidden: UnsafeMutablePointer<Float>) throws { throw M5RunnerError.frameworkMissing }
    public func runCodec(codes: UnsafePointer<Int64>, frames: Int, offset: Int, past: UnsafePointer<Float>?, pastFrames: Int,
                         convs: UnsafePointer<Float>?, outAudio: UnsafeMutablePointer<Float>,
                         outPresent: UnsafeMutablePointer<Float>, outConvs: UnsafeMutablePointer<Float>) throws -> (Int, Int) { throw M5RunnerError.frameworkMissing }
}

#if canImport(OnnxRuntime)
/// 真实 ORT 实现。张量生命周期约定：ORTValue 在 run 结束前保持引用，输出指针立刻拷贝到 Swift 数组。
public final class OnnxRuntimeRunner: ORTRunner {
    private let env: ORTEnv
    private let backbone: ORTSession, depth: ORTSession, codec: ORTSession
    private let codecStateShapes: [[NSNumber]]   // 22 个状态张量 shape（present 在前，conv_0..20 在后）

    public init(model: ModelDirectory, threads: Int = 4) throws {
        env = ORTEnv(loggingLevel: ORTLoggingLevel.warning, error: nil)!
        let opts = ORTSessionOptions()
        opts.setIntraOpNumThreads(UInt32(threads), error: nil)
        opts.setGraphOptimizationLevel(ORTGraphOptimizationLevel.all, error: nil)
        func load(_ name: String) throws -> ORTSession {
            let url = model.url.appendingPathComponent(name)
            guard let s = try? ORTSession(env: env, modelPath: url.path, sessionOptions: opts) else {
                throw M5EngineError.badModel("加载 \(name) 失败")
            }
            return s
        }
        backbone = try load("backbone.onnx")
        depth = try load("depth_frame.onnx")
        codec = try load("codec_decoder_cached.onnx")
        // 22 个 codec 状态张量：present + 21 conv（shape 来自 codec_cached.json）
        let meta = try JSONSerialization.jsonObject(with: Data(contentsOf: model.url.appendingPathComponent("codec_cached.json"))) as! [String: Any]
        let states = meta["states"] as! [[String: Any]]
        codecStateShapes = [[NSNumber(value: 8), 2, 1, 16, 0, 64]] + states.map { ($0["shape"] as! [Int]).map { NSNumber(value: $0) } }
    }

    public var codecStateBytes: Int {
        codecStateShapes.dropFirst().reduce(0) { $0 + $1.reduce(1, *) } * 4
    }

    private func tensor<T>(_ data: UnsafeMutablePointer<T>, _ shape: [NSNumber]) throws -> ORTValue {
        let bytes = shape.reduce(4) { $0 * $1.intValue }
        guard let v = ORTValue(tensorDataPointer: data, elementDataType: T.self == Float.self ? .float : .int64, shape: shape, error: nil) else {
            throw M5EngineError.badModel("构造输入张量失败")
        }
        _ = bytes
        return v
    }

    public func runBackbone(hidden: UnsafePointer<Float>, hiddenCount: Int, past: UnsafePointer<Float>?, pastFrames: Int,
                            positions: [Int64], outLastHidden: UnsafeMutablePointer<Float>,
                            outPresent: UnsafeMutablePointer<Float>) throws -> Int {
        var h = hidden; var p = past
        let seq = positions.count
        let inHidden = try tensor(&h, [1, seq, 512])
        let posCopy = positions
        let posVal = try posCopy.withUnsafeMutableBufferPointer { try tensor($0.baseAddress!, [1, seq]) }
        let pastVal: ORTValue?
        if let pp = p {
            pastVal = try tensor(UnsafeMutablePointer(mutating: pp), [8, 2, 2, pastFrames, 64])
        } else { pastVal = nil }
        let inputs: [String: ORTValue] = ["hidden": inHidden, "positions": posVal] +
            (pastVal.map { ["past": $0] } ?? [:])
        let outOpts = ORTRunOptions()
        // 输出直接映射到调用方缓冲：ORTValue(tensorDataPointer:) 同样可用于输出占位（run 写入该缓冲）
        let outHiddenVal = try tensor(outLastHidden, [1, 512])
        let presentLen = 8 * 2 * 2 * (pastFrames + seq) * 64
        let outPresentVal = try tensor(outPresent, [8, 2, 2, pastFrames + seq, 64])
        let outputs = try backbone.run(withInputs: inputs, outputNames: ["last_hidden", "present"], runOptions: outOpts)
        _ = outHiddenVal; _ = outPresentVal
        // ORT 实际输出在 outputs 字典里，拷回调用方缓冲
        try copy(outputs["last_hidden"]!, to: outLastHidden, count: 512)
        try copy(outputs["present"]!, to: outPresent, count: presentLen)
        return pastFrames + seq
    }

    public func runDepth(lastHidden: UnsafePointer<Float>, noise: UnsafePointer<Float>, allowEos: Bool,
                         outCodes: UnsafeMutablePointer<Int64>, outNextHidden: UnsafeMutablePointer<Float>) throws {
        var lh = lastHidden, nz = noise
        let hv = try tensor(&lh, [1, 512])
        let nv = try tensor(&nz, [16, 1, 2051])
        var allow: Int8 = allowEos ? 1 : 0
        let av = ORTValue(tensorDataPointer: &allow, elementDataType: .bool, shape: [1], error: nil)!
        let outputs = try depth.run(withInputs: ["hidden": hv, "noise": nv, "allow_eos": av],
                                    outputNames: ["codes", "next_hidden"], runOptions: ORTRunOptions())
        try copy(outputs["codes"]!, to: outCodes, count: 16)
        try copy(outputs["next_hidden"]!, to: outNextHidden, count: 512)
    }

    public func runCodec(codes: UnsafePointer<Int64>, frames: Int, offset: Int, past: UnsafePointer<Float>?, pastFrames: Int,
                         convs: UnsafePointer<Float>?, outAudio: UnsafeMutablePointer<Float>,
                         outPresent: UnsafeMutablePointer<Float>, outConvs: UnsafeMutablePointer<Float>) throws -> (Int, Int) {
        var c = codes
        let cv = try tensor(&c, [1, 16, frames])
        var off = Int64(offset)
        let ov = try off.withUnsafeMutableBufferPointer { try tensor($0.baseAddress!, [1]) }
        var inputs: [String: ORTValue] = ["codes": cv, "offset": ov]
        var p = past
        if let pp = p {
            inputs["past"] = try tensor(UnsafeMutablePointer(mutating: pp), [8, 2, 1, 16, pastFrames, 64])
        }
        if let cvp = convs {
            var base = UnsafeMutablePointer(mutating: cvp)
            for i in 1..<codecStateShapes.count {
                let shape = codecStateShapes[i]
                let len = shape.reduce(1, *) * 4
                inputs["conv_\(i - 1)"] = try tensor(base, shape)
                base = base.advanced(by: len / 4)
            }
        }
        let outNames = ["audio", "present"] + (0..<(codecStateShapes.count - 1)).map { "next_conv_\($0)" }
        let outputs = try codec.run(withInputs: inputs, outputNames: outNames, runOptions: ORTRunOptions())
        let audioCount = frames * 1920
        try copy(outputs["audio"]!, to: outAudio, count: audioCount)
        try copy(outputs["present"]!, to: outPresent, count: 8 * 2 * 1 * 16 * (pastFrames + frames) * 64)
        var base = outConvs
        for i in 1..<codecStateShapes.count {
            let len = codecStateShapes[i].reduce(1, *) * 4
            try copy(outputs["next_conv_\(i - 1)"]!, to: base, count: len / 4)
            base = base.advanced(by: len / 4)
        }
        return (audioCount, pastFrames + frames)
    }

    private func copy(_ value: ORTValue, to pointer: UnsafeMutablePointer<Float>, count: Int) throws {
        guard let src = value.tensorDataPointer() as? UnsafeMutablePointer<Float> else {
            throw M5EngineError.badModel("输出张量指针获取失败")
        }
        pointer.update(from: src, count: count)
    }
    private func copy(_ value: ORTValue, to pointer: UnsafeMutablePointer<Int64>, count: Int) throws {
        guard let src = value.tensorDataPointer() as? UnsafeMutablePointer<Int64> else {
            throw M5EngineError.badModel("输出张量指针获取失败")
        }
        pointer.update(from: src, count: count)
    }
}
#endif
