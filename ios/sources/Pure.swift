import Foundation

// MARK: - 音色（voices.json）

public struct M5Voice: Codable {
    public let name: String
    public let referenceText: String
    let codes: [[Int]]
    enum CodingKeys: String, CodingKey { case name, referenceText = "reference_text", codes }
    public var frames: Int { codes.count }
    public func code(_ frame: Int, _ book: Int) -> Int { codes[frame][book] }
}

public struct VoicesDocument: Codable {
    public let voices: [M5Voice]
}

// MARK: - 条件文本（六语言）与分段（与 Android M5Engine 等价）

public enum M5Language: String, CaseIterable, Identifiable {
    case chinese = "Chinese", english = "English", german = "German", french = "French", japanese = "Japanese", korean = "Korean"
    public var id: String { rawValue }
    public var display: String {
        switch self {
        case .chinese: "中文 Chinese"; case .english: "English"; case .german: "Deutsch"
        case .french: "Français"; case .japanese: "日本語"; case .korean: "한국어"
        }
    }
}

public enum M5EngineError: Error, LocalizedError {
    case reservedToken, unsupportedLanguage(String), textTooLong(Int), emptyText, badModel(String)
    public var errorDescription: String? {
        switch self {
        case .reservedToken: "文本包含保留的控制标记"
        case .unsupportedLanguage(let l): "不支持的语言：\(l)"
        case .textTooLong(let n): "一次最多输入 2000 字（当前 \(n)）"
        case .emptyText: "请输入要合成的文字"
        case .badModel(let m): m
        }
    }
}

public func conditioned(reference: String, text: String, language: M5Language) throws -> String {
    let combined = reference.trimmingCharacters(in: .whitespacesAndNewlines) + text.trimmingCharacters(in: .whitespacesAndNewlines)
    for token in ["<|caption_begin|>", "<|caption_end|>", "<|accent_begin|>", "<|accent_end|>",
                  "<|speaker_start|>", "<|speaker_end|>", "<|lang_begin|>", "<|lang_end|>"] where combined.contains(token) {
        throw M5EngineError.reservedToken
    }
    return "<|caption_begin|><|caption_end|><|accent_begin|><|accent_end|>" +
        "<|speaker_start|>None<|speaker_end|><|lang_begin|>\(language.rawValue)<|lang_end|>" + combined
}

public func splitText(_ text: String) throws -> [String] {
    let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty else { throw M5EngineError.emptyText }
    guard trimmed.count <= 2000 else { throw M5EngineError.textTooLong(trimmed.count) }
    var result: [String] = []; var part = ""; var count = 0
    for ch in trimmed {
        part.append(ch); count += 1
        if count >= 80 || "。！？!?；;\n".contains(ch) {
            let p = part.trimmingCharacters(in: .whitespacesAndNewlines)
            if !p.isEmpty { result.append(p) }
            part = ""; count = 0
        }
    }
    let p = part.trimmingCharacters(in: .whitespacesAndNewlines)
    if !p.isEmpty { result.append(p) }
    guard !result.isEmpty else { throw M5EngineError.emptyText }
    return result
}

// MARK: - 模型目录

public struct ModelDirectory {
    public let url: URL
    public let precision: String
    public let voices: [M5Voice]
    public static let requiredFiles = [
        "bundle.json", "manifest.json", "codec_cached.json", "tokenizer.bin", "voices.json",
        "backbone.onnx", "depth_frame.onnx", "codec_decoder_cached.onnx",
        "text_embeddings.npy", "audio_embeddings.npy",
    ]

    public init(directory: URL) throws {
        var missing: [String] = []
        for f in Self.requiredFiles where !FileManager.default.isReadableFile(atPath: directory.appendingPathComponent(f).path) {
            missing.append(f)
        }
        guard missing.isEmpty else { throw M5EngineError.badModel("模型目录缺文件：\(missing.joined(separator: "、"))") }
        let manifest = try JSONSerialization.jsonObject(with: Data(contentsOf: directory.appendingPathComponent("manifest.json"))) as? [String: Any]
        precision = (manifest?["precision"] as? String) ?? "fp32"
        let voicesData = try Data(contentsOf: directory.appendingPathComponent("voices.json"))
        voices = try JSONDecoder().decode(VoicesDocument.self, from: voicesData).voices
        guard !voices.isEmpty else { throw M5EngineError.badModel("模型包没有音色") }
        url = directory
    }

    /// App 的 Documents/m5-model（用户通过"文件"App 或访达拷入）
    public static func defaultDirectory() -> URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return docs.appendingPathComponent("m5-model", isDirectory: true)
    }
}

// MARK: - NPY 矩阵（mmap 只读，FP32 C-order）

public final class NpyMatrix {
    public let rows: Int, columns: Int
    private let data: Data
    private let base: UnsafePointer<Float>

    public init(url: URL, expectRows: Int? = nil, expectColumns: Int? = nil) throws {
        data = try Data(contentsOf: url, options: .mappedIfSafe)
        var head = data.startIndex
        let magic: [UInt8] = [147, 78, 85, 77, 80, 89]
        guard data.count > 100, Array(data[head..<data.index(head, offsetBy: 6)]) == magic else {
            throw M5EngineError.badModel("无效 embedding 文件")
        }
        head = data.index(head, offsetBy: 6)
        let version = data[head]; head = data.index(head, offsetBy: 1); head = data.index(head, offsetBy: 1)
        var headerLength = 0
        if version == 1 {
            headerLength = Int(data[head]) | Int(data[data.index(head, offsetBy: 1)]) << 8
            head = data.index(head, offsetBy: 2)
        } else if version == 2 {
            headerLength = Int(data[head]) | Int(data[data.index(head, offsetBy: 1)]) << 8 | Int(data[data.index(head, offsetBy: 2)]) << 16 | Int(data[data.index(head, offsetBy: 3)]) << 24
            head = data.index(head, offsetBy: 4)
        } else { throw M5EngineError.badModel("不支持的 NPY 版本") }
        let headerEnd = data.index(head, offsetBy: headerLength)
        guard let header = String(data: data[head..<headerEnd], encoding: .ascii),
              header.contains("'<f4'"), header.contains("'fortran_order': False"),
              let m = header.range(of: #"'shape': \((\d+), (\d+)\)"#, options: .regularExpression) else {
            throw M5EngineError.badModel("需要 C-order FP32 二维 embedding")
        }
        let shape = header[m].replacingOccurrences(of: "'shape': (", with: "").replacingOccurrences(of: ")", with: "")
        let dims = shape.split(separator: ",").map { Int($0.trimmingCharacters(in: .whitespaces))! }
        rows = dims[0]; columns = dims[1]
        if let er = expectRows, er != rows { throw M5EngineError.badModel("embedding 行数 \(rows) ≠ \(er)") }
        if let ec = expectColumns, ec != columns { throw M5EngineError.badModel("embedding 列数 \(columns) ≠ \(ec)") }
        let bodyLength = rows * columns * 4
        guard data.distance(from: headerEnd, to: data.endIndex) == bodyLength else { throw M5EngineError.badModel("Embedding 尺寸错误") }
        base = data.withUnsafeBytes { $0.baseAddress!.assumingMemoryBound(to: Float.self) }
    }

    public func copyRow(_ row: Int, into out: UnsafeMutablePointer<Float>) {
        precondition(row >= 0 && row < rows)
        let p = base + row * columns
        for i in 0..<columns { out[i] = p[i] }
    }
    public func addRow(_ row: Int, into out: UnsafeMutablePointer<Float>) {
        precondition(row >= 0 && row < rows)
        let p = base + row * columns
        for i in 0..<columns { out[i] += p[i] }
    }
}

// MARK: - WAV 写出（24kHz mono PCM16）

public final class WavWriter {
    private let handle: FileHandle
    private var samples = 0
    public init(url: URL) throws {
        FileManager.default.createFile(atPath: url.path, contents: nil)
        handle = try FileHandle(forWritingTo: url)
        handle.write(Data(repeating: 0, count: 44))
    }
    /// pcm: 归一化浮点
    public func append(_ pcm: UnsafePointer<Float>, count: Int) throws {
        var bytes = [UInt8](); bytes.reserveCapacity(count * 2)
        for i in 0..<count {
            let v = pcm[i]
            guard v.isFinite else { throw M5EngineError.badModel("生成音频出现无效数值") }
            let s = Int16(max(-32768, min(32767, Int(v * 32768.0))))
            bytes.append(UInt8(truncatingIfNeeded: s))
            bytes.append(UInt8(truncatingIfNeeded: s >> 8))
        }
        handle.write(Data(bytes))
        samples += count
    }
    public var duration: Double { Double(samples) / 24000.0 }
    public func close() throws {
        defer { try? handle.close() }
        let dataSize = UInt32(samples * 2)
        var head = [UInt8]()
        head.append(contentsOf: Array("RIFF".utf8))
        appendLE(&head, UInt32(36 + dataSize))
        head.append(contentsOf: Array("WAVEfmt ".utf8))
        appendLE(&head, UInt32(16)); appendLE(&head, UInt16(1)); appendLE(&head, UInt16(1))
        appendLE(&head, UInt32(24000)); appendLE(&head, UInt32(48000))
        appendLE(&head, UInt16(2)); appendLE(&head, UInt16(16))
        head.append(contentsOf: Array("data".utf8)); appendLE(&head, dataSize)
        try handle.seek(toOffset: 0)
        handle.write(Data(head))
    }
    private func appendLE<T: FixedWidthInteger>(_ a: inout [UInt8], _ v: T) {
        for i in 0..<MemoryLayout<T>.size { a.append(UInt8(truncatingIfNeeded: v >> (8 * i))) }
    }
}
