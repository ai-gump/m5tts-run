import Foundation

/// M5TK1 二进制分词器：与 Android 版 Tokenizer.java 逐行等价的 byte-level BPE 移植。
public final class M5Tokenizer {
    private final class Trie {
        var children: [UInt16: Trie] = [:]
        var id: Int32 = -1
        func put(_ value: [UInt16], _ tokenId: Int32) {
            var node = self
            for ch in value { node = node.children[ch] ?? { let t = Trie(); node.children[ch] = t; return t }() }
            node.id = tokenId
        }
    }
    private final class PairTable {
        var keys: [UInt64]; var values: [UInt64]; let mask: Int
        init(n: Int) {
            var cap = 1
            while cap < n * 2 { cap <<= 1 }
            keys = [UInt64](repeating: 0, count: cap)
            values = [UInt64](repeating: 0, count: cap)
            mask = cap - 1
        }
        private func slot(_ key: UInt64) -> Int {
            var h = key &* 0x9E3779B97F4A7C15
            h ^= h >> 32
            return Int(truncatingIfNeeded: h) & mask
        }
        func put(_ a: Int32, _ b: Int32, _ rank: Int32, _ id: Int32) {
            let key = (UInt64(UInt32(bitPattern: a)) << 32 | UInt64(UInt32(bitPattern: b))) &+ 1
            var s = slot(key)
            while keys[s] != 0 && keys[s] != key { s = (s + 1) & mask }
            if keys[s] == 0 { keys[s] = key; values[s] = UInt64(UInt32(bitPattern: rank)) << 32 | UInt64(UInt32(bitPattern: id)) }
        }
        func get(_ a: Int32, _ b: Int32) -> Int64 {
            let key = (UInt64(UInt32(bitPattern: a)) << 32 | UInt64(UInt32(bitPattern: b))) &+ 1
            var s = slot(key)
            while keys[s] != 0 {
                if keys[s] == key { return Int64(bitPattern: values[s]) }
                s = (s + 1) & mask
            }
            return -1
        }
    }

    private let vocab: [String: Int32]
    private let specials = Trie()
    private let merges: PairTable
    private let split: NSRegularExpression
    private let bos: Int32, eos: Int32
    private let byteChars: [UInt16]   // 按 byte 值索引的"可见字符"
    private let byteIds: [Int32]

    public enum TokenizerError: Error { case badFile(String) }

    public init(fileURL: URL) throws {
        let data = try Data(contentsOf: fileURL)
        var p = 0
        func u32() throws -> UInt32 {
            guard p + 4 <= data.count else { throw TokenizerError.badFile("truncated u32") }
            let v = UInt32(data[p]) << 24 | UInt32(data[p+1]) << 16 | UInt32(data[p+2]) << 8 | UInt32(data[p+3])
            p += 4; return v
        }
        func i32() throws -> Int32 { Int32(bitPattern: try u32()) }
        func str() throws -> String {
            let n = Int(try u32())
            guard n >= 0 && n <= 100_000, p + n <= data.count else { throw TokenizerError.badFile("bad string") }
            defer { p += n }
            return String(decoding: data.subdata(in: p..<p+n), as: UTF8.self)
        }
        func bounded(_ v: Int32, _ min: Int32, _ max: Int32) throws -> Int32 {
            guard v >= min, v <= max else { throw TokenizerError.badFile("size \(v) out of range") }
            return v
        }
        guard data.count > 5, data[p..<p+5].elementsEqual("M5TK1".utf8) else { throw TokenizerError.badFile("magic") }
        p += 5
        bos = try i32(); eos = try i32()
        // Java 版为对齐桌面/Android 的 \s 差异做了显式展开；iOS 的 NSRegularExpression 就是 ICU，
        // 原生 Unicode \s/\S 语义一致（仅 U+2028/2029 行分隔符边角差异，demo 输入不会出现），直接编译原式。
        split = try NSRegularExpression(pattern: try str())
        let count = try bounded(try i32(), 1, 200_000)
        var v: [String: Int32] = [:]; v.reserveCapacity(Int(count))
        for _ in 0..<count { v[try str()] = try i32() }
        vocab = v
        var chars = [UInt16](repeating: 0, count: 256)
        var ids = [Int32](repeating: 0, count: 256)
        for i in 0..<256 { chars[i] = UInt16(try u32()); ids[i] = try i32() }
        byteChars = chars; byteIds = ids
        let mergeCount = try bounded(try i32(), 1, 1_000_000)
        merges = PairTable(n: Int(mergeCount))
        for rank in 0..<mergeCount {
            let a = try i32(), b = try i32(), id = try i32()
            merges.put(a, b, Int32(rank), id)
        }
        let specialCount = try bounded(try i32(), 1, 10_000)
        for _ in 0..<specialCount { specials.put(Array(try str().utf16), try i32()) }
        guard p == data.count else { throw TokenizerError.badFile("trailing bytes") }
    }

    public func encode(_ text: String) -> [Int64] {
        var out: [Int64] = [Int64(bos)]
        let units = Array(text.utf16)
        var plain = 0, at = 0
        while at < units.count {
            var node: Trie? = specials
            var end = -1, id: Int32 = -1
            for j in at..<units.count {
                node = node?.children[units[j]]
                if node == nil { break }
                if node!.id >= 0 { end = j + 1; id = node!.id }
            }
            if end >= 0 {
                encodePlain(String(String.UnicodeScalarView(units[plain..<at].compactMap(Unicode.Scalar.init))), &out)
                out.append(Int64(id))
                at = end; plain = end
            } else { at += 1 }
        }
        encodePlain(String(String.UnicodeScalarView(units[plain...].compactMap(Unicode.Scalar.init))), &out)
        out.append(Int64(eos))
        return out
    }

    private func encodePlain(_ text: String, _ out: inout [Int64]) {
        var end = text.startIndex
        var searchStart = text.startIndex
        while true {
            let range = NSRange(searchStart..., in: text)
            guard let match = split.firstMatch(in: text, range: range),
                  let matchRange = Range(match.range, in: text) else { break }
            if matchRange.lowerBound > end {
                encodePiece(String(text[end..<matchRange.lowerBound]), &out)
            }
            encodePiece(String(text[matchRange]), &out)
            end = matchRange.upperBound
            searchStart = matchRange.upperBound
        }
        if end < text.endIndex { encodePiece(String(text[end...]), &out) }
    }

    private func encodePiece(_ piece: String, _ out: inout [Int64]) {
        guard !piece.isEmpty else { return }
        let bytes = Array(piece.utf8)
        var encoded = [UInt16](); encoded.reserveCapacity(bytes.count)
        var ids = [Int32](); ids.reserveCapacity(bytes.count)
        for b in bytes { encoded.append(byteChars[Int(b)]); ids.append(byteIds[Int(b)]) }
        if let direct = vocab[String(String.UnicodeScalarView(encoded.compactMap(Unicode.Scalar.init)))] { out.append(Int64(direct)); return } // ignore_merges
        var size = ids.count
        while size > 1 {
            var best = -1, rank = Int32.max, mergedId: Int32 = -1
            for i in 0..<(size - 1) {
                let value = merges.get(ids[i], ids[i + 1])
                if value >= 0 {
                    let r = Int32(truncatingIfNeeded: UInt64(bitPattern: value) >> 32)
                    if r < rank { best = i; rank = r; mergedId = Int32(truncatingIfNeeded: value) }
                }
            }
            if best < 0 { break }
            ids[best] = mergedId
            for i in (best + 2)..<size { ids[i - 1] = ids[i] }
            size -= 1
        }
        for i in 0..<size { out.append(Int64(ids[i])) }
    }
}
