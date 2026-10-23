import Foundation

struct Decoding {
    let closing: [Unicode.Scalar: Unicode.Scalar]
    let opening: [Unicode.Scalar: Unicode.Scalar]
}

private func unbalanced(_ part: ArraySlice<Unicode.Scalar>, _ scalar: Unicode.Scalar, _ mate: Unicode.Scalar) -> Bool {
    let count = part.count(where: { $0 == scalar })
    return scalar == mate ? count % 2 == 1 : count > part.count(where: { $0 == mate })
}
private func trimmed(_ text: Scalars, _ span: Range<Int>, _ decoding: Decoding) -> Range<Int> {
    var start = span.lowerBound
    var end = span.upperBound
    while end - start > 1, let mate = decoding.closing[text[end - 1]] {
        guard unbalanced(text[start..<end], text[end - 1], mate) else { break }
        end -= 1
    }
    while end - start > 1, let mate = decoding.opening[text[start]] {
        guard unbalanced(text[start..<end], text[start], mate) else { break }
        start += 1
    }
    return start..<end
}

private func balancedTicks(_ text: Scalars, _ span: Range<Int>) -> Range<Int> {
    let tick: Unicode.Scalar = "`"
    if text[span].count(where: { $0 == tick }) % 2 == 0 { return span }
    if let closer = text[span.upperBound...].firstIndex(of: tick), closer - span.upperBound <= 8,
       !text[span.upperBound..<closer].contains(where: isSpace) {
        return span.lowerBound..<(closer + 1)
    }
    if span.lowerBound > 0, text[span.lowerBound - 1] == tick { return (span.lowerBound - 1)..<span.upperBound }
    if text[span.lowerBound] == tick { return (span.lowerBound + 1)..<span.upperBound }
    return text[span.upperBound - 1] == tick ? span.lowerBound..<(span.upperBound - 1) : span
}

private func cutTrailing(_ text: Scalars, _ span: Range<Int>, _ trailing: Set<Unicode.Scalar>) -> Range<Int> {
    var end = span.upperBound
    while end - span.lowerBound > 1, trailing.contains(text[end - 1]) { end -= 1 }
    return span.lowerBound..<end
}
private func withMarks(_ text: Scalars, _ end: Int) -> Int {
    var end = end
    while end < text.count, [.nonspacingMark, .spacingMark, .enclosingMark].contains(text[end].properties.generalCategory) { end += 1 }
    return end
}

func spans(in text: Scalars, offsets: [Range<Int>], tags: [Int], _ decoding: Decoding, trailing: Set<Unicode.Scalar>) -> [(span: Range<Int>, kind: Int)] {
    var found: [(span: Range<Int>, kind: Int)] = []
    var current: Range<Int>?
    var kind = 0
    func close() {
        if let run = current {
            let marked = run.lowerBound..<withMarks(text, run.upperBound)
            let span = trimmed(text, cutTrailing(text, balancedTicks(text, trimmed(text, marked, decoding)), trailing), decoding)
            if text[span].contains(where: { !trailing.contains($0) }) { found.append((span, kind)) }
        }
    }
    for (offset, tag) in zip(offsets, tags) where offset.upperBound != 0 {
        if tag > 0, tag % 2 == 0, let run = current, (tag - 1) / 2 == kind {
            current = run.lowerBound..<offset.upperBound
            continue
        }
        close()
        let begins = tag > 0 && tag % 2 == 1
        current = begins ? offset : nil
        kind = max(tag - 1, 0) / 2
    }
    close()
    return found
}

func afterContext(_ spans: [(span: Range<Int>, kind: Int)], _ context: Int) -> [(span: Range<Int>, kind: Int)] {
    spans.filter { $0.span.upperBound > context }.map { (max($0.span.lowerBound, context)..<$0.span.upperBound, $0.kind) }
}

private let sentenceEnd = Set(".?!".unicodeScalars)
private let sentenceClosers = Set("\"'”’)]`".unicodeScalars)
private let abbreviations = ["e.g.", "i.e.", "etc.", "vs.", "cf."].map { Array($0.unicodeScalars) }

private func sentenceEnd(_ text: Scalars, from start: Int, limit: Int) -> Int {
    var i = start
    while i < limit {
        guard sentenceEnd.contains(text[i]) else {
            i += 1
            continue
        }
        while i < limit, sentenceEnd.contains(text[i]) { i += 1 }
        while i < limit, sentenceClosers.contains(text[i]) { i += 1 }
        let abbreviated = abbreviations.contains { word in
            i >= word.count && zip(text[(i - word.count)..<i], word).allSatisfy { $0.properties.lowercaseMapping.unicodeScalars.elementsEqual([$1]) }
        }
        if (i == limit || isSpace(text[i])) && !abbreviated { return i }
    }
    return limit
}

private func sentences(_ text: Scalars) -> [Range<Int>] {
    var found: [Range<Int>] = []
    var lineStart = 0
    while true {
        let lineEnd = text[lineStart...].firstIndex(of: "\n") ?? text.count
        var i = lineStart
        if let marker = Line(text[lineStart..<lineEnd]).starting(item) { i += Line(text[lineStart..<lineEnd]).scalars(marker.range).upperBound }
        var start = lineStart
        while i < lineEnd {
            while i < lineEnd, isSpace(text[i]) { i += 1 }
            if i >= lineEnd { break }
            let stop = sentenceEnd(text, from: i, limit: lineEnd)
            found.append((start == lineStart ? start : i)..<stop)
            start = stop
            i = stop
        }
        if lineEnd == text.count { break }
        lineStart = lineEnd + 1
    }
    return found.filter { text[$0].contains { !isSpace($0) } }
}

private func softmax(_ row: ArraySlice<Float>) -> [Double] {
    let top = Double(row.max()!)
    let exps = row.map { exp(Double($0) - top) }
    var total = 0.0
    for value in exps { total += value }
    return exps.map { $0 / total }
}

private func largest(_ values: [Double]) -> Int {
    var best = 0
    for index in values.indices where values[index] > values[best] { best = index }
    return best
}

private struct Candidate {
    var start: Int
    var end: Int
    var inside: [Double]
}

/// A sentence is a question sentence when the mean probability of its tokens inside a kind is above
/// that of O; it starts a question when its first token is likelier a B than an I, or when no
/// question sentence is before it. The kind of a question is the likeliest over its sentences.
func sentenceSpans(in text: Scalars, offsets: [Range<Int>], logits: [Float], classes: Int) -> [(span: Range<Int>, kind: Int)] {
    let probabilities = offsets.indices.map { softmax(logits[($0 * classes)..<(($0 + 1) * classes)]) }
    var found: [Candidate] = []
    var open = false
    for sentence in sentences(text) {
        var rows: [[Double]] = []
        for token in offsets.indices where offsets[token].lowerBound < offsets[token].upperBound && sentence.contains(offsets[token].lowerBound) {
            rows.append(probabilities[token])
        }
        open = open && !rows.isEmpty
        guard !rows.isEmpty else { continue }
        var mean = [Double](repeating: 0, count: classes)
        for row in rows { for column in 0..<classes { mean[column] += row[column] } }
        for column in 0..<classes { mean[column] /= Double(rows.count) }
        var inside: [Double] = []
        for kind in 0..<(classes - 1) / 2 { inside.append(mean[1 + 2 * kind] + mean[2 + 2 * kind]) }
        let kind = largest(inside)
        guard mean[0] < inside[kind] else {
            open = false
            continue
        }
        let first = rows[0]
        if open, first[2 * kind + 1] <= first[2 * kind + 2] {
            found[found.count - 1].end = sentence.upperBound
            for index in inside.indices { found[found.count - 1].inside[index] += inside[index] }
        } else {
            found.append(Candidate(start: sentence.lowerBound, end: sentence.upperBound, inside: inside))
        }
        open = true
    }
    var out: [(span: Range<Int>, kind: Int)] = []
    for item in found {
        var (start, end) = (item.start, item.end)
        while start < end, isSpace(text[start]) { start += 1 }
        while end > start, isSpace(text[end - 1]) { end -= 1 }
        out.append((start..<end, largest(item.inside)))
    }
    return out
}
