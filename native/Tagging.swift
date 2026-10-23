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

func completeSentences(_ text: Scalars, _ spans: [(span: Range<Int>, kind: Int)]) -> [(span: Range<Int>, kind: Int)] {
    spans.enumerated().map { number, item in
        let (a, b) = (item.span.lowerBound, item.span.upperBound)
        var limit = text[b...].firstIndex(of: "\n") ?? text.count
        if number + 1 < spans.count { limit = min(limit, spans[number + 1].span.lowerBound) }
        var i = b
        while i > 0, sentenceClosers.contains(text[i - 1]) { i -= 1 }
        let closed = i > 0 && (sentenceEnd.contains(text[i - 1]) || text[i - 1] == ":" || text[i - 1] == ";")
        if !text[b..<limit].contains(where: { !isSpace($0) })
            || closed && (b == text.count || isSpace(text[b]) || sentenceClosers.contains(text[b])) {
            return item
        }
        var cut: Int?
        for mark in a..<b where text[mark] == "?" {
            var j = mark + 1
            while j < b, sentenceClosers.contains(text[j]) { j += 1 }
            if j < b, isSpace(text[j]) { cut = j }
        }
        if let cut { return (a..<cut, item.kind) }
        var end = sentenceEnd(text, from: b, limit: limit)
        while end > b {
            if isSpace(text[end - 1]) || text[end - 1] == "," {
                end -= 1
                continue
            }
            let word = ["and", "or"].map { Array($0.unicodeScalars) }.first {
                end - $0.count > a && text[(end - $0.count)..<end].elementsEqual($0) && isSpace(text[end - $0.count - 1])
            }
            guard let word else { break }
            end -= word.count
        }
        return (a..<end, item.kind)
    }
}
