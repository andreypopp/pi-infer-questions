import Foundation

struct Unit {
    let text: Scalars
    let origin: [Int]
    let context: Int
}

let item = regex(#"\s*(?:[-*+•]|\d+[.)]|[A-Za-z][.)])\s"#)
private let closers: Set<Unicode.Scalar> = [")", "]", "\"", "'", "”"]

private func itemFlags(_ mapped: Mapped, _ bounds: [Range<Int>]) -> [Bool] {
    bounds.map { Line(mapped.text[$0]).starting(item) != nil }
}

private func endsWithQuestion(_ text: ArraySlice<Unicode.Scalar>) -> Bool {
    var end = text.endIndex
    while end > text.startIndex, isSpace(text[end - 1]) { end -= 1 }
    while end > text.startIndex, closers.contains(text[end - 1]) { end -= 1 }
    return end > text.startIndex && (text[end - 1] == "?" || text[end - 1] == ":")
}

/// A list paragraph joins the unit before it when that unit ends with ? or : (before closing
/// brackets and quotes), holds a list item or starts with a heading line. `context` is the length
/// of the heading or bold line at the start of a unit, with its line break.
func units(_ mapped: Mapped, _ headings: Set<Int>) -> [Unit] {
    let bounds = mapped.lines
    let isItem = itemFlags(mapped, bounds)
    var runs: [[Int]] = [[]]
    for number in bounds.indices {
        if !mapped.text.isBlank(bounds[number]) {
            runs[runs.count - 1].append(number)
        } else if !runs[runs.count - 1].isEmpty {
            runs.append([])
        }
    }
    var groups: [[Int]] = []
    for run in runs where !run.isEmpty {
        if let before = groups.last, isItem[run[0]],
           headings.contains(before[0]) || endsWithQuestion(mapped.text[bounds[before[before.count - 1]]])
            || before.contains(where: { isItem[$0] }) {
            groups[groups.count - 1] += run
        } else {
            groups.append(run)
        }
    }
    return groups.flatMap { group -> [Unit] in
        let lined = mapped.slice(bounds[group[0]].lowerBound..<bounds[group[group.count - 1]].upperBound).lined()
        var pieces = segments(lined)
        var context = 0
        if headings.contains(group[0]) {
            if pieces.count > 1 {
                pieces.removeFirst()
            } else {
                let head = lined.text.firstIndex(of: "\n")!
                context = lined.text[head...].firstIndex { $0 != "\n" }!
            }
        }
        return pieces.map { Unit(text: $0.text, origin: $0.origin, context: context) }
    }
}

private func markerKind(_ line: ArraySlice<Unicode.Scalar>) -> Int {
    guard let first = line.first(where: { !isSpace($0) }) else { return 2 }
    if first.properties.generalCategory == .decimalNumber { return 0 }
    return first.isASCII && first.properties.isAlphabetic ? 1 : 2
}

/// A unit with a top-level list item that holds a "?" splits: the text before the list, each such
/// item with its nested items, and each run of other top-level items are units.
private func segments(_ unit: Mapped) -> [Mapped] {
    let lines = unit.lines
    let isItem = itemFlags(unit, lines)
    let items = lines.indices.filter { isItem[$0] }
    guard let first = items.first else { return [unit] }
    let top = items.filter { markerKind(unit.text[lines[$0]]) == markerKind(unit.text[lines[first]]) }
    let asks = top.map { start in
        let end = items.first { $0 > start } ?? lines.count
        return unit.text[lines[start].lowerBound..<lines[end - 1].upperBound].contains("?")
    }
    if !asks.contains(true) { return [unit] }
    var cuts = Set(top.indices.filter { asks[$0] || ($0 > 0 && asks[$0 - 1]) }.map { top[$0] })
    cuts.insert(top[0])
    if top[0] > 0 { cuts.insert(0) }
    let bounds = cuts.sorted() + [lines.count]
    return zip(bounds, bounds.dropFirst()).map { from, to in
        let start = lines[from].lowerBound
        var end = to < lines.count ? lines[to].lowerBound - 1 : unit.text.count
        while end > start, isSpace(unit.text[end - 1]) { end -= 1 }
        return unit.slice(start..<end)
    }
}
