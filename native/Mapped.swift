typealias Scalars = [Unicode.Scalar]
func isSpace(_ scalar: Unicode.Scalar) -> Bool {
    switch scalar.value {
    case 0x09...0x0D, 0x1C...0x20, 0x85, 0xA0, 0x1680, 0x2000...0x200A, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000:
        true
    default:
        false
    }
}

struct Mapped: Equatable {
    var text: Scalars
    var origin: [Int]

    func slice(_ range: Range<Int>) -> Mapped {
        Mapped(text: Array(text[range]), origin: Array(origin[range]))
    }

    func keep(_ mask: [Bool]) -> Mapped {
        var kept = Mapped(text: [], origin: [])
        for index in text.indices where mask[index] {
            kept.text.append(text[index])
            kept.origin.append(origin[index])
        }
        return kept
    }

    func collapsed() -> Mapped {
        var result = Mapped(text: [], origin: [])
        for (scalar, index) in zip(text, origin) {
            if isSpace(scalar) {
                if let last = result.text.last, last != " " {
                    result.text.append(" ")
                    result.origin.append(index)
                }
            } else {
                result.text.append(scalar)
                result.origin.append(index)
            }
        }
        if result.text.last == " " {
            result.text.removeLast()
            result.origin.removeLast()
        }
        return result
    }

    func lined() -> Mapped {
        var result = Mapped(text: [], origin: [])
        var blank = false
        for range in lines {
            let line = slice(range).collapsed()
            if line.text.isEmpty {
                blank = true
                continue
            }
            if !result.text.isEmpty {
                let gap: Scalars = blank ? ["\n", "\n"] : ["\n"]
                result.text += gap
                result.origin += Array(repeating: origin[range.lowerBound - 1], count: gap.count)
            }
            result.text += line.text
            result.origin += line.origin
            blank = false
        }
        return result
    }

    var lines: [Range<Int>] {
        var bounds: [Range<Int>] = []
        var start = 0
        for index in text.indices where text[index] == "\n" {
            bounds.append(start..<index)
            start = index + 1
        }
        bounds.append(start..<text.count)
        return bounds
    }
}

extension Scalars {
    func isBlank(_ range: Range<Int>) -> Bool {
        self[range].allSatisfy(isSpace)
    }
}
