#if canImport(CoreML)
import CoreML
import Foundation

private func packGreedy(_ lengths: [Int], limit: Int) -> [[Int]] {
    var rows: [[Int]] = []
    var used = limit
    for (index, length) in lengths.enumerated() {
        if used + length > limit {
            rows.append([])
            used = 0
        }
        rows[rows.count - 1].append(index)
        used += length
    }
    return rows
}

struct Models {
    private let models: [String: MLModel]
    private let padTokenId: Int
    private let length: Int
    private let classes: [String: Int]
    private let lock = NSLock()

    init(directory: URL, settings: Settings) throws {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndNeuralEngine
        models = try Dictionary(uniqueKeysWithValues: settings.tags.keys.sorted().map { task in
            (task, try MLModel(contentsOf: directory.appendingPathComponent("\(task).mlmodelc"), configuration: configuration))
        })
        padTokenId = settings.padTokenId
        length = settings.maxTokens
        classes = settings.tags.mapValues(\.count)
    }

    func logits(_ task: String, _ ids: [[Int]]) throws -> (logits: [[Float]], packs: [[Int]]) {
        var out = [[Float]](repeating: [], count: ids.count)
        let packs = packGreedy(ids.map(\.count), limit: length)
        for row in packs {
            let inputIds = try MLMultiArray(shape: [1, NSNumber(value: length)], dataType: .int32)
            let positions = try MLMultiArray(shape: [1, NSNumber(value: length)], dataType: .int32)
            let segments = try MLMultiArray(shape: [1, NSNumber(value: length)], dataType: .int32)
            let idPointer = inputIds.dataPointer.bindMemory(to: Int32.self, capacity: length)
            let positionPointer = positions.dataPointer.bindMemory(to: Int32.self, capacity: length)
            let segmentPointer = segments.dataPointer.bindMemory(to: Int32.self, capacity: length)
            for index in 0..<length {
                idPointer[index] = Int32(padTokenId)
                positionPointer[index] = 0
                segmentPointer[index] = 0
            }
            var placed: [(unit: Int, offset: Int)] = []
            var offset = 0
            for (number, unit) in row.enumerated() {
                for (index, id) in ids[unit].enumerated() {
                    idPointer[offset + index] = Int32(id)
                    positionPointer[offset + index] = Int32(index)
                    segmentPointer[offset + index] = Int32(number + 1)
                }
                placed.append((unit, offset))
                offset += ids[unit].count
            }
            let classes = self.classes[task]!
            let scores = try predict(task, inputIds, positions, segments, used: offset)
            for (unit, start) in placed {
                out[unit] = Array(scores[(start * classes)..<((start + ids[unit].count) * classes)])
            }
        }
        return (out, packs)
    }

    private func predict(_ task: String, _ ids: MLMultiArray, _ positions: MLMultiArray, _ segments: MLMultiArray, used: Int) throws -> [Float] {
        lock.lock()
        defer { lock.unlock() }
        let result = try models[task]!.prediction(from: MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLFeatureValue(multiArray: ids),
            "position_ids": MLFeatureValue(multiArray: positions),
            "segment_ids": MLFeatureValue(multiArray: segments),
        ]))
        guard let logits = result.featureValue(for: "logits")?.multiArrayValue, logits.shape.count == 4 else {
            throw CocoaError(.coderReadCorrupt)
        }
        guard let classes = self.classes[task], logits.shape[1].intValue == classes else { throw CocoaError(.coderReadCorrupt) }
        let strides = logits.strides.map(\.intValue)
        let half = logits.dataType == .float16
        let pointer = logits.dataPointer
        func value(_ tag: Int, _ position: Int) -> Float {
            let index = tag * strides[1] + position * strides[3]
            return half ? Float(pointer.load(fromByteOffset: index * 2, as: Float16.self)) : pointer.load(fromByteOffset: index * 4, as: Float.self)
        }
        return (0..<used).flatMap { position in (0..<classes).map { value($0, position) } }
    }
}
#endif
