// Direct JS port of RoboCam's FaceDetector (face_detector.py).
// Model: face_detector.tflite — input 320x320x3 float32, output [1,4200,5]
// where each row is [x1, y1, x2, y2, score] already in letterboxed-image
// pixel coordinates (no manual anchor decoding needed — baked into the model).

const INPUT_SIZE = 320

/**
 * Letterbox-resizes a source canvas/image into a 320x320 square, padded
 * with mid-gray (127,127,127), matching face_detector.py's preprocess().
 * Returns { canvas, scale, dx, dy } so boxes can be mapped back later.
 */
export function letterboxToCanvas(sourceCanvas, targetSize = INPUT_SIZE) {
  const srcW = sourceCanvas.width
  const srcH = sourceCanvas.height

  const scale = Math.min(targetSize / srcW, targetSize / srcH)
  const newW = Math.round(srcW * scale)
  const newH = Math.round(srcH * scale)
  const dx = Math.floor((targetSize - newW) / 2)
  const dy = Math.floor((targetSize - newH) / 2)

  const out = document.createElement('canvas')
  out.width = targetSize
  out.height = targetSize
  const ctx = out.getContext('2d')
  ctx.fillStyle = 'rgb(127,127,127)'
  ctx.fillRect(0, 0, targetSize, targetSize)
  ctx.drawImage(sourceCanvas, 0, 0, srcW, srcH, dx, dy, newW, newH)

  return { canvas: out, scale, dx, dy }
}

/**
 * Standard greedy NMS, ported 1:1 from py_nms() in face_detector.py.
 * boxes: Float32Array-like rows of [x1,y1,x2,y2,score]
 */
export function nms(boxes, scoreThres = 0.5, iouThres = 0.3) {
  const kept = []
  const filtered = boxes.filter((b) => b[4] > scoreThres)
  filtered.sort((a, b) => b[4] - a[4])

  const removed = new Array(filtered.length).fill(false)

  for (let i = 0; i < filtered.length; i++) {
    if (removed[i]) continue
    const cur = filtered[i]
    kept.push(cur)
    const area = (cur[2] - cur[0]) * (cur[3] - cur[1])

    for (let j = i + 1; j < filtered.length; j++) {
      if (removed[j]) continue
      const other = filtered[j]
      const xx1 = Math.max(cur[0], other[0])
      const yy1 = Math.max(cur[1], other[1])
      const xx2 = Math.min(cur[2], other[2])
      const yy2 = Math.min(cur[3], other[3])
      const inter = Math.max(0, yy2 - yy1) * Math.max(0, xx2 - xx1)
      const otherArea = (other[2] - other[0]) * (other[3] - other[1])
      const iou = inter / (area + otherArea - inter)
      if (iou >= iouThres) removed[j] = true
    }
  }
  return kept
}

/**
 * Maps NMS'd boxes (in 320x320 letterboxed space) back to the original
 * source image's pixel coordinates. Ported from the boxes_scaler /
 * boxes_bias math in face_detector.py's __call__().
 */
export function unletterboxBoxes(boxes, scale, dx, dy) {
  return boxes.map((b) => [
    (b[0] - dx) / scale,
    (b[1] - dy) / scale,
    (b[2] - dx) / scale,
    (b[3] - dy) / scale,
    b[4],
  ])
}
