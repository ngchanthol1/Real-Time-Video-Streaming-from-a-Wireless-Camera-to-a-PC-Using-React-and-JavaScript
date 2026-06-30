// Shared face-crop logic, ported from the __preprocess() methods in
// face_landmark.py and face_recognizer.py. Both pad the source image,
// square the bbox with 20% extra width, crop, then resize to the
// model's expected input size (160x160 for landmarks, 112x112 for
// recognition embeddings).

/**
 * Crops a square, padded region around a face bbox from a source canvas,
 * resized to targetSize x targetSize. Mirrors the Python logic exactly:
 * pad image by `add` on all sides (mid-gray), square the box at 1.4x
 * width, crop, resize.
 *
 * bbox: [x1, y1, x2, y2] in source-image pixel coordinates.
 */
export function cropFaceSquare(sourceCanvas, bbox, targetSize, minFace = 20) {
  const srcW = sourceCanvas.width
  const srcH = sourceCanvas.height

  let [x1, y1, x2, y2] = bbox
  const bboxWidth = x2 - x1
  const bboxHeight = y2 - y1

  if (bboxWidth <= minFace || bboxHeight <= minFace) return null

  const add = Math.round(Math.max(bboxWidth, bboxHeight))

  // Padded canvas (mid-gray border), matching cv2.copyMakeBorder(...127,127,127)
  const padded = document.createElement('canvas')
  padded.width = srcW + add * 2
  padded.height = srcH + add * 2
  const pctx = padded.getContext('2d')
  pctx.fillStyle = 'rgb(127,127,127)'
  pctx.fillRect(0, 0, padded.width, padded.height)
  pctx.drawImage(sourceCanvas, add, add)

  // Shift bbox into padded coordinate space
  x1 += add
  y1 += add
  x2 += add
  y2 += add

  const faceWidth = (1 + 2 * 0.2) * bboxWidth
  const centerX = Math.floor((x1 + x2) / 2)
  const centerY = Math.floor((y1 + y2) / 2)

  const sx1 = Math.floor(centerX - faceWidth / 2)
  const sy1 = Math.floor(centerY - faceWidth / 2)
  const sx2 = Math.floor(centerX + faceWidth / 2)
  const sy2 = Math.floor(centerY + faceWidth / 2)

  const cropW = Math.max(1, sx2 - sx1)
  const cropH = Math.max(1, sy2 - sy1)

  const out = document.createElement('canvas')
  out.width = targetSize
  out.height = targetSize
  const octx = out.getContext('2d')
  octx.drawImage(padded, sx1, sy1, cropW, cropH, 0, 0, targetSize, targetSize)

  return out
}

/** Reads an HTMLCanvasElement into a flat Float32Array, RGB order, [0,255] range. */
export function canvasToFloat32RGB(canvas) {
  const ctx = canvas.getContext('2d')
  const { data, width, height } = ctx.getImageData(0, 0, canvas.width, canvas.height)
  const out = new Float32Array(width * height * 3)
  let j = 0
  for (let i = 0; i < data.length; i += 4) {
    out[j++] = data[i]
    out[j++] = data[i + 1]
    out[j++] = data[i + 2]
  }
  return out
}

/** Euclidean distance between two equal-length numeric arrays. */
export function euclideanDistance(a, b) {
  let sum = 0
  for (let i = 0; i < a.length; i++) {
    const diff = a[i] - b[i]
    sum += diff * diff
  }
  return Math.sqrt(sum)
}
