// tfjs-tflite's npm ESM build is broken for bundlers (it expects to be
// loaded as a classic <script>, see index.html), so we use the global
// `tflite` object that script tag attaches to window instead of an import.
import {
  letterboxToCanvas,
  nms,
  unletterboxBoxes,
} from './faceDetectorUtils'
import { cropFaceSquare, euclideanDistance } from './faceCropUtils'

const DETECTOR_INPUT = 320
const LANDMARK_INPUT = 160
const RECOGNIZER_INPUT = 112
const RECOGNITION_THRESHOLD = 0.1 // exact threshold from face_recognizer.py

/**
 * RoboCamEngine — a browser-side port of RoboCam's Python face pipeline.
 * Loads the same .tflite models RoboCam ships (face_detector, face_keypoints,
 * face_recognizer) and reproduces detection -> landmark -> recognition,
 * including named-face capture/train/delete, matching chapter8.py's flow.
 */
export class RoboCamEngine {
  constructor() {
    this.detectorModel = null
    this.landmarkModel = null
    this.recognizerModel = null
    // registered[name] = array of embedding arrays
    this.registered = {}
  }

  async loadModels(onProgress) {
    const tflite = globalThis.tflite
    if (!tflite) {
      throw new Error(
        'tfjs-tflite did not load. Check your internet connection (the library loads from a CDN) and reload the page.'
      )
    }

    onProgress?.('Loading face detector model...')
    this.detectorModel = await tflite.loadTFLiteModel('/models/face_detector.tflite')

    onProgress?.('Loading landmark model...')
    this.landmarkModel = await tflite.loadTFLiteModel('/models/face_keypoints.tflite')

    onProgress?.('Loading recognizer model...')
    this.recognizerModel = await tflite.loadTFLiteModel('/models/face_recognizer.tflite')

    onProgress?.('All models loaded.')
  }

  /**
   * Runs face detection on a source canvas (the current video frame).
   * Returns an array of [x1,y1,x2,y2,score] boxes in source-image coords.
   * Direct port of FaceDetector.__call__ in face_detector.py.
   */
  detectFaces(sourceCanvas, scoreThres = 0.5, iouThres = 0.3) {
    const { canvas: letterboxed, scale, dx, dy } = letterboxToCanvas(
      sourceCanvas,
      DETECTOR_INPUT
    )

    const tf = globalThis.tf
    const input = tf.tidy(() => {
      const img = tf.browser.fromPixels(letterboxed).toFloat()
      return img.expandDims(0)
    })

    const outputTensor = this.detectorModel.predict(input)
    const raw = outputTensor.dataSync() // flat [4200*5]
    input.dispose()
    outputTensor.dispose()

    const numBoxes = raw.length / 5
    const boxes = []
    for (let i = 0; i < numBoxes; i++) {
      const off = i * 5
      boxes.push([raw[off], raw[off + 1], raw[off + 2], raw[off + 3], raw[off + 4]])
    }

    const kept = nms(boxes, scoreThres, iouThres)
    return unletterboxBoxes(kept, scale, dx, dy)
  }

  /**
   * Runs the 68-point landmark model on a single face bbox.
   * Direct port of FaceLandmark.__call__ in face_landmark.py.
   */
  getLandmarks(sourceCanvas, bbox) {
    const cropped = cropFaceSquare(sourceCanvas, bbox, LANDMARK_INPUT)
    if (!cropped) return null

    const tf = globalThis.tf
    const input = tf.tidy(() => tf.browser.fromPixels(cropped).toFloat().expandDims(0))

    const outputs = this.landmarkModel.predict(input)
    // Identity_2 (136 = 68*2) holds the landmark points; tfjs-tflite
    // returns either an array of tensors or a named map depending on
    // the model signature — handle both shapes defensively.
    const landmarkTensor = Array.isArray(outputs)
      ? outputs[2]
      : outputs.Identity_2 ?? outputs

    const flat = landmarkTensor.dataSync()
    input.dispose()
    if (Array.isArray(outputs)) outputs.forEach((o) => o.dispose())
    else if (outputs.dispose) outputs.dispose()
    else Object.values(outputs).forEach((o) => o.dispose?.())

    // Map normalized landmark coords back into source-image pixel space.
    // Mirrors postprocess(): scaled by crop box size, anchored at bbox origin.
    const bboxWidth = bbox[2] - bbox[0]
    const add = Math.round(Math.max(bboxWidth, bbox[3] - bbox[1]))
    const faceWidth = (1 + 2 * 0.2) * bboxWidth
    const centerX = Math.floor((bbox[0] + add + (bbox[2] + add)) / 2)
    const centerY = Math.floor((bbox[1] + add + (bbox[3] + add)) / 2)
    const sx1 = Math.floor(centerX - faceWidth / 2) - add
    const sy1 = Math.floor(centerY - faceWidth / 2) - add
    const cropSize = faceWidth

    const points = []
    for (let i = 0; i < 68; i++) {
      const nx = flat[i * 2]
      const ny = flat[i * 2 + 1]
      points.push([nx * cropSize + sx1, ny * cropSize + sy1])
    }
    return points
  }

  /**
   * Computes a 192-dim embedding for a face bbox.
   * Direct port of the embedding portion of face_recognizer.py.
   */
  getEmbedding(sourceCanvas, bbox) {
    const cropped = cropFaceSquare(sourceCanvas, bbox, RECOGNIZER_INPUT)
    if (!cropped) return null

    const tf = globalThis.tf
    const input = tf.tidy(() => tf.browser.fromPixels(cropped).toFloat().expandDims(0))
    const output = this.recognizerModel.predict(input)
    const embedding = Array.from(output.dataSync())
    input.dispose()
    output.dispose()
    return embedding
  }

  /** Finds the nearest registered name for an embedding. Mirrors __findNearest(). */
  findNearest(embedding) {
    let best = null
    for (const [name, embeddings] of Object.entries(this.registered)) {
      for (const known of embeddings) {
        const dist = euclideanDistance(embedding, known)
        if (best === null || dist < best.distance) {
          best = { name, distance: dist }
        }
      }
    }
    return best
  }

  /** Recognizes a face: returns the matched name, or 'Human' if unknown/unregistered. */
  recognize(embedding) {
    if (Object.keys(this.registered).length === 0) return 'Human'
    const nearest = this.findNearest(embedding)
    if (nearest && nearest.distance < RECOGNITION_THRESHOLD) return nearest.name
    return 'Human'
  }

  /** Registers a new embedding under a name. */
  registerEmbedding(name, embedding) {
    if (!this.registered[name]) this.registered[name] = []
    this.registered[name].push(embedding)
  }

  /** Removes all trained data for a given name. Mirrors DeleteFaceData(). */
  deleteFaceData(name) {
    delete this.registered[name]
  }

  /** Clears all trained embeddings (used before a full retrain pass). */
  clearRegistered() {
    this.registered = {}
  }

  /** List of currently trained names with sample counts. */
  listRegistered() {
    return Object.entries(this.registered).map(([name, embeddings]) => ({
      name,
      samples: embeddings.length,
    }))
  }
}
