// Browser-side replacement for RoboCam's res/face/ folder.
// Stores captured face crops (as data URLs) keyed by name, using
// IndexedDB so captured data survives page reloads — mirroring the
// persistence chapter8.py gets for free from the filesystem.

const DB_NAME = 'robocam-faces'
const DB_VERSION = 1
const STORE_NAME = 'captures'

function openDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION)
    req.onupgradeneeded = () => {
      const db = req.result
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        const store = db.createObjectStore(STORE_NAME, { keyPath: 'id', autoIncrement: true })
        store.createIndex('name', 'name', { unique: false })
      }
    }
    req.onsuccess = () => resolve(req.result)
    req.onerror = () => reject(req.error)
  })
}

/** Saves a captured face crop (data URL) under a person's name. */
export async function saveCapture(name, dataUrl) {
  const db = await openDB()
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readwrite')
    tx.objectStore(STORE_NAME).add({ name, dataUrl, createdAt: Date.now() })
    tx.oncomplete = () => resolve()
    tx.onerror = () => reject(tx.error)
  })
}

/** Returns all captures, grouped by name: { name: [dataUrl, ...] } */
export async function getAllCapturesByName() {
  const db = await openDB()
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readonly')
    const req = tx.objectStore(STORE_NAME).getAll()
    req.onsuccess = () => {
      const grouped = {}
      for (const row of req.result) {
        if (!grouped[row.name]) grouped[row.name] = []
        grouped[row.name].push(row.dataUrl)
      }
      resolve(grouped)
    }
    req.onerror = () => reject(req.error)
  })
}

/** Deletes all captures for a given name. Mirrors DeleteFaceData(). */
export async function deleteCapturesByName(name) {
  const db = await openDB()
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readwrite')
    const store = tx.objectStore(STORE_NAME)
    const index = store.index('name')
    const req = index.openCursor(IDBKeyRange.only(name))
    req.onsuccess = () => {
      const cursor = req.result
      if (cursor) {
        cursor.delete()
        cursor.continue()
      }
    }
    tx.oncomplete = () => resolve()
    tx.onerror = () => reject(tx.error)
  })
}

/** Loads a data URL into an HTMLCanvasElement (for re-running detection on saved captures). */
export function dataUrlToCanvas(dataUrl) {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => {
      const canvas = document.createElement('canvas')
      canvas.width = img.naturalWidth
      canvas.height = img.naturalHeight
      canvas.getContext('2d').drawImage(img, 0, 0)
      resolve(canvas)
    }
    img.onerror = reject
    img.src = dataUrl
  })
}
