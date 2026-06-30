import { useRef, useState } from 'react'
import './App.css'

const DEFAULT_STREAM_URL = 'http://192.168.4.1:81/stream'

function App() {
  const [streamUrl, setStreamUrl] = useState(DEFAULT_STREAM_URL)
  const [connected, setConnected] = useState(false)
  const [error, setError] = useState('')
  const imgRef = useRef(null)

  const handleConnect = () => {
    setError('')
    if (imgRef.current) {
      imgRef.current.src = `${streamUrl}${streamUrl.includes('?') ? '&' : '?'}t=${Date.now()}`
    }
  }

  const handleDisconnect = () => {
    if (imgRef.current) imgRef.current.src = ''
    setConnected(false)
  }

  const handleLoad = () => setConnected(true)

  const handleError = () => {
    setConnected(false)
    setError(`Could not reach ${streamUrl}. Make sure your PC WiFi is connected to the RoboCam network.`)
  }

  return (
    <div className="app">
      <header className="header">
        <h1>RoboCam — Camera Viewer</h1>
        <p className="status-line">
          Status:{' '}
          <span className={connected ? 'status-ok' : 'status-idle'}>
            {connected ? 'Connected — stream live' : 'Disconnected'}
          </span>
        </p>
      </header>

      <section className="controls">
        <label htmlFor="streamUrl">Camera stream URL</label>
        <input
          id="streamUrl"
          type="text"
          value={streamUrl}
          onChange={(e) => setStreamUrl(e.target.value)}
          disabled={connected}
        />
        <div className="button-row">
          {!connected ? (
            <button onClick={handleConnect}>Connect to Camera</button>
          ) : (
            <button onClick={handleDisconnect} className="danger">Disconnect</button>
          )}
        </div>
        {error && <p className="error">{error}</p>}
      </section>

      <section className="video-stage">
        <div className="video-wrapper">
          {!connected && <div className="placeholder">No video connected</div>}
          <img
            ref={imgRef}
            alt="RoboCam stream"
            crossOrigin="anonymous"
            onLoad={handleLoad}
            onError={handleError}
            className={connected ? 'stream-img' : 'stream-img hidden'}
          />
        </div>
      </section>
    </div>
  )
}

export default App
