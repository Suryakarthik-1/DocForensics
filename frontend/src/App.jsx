import { useState, useRef } from 'react'
import './App.css'

const API = 'http://localhost:8000/api'

const VERDICT = {
  'AUTHENTIC':    { icon: '✓', tone: 'good', note: 'No significant tampering signals detected.' },
  'TAMPERED':     { icon: '!', tone: 'bad',  note: 'This document shows signs of manipulation.' },
  'AI-GENERATED': { icon: '◆', tone: 'warn', note: 'This image appears to be AI-generated.' },
}

function Dropzone({ onFile }) {
  const [drag, setDrag] = useState(false)
  const ref = useRef()
  return (
    <div
      className={`drop ${drag ? 'drag' : ''}`}
      onClick={() => ref.current.click()}
      onDragOver={e => { e.preventDefault(); setDrag(true) }}
      onDragLeave={() => setDrag(false)}
      onDrop={e => { e.preventDefault(); setDrag(false); e.dataTransfer.files[0] && onFile(e.dataTransfer.files[0]) }}
    >
      <input ref={ref} type="file" accept=".jpg,.jpeg,.png,.tiff,.pdf" hidden
             onChange={e => e.target.files[0] && onFile(e.target.files[0])} />
      <div className="drop-icon">⬆</div>
      <div className="drop-title">Drop a document or image</div>
      <div className="drop-sub">JPG · PNG · TIFF · PDF — up to 20 MB</div>
    </div>
  )
}

function Bar({ name, score }) {
  const pct = Math.round(score * 100)
  const tone = score > 0.6 ? 'bad' : score > 0.35 ? 'warn' : 'good'
  return (
    <div className="bar">
      <span className="bar-name">{name.replace(/_/g, ' ')}</span>
      <div className="bar-track"><div className={`bar-fill ${tone}`} style={{ width: `${pct}%` }} /></div>
      <span className="bar-pct">{pct}%</span>
    </div>
  )
}

export default function App() {
  const [file, setFile]     = useState(null)
  const [state, setState]   = useState('idle')   // idle | loading | done | error
  const [res, setRes]       = useState(null)
  const [err, setErr]       = useState('')
  const [view, setView]     = useState('heatmap')

  async function run(f) {
    setFile(f); setState('loading'); setRes(null); setView('heatmap')
    const form = new FormData(); form.append('file', f)
    try {
      const r = await fetch(`${API}/analyze`, { method: 'POST', body: form })
      if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || `Server error ${r.status}`)
      setRes(await r.json()); setState('done')
    } catch (e) { setErr(e.message); setState('error') }
  }

  const reset = () => { setFile(null); setRes(null); setState('idle') }
  const v = res ? (VERDICT[res.label] || VERDICT.AUTHENTIC) : null

  return (
    <div className="page">
      <header className="nav">
        <div className="brand"><span className="brand-mark">🔍</span> DocForensics</div>
        <div className="brand-sub">AI-powered tampering &amp; forgery detection</div>
      </header>

      <main className="wrap">
        {state === 'idle' && (
          <section className="intro">
            <h1>Is this document real?</h1>
            <p>Upload an image or PDF. Eight forensic detectors and a trained CNN
               inspect it for forgery, cloning, recompression, and AI generation.</p>
            <Dropzone onFile={run} />
          </section>
        )}

        {state === 'loading' && (
          <section className="status">
            <div className="ring" />
            <p className="status-main">Analyzing <b>{file?.name}</b></p>
            <p className="status-sub">Running detectors + CNN model…</p>
          </section>
        )}

        {state === 'error' && (
          <section className="status">
            <div className="status-bad">⚠</div>
            <p className="status-main">{err}</p>
            <button className="btn" onClick={reset}>Try again</button>
          </section>
        )}

        {state === 'done' && res && (
          <section className="result">
            <div className={`verdict ${v.tone}`}>
              <div className="verdict-badge">{v.icon}</div>
              <div className="verdict-text">
                <div className="verdict-label">{res.label}</div>
                <div className="verdict-note">{v.note}</div>
              </div>
              <div className="verdict-score">
                <div className="verdict-score-num">{Math.round(res.confidence * 100)}%</div>
                <div className="verdict-score-cap">suspicion</div>
              </div>
              <button className="btn ghost" onClick={reset}>New scan</button>
            </div>

            <div className="grid">
              <div className="card">
                <div className="card-head">
                  <h2>Evidence map</h2>
                  <div className="toggle">
                    <button className={view === 'heatmap' ? 'on' : ''} onClick={() => setView('heatmap')}>Heatmap</button>
                    <button className={view === 'original' ? 'on' : ''} onClick={() => setView('original')}>Original</button>
                  </div>
                </div>
                <div className="frame">
                  {file && <img className="frame-base" src={URL.createObjectURL(file)} alt="" />}
                  {view === 'heatmap' && res.heatmap_base64 &&
                    <img className="frame-heat" src={`data:image/png;base64,${res.heatmap_base64}`} alt="" />}
                </div>
                <p className="hint">Brighter / red areas indicate higher suspicion.</p>
              </div>

              <div className="card">
                <div className="card-head"><h2>Detector breakdown</h2></div>
                <div className="bars">
                  {res.per_detector.map(d => <Bar key={d.name} name={d.name} score={d.score} />)}
                </div>
                {res.evidence?.length > 0 && (
                  <div className="evidence">
                    <h3>Findings</h3>
                    <ul>{res.evidence.map((e, i) => <li key={i}>{e}</li>)}</ul>
                  </div>
                )}
              </div>
            </div>
          </section>
        )}
      </main>

      <footer className="foot">DocForensics · built with FastAPI, PyTorch &amp; React</footer>
    </div>
  )
}
