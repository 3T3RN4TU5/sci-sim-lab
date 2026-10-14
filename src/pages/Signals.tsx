import { useEffect, useMemo, useRef, useState } from 'react'
import { Plot } from '../components/Plot.tsx'
import { Tex } from '../components/Tex.tsx'
import { MethodSection, PageHeader, Panel, Readout, Segmented, Select, Slider } from '../components/ui.tsx'
import { fmt, fmtFixed } from '../lib/format.ts'
import { fitCanvas, useAnimationLoop, useElementWidth } from '../lib/hooks.ts'
import { C } from '../lib/palette.ts'
import { colormapLUT } from '../numerics/colormaps.ts'
import { amplitudeSpectrum, applyFir, designFir, frequencyResponse, stft, WINDOW_NAMES, type FilterType, type WindowId } from '../numerics/fft.ts'
import { Rng } from '../numerics/rng.ts'

const FS = 1024
const N = 1024
const WINDOWS = (Object.keys(WINDOW_NAMES) as WindowId[]).map((w) => ({ value: w, label: WINDOW_NAMES[w] }))
const db = (a: number) => 20 * Math.log10(Math.max(a, 1e-7))

export default function Signals() {
  const [comps, setComps] = useState([
    { f: 50, a: 1 },
    { f: 120.5, a: 0.5 },
    { f: 300, a: 0.05 },
  ])
  const [noise, setNoise] = useState(0.1)
  const [win, setWin] = useState<WindowId>('hann')
  const [ftype, setFtype] = useState<FilterType>('lowpass')
  const [fc1, setFc1] = useState(90)
  const [fc2, setFc2] = useState(200)
  const [taps, setTaps] = useState(101)
  const [fwin, setFwin] = useState<WindowId>('blackman')

  const signal = useMemo(() => {
    const rng = new Rng(3)
    return Float64Array.from({ length: N }, (_, i) => comps.reduce((s, c) => s + c.a * Math.sin((2 * Math.PI * c.f * i) / FS), 0) + noise * rng.normal())
  }, [comps, noise])
  const t = useMemo(() => Array.from({ length: N }, (_, i) => i / FS), [])
  const freqs = useMemo(() => Array.from({ length: N / 2 + 1 }, (_, k) => (k * FS) / N), [])
  const spec = useMemo(() => Array.from(amplitudeSpectrum(signal, win), db), [signal, win])
  const h = useMemo(() => designFir(ftype, taps, FS, fc1, fc2, fwin), [ftype, taps, fc1, fc2, fwin])
  const H = useMemo(() => Array.from(frequencyResponse(h, 4096), db), [h])
  const Hf = useMemo(() => Array.from({ length: 2049 }, (_, k) => (k * FS) / 4096), [])
  const filtered = useMemo(() => applyFir(signal, h), [signal, h])
  const specF = useMemo(() => Array.from(amplitudeSpectrum(filtered, win), db), [filtered, win])
  const show = 256

  const setComp = (i: number, key: 'f' | 'a', v: number) => setComps((cs) => cs.map((c, k) => (k === i ? { ...c, [key]: v } : c)))

  return (
    <div>
      <PageHeader eyebrow="Module 6 · Signal processing" title="Spectral analysis with a hand-written FFT">
        An iterative radix-2 Cooley–Tukey FFT (verified against an O(N²) DFT and NumPy) powers every view: windowed spectra, a short-time Fourier transform of a chirp and
        windowed-sinc FIR filter design.
      </PageHeader>
      <div className="grid gap-4 lg:grid-cols-[280px_minmax(0,1fr)]">
        <Panel title="Signal composer (fs = 1024 Hz, N = 1024)">
          <div className="space-y-4">
            {comps.map((c, i) => (
              <div key={i} className="space-y-2 rounded-xl border border-white/5 p-3">
                <Slider label={`Sine ${i + 1} frequency (Hz)`} value={c.f} min={1} max={500} step={0.5} onChange={(v) => setComp(i, 'f', v)} />
                <Slider label="Amplitude" value={c.a} min={0} max={1.5} step={0.01} onChange={(v) => setComp(i, 'a', v)} />
              </div>
            ))}
            <Slider label="White noise σ" value={noise} min={0} max={1} step={0.01} onChange={setNoise} />
            <Select<WindowId> label="Analysis window" value={win} onChange={setWin} options={WINDOWS} />
            <p className="text-xs text-slate-400">Frequencies off a bin centre (e.g. 120.5 Hz) leak into neighbouring bins; compare the Rectangular and Blackman windows.</p>
          </div>
        </Panel>
        <div className="min-w-0 space-y-4">
          <Panel title="Time domain (first 0.25 s)">
            <Plot
              ariaLabel="Signal in the time domain"
              height={200}
              xLabel="t (s)"
              yLabel="x(t)"
              series={[
                { x: t.slice(0, show), y: Array.from(signal.slice(0, show)), color: C.blue, label: 'input', width: 1.5 },
                { x: t.slice(0, show), y: Array.from(filtered.slice(0, show)), color: C.orange, label: 'filtered', width: 2 },
              ]}
            />
          </Panel>
          <Panel title={`Amplitude spectrum (${WINDOW_NAMES[win]} window)`}>
            <Plot
              ariaLabel="Amplitude spectrum in dB"
              height={230}
              xLabel="f (Hz)"
              yLabel="dB"
              yDomain={[-100, 5]}
              series={[
                { x: freqs, y: spec, color: C.blue, label: 'input', width: 1.5 },
                { x: freqs, y: specF, color: C.orange, label: 'filtered', width: 1.5 },
              ]}
            />
          </Panel>
          <Panel title="FIR filter design (windowed sinc, linear phase)">
            <div className="grid gap-4 md:grid-cols-[240px_minmax(0,1fr)]">
              <div className="space-y-3">
                <Segmented
                  label="Filter type"
                  value={ftype}
                  onChange={setFtype}
                  options={[
                    { value: 'lowpass', label: 'Low' },
                    { value: 'highpass', label: 'High' },
                    { value: 'bandpass', label: 'Band' },
                  ]}
                />
                <Slider label={ftype === 'bandpass' ? 'Lower cutoff (Hz)' : 'Cutoff (Hz)'} value={fc1} min={5} max={500} step={1} onChange={setFc1} />
                {ftype === 'bandpass' && <Slider label="Upper cutoff (Hz)" value={fc2} min={5} max={500} step={1} onChange={setFc2} />}
                <Slider label="Taps" value={taps} min={11} max={301} step={2} onChange={setTaps} />
                <Select<WindowId> label="Design window" value={fwin} onChange={setFwin} options={WINDOWS} />
              </div>
              <Plot
                ariaLabel="Filter frequency response"
                height={240}
                xLabel="f (Hz)"
                yLabel="|H(f)| dB"
                yDomain={[-120, 5]}
                series={[{ x: Hf, y: H, color: C.violet, label: '|H(f)|' }]}
                vLines={ftype === 'bandpass' ? [{ x: fc1 }, { x: fc2 }] : [{ x: fc1 }]}
              />
            </div>
          </Panel>
        </div>
      </div>
      <SpectrogramPanel />
      <MicPanel />
      <MethodSection
        references={[
          { text: 'J. W. Cooley, J. W. Tukey — An algorithm for the machine calculation of complex Fourier series, Math. Comp. 19 (1965) 297–301.' },
          { text: 'F. J. Harris — On the use of windows for harmonic analysis with the discrete Fourier transform, Proc. IEEE 66 (1978) 51–83.' },
          { text: 'A. V. Oppenheim, R. W. Schafer — Discrete-Time Signal Processing, 3rd ed., Pearson (2010).' },
        ]}
      >
        <p>
          The DFT <Tex>{String.raw`X_k = \sum_{n=0}^{N-1} x_n e^{-2\pi i kn/N}`}</Tex> costs <Tex>{String.raw`\mathcal O(N^2)`}</Tex>. Radix-2 decimation in time splits it into even and odd
          halves, <Tex>{String.raw`X_k = E_k + W_N^k O_k`}</Tex>, <Tex>{String.raw`X_{k+N/2} = E_k - W_N^k O_k`}</Tex> with <Tex>{String.raw`W_N = e^{-2\pi i/N}`}</Tex>, giving{' '}
          <Tex>{String.raw`\mathcal O(N\log N)`}</Tex>. The implementation is iterative and in place: a bit-reversal permutation followed by <Tex>{String.raw`\log_2 N`}</Tex> butterfly
          stages with cached twiddles. The inverse uses the conjugate twiddles and a <Tex>1/N</Tex> scale. Tests verify agreement with the naive DFT to{' '}
          <Tex>{String.raw`7\times10^{-15}\sqrt N`}</Tex>, exact round-trips and Parseval’s theorem <Tex>{String.raw`\sum |x_n|^2 = \frac1N \sum |X_k|^2`}</Tex>.
        </p>
        <p>
          Windows taper the frame to reduce spectral leakage: Hann <Tex>{String.raw`w_n = \tfrac12 - \tfrac12\cos\frac{2\pi n}{N-1}`}</Tex>, Hamming (0.54/0.46) and Blackman (0.42/0.5/0.08)
          trade main-lobe width for side-lobe level (−31, −43, −58 dB). The spectrum is scaled by <Tex>{String.raw`2/\sum w_n`}</Tex> so a sinusoid reads its amplitude.
        </p>
        <p>
          The STFT slides a window with hop <Tex>{String.raw`H`}</Tex>: <Tex>{String.raw`S(m, k) = \sum_n x_{n + mH}\, w_n\, e^{-2\pi i kn/M}`}</Tex>; longer windows sharpen frequency
          and blur time (uncertainty). FIR low-pass taps are the ideal impulse response <Tex>{String.raw`h_n = 2f_c\,\mathrm{sinc}(2 f_c (n - \tfrac{M-1}{2}))`}</Tex> times a window,
          normalised to unit DC gain; high-pass uses spectral inversion and band-pass the difference of two low-passes. They match <code>scipy.signal.firwin</code> to 10⁻¹².
        </p>
      </MethodSection>
    </div>
  )
}

function SpectrogramPanel() {
  const [size, setSize] = useState('256')
  const [win, setWin] = useState<WindowId>('hann')
  const wrapRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const width = useElementWidth(wrapRef)
  const height = 260
  const fs = 4096
  const dur = 4
  const chirp = useMemo(() => {
    const n = fs * dur
    const f0 = 20
    const f1 = 1000
    return Float64Array.from({ length: n }, (_, i) => {
      const t = i / fs
      const phase = 2 * Math.PI * (f0 * t + ((f1 - f0) / (2 * dur)) * t * t)
      return Math.sin(phase) + 0.25 * Math.sin(2 * phase)
    })
  }, [])
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || width < 50) return
    const M = Number(size)
    const s = stft(chirp, M, M / 4, win)
    const ctx = fitCanvas(canvas, width, height)
    if (!ctx) return
    const img = new ImageData(s.frames, s.bins)
    const lut = colormapLUT('inferno')
    for (let f = 0; f < s.frames; f++) {
      for (let k = 0; k < s.bins; k++) {
        const v = s.db[f * s.bins + k]
        const i = Math.max(0, Math.min(255, Math.round(((v + 90) / 90) * 255)))
        const p = ((s.bins - 1 - k) * s.frames + f) * 4
        img.data[p] = lut[i * 3]
        img.data[p + 1] = lut[i * 3 + 1]
        img.data[p + 2] = lut[i * 3 + 2]
        img.data[p + 3] = 255
      }
    }
    const off = document.createElement('canvas')
    off.width = s.frames
    off.height = s.bins
    off.getContext('2d')!.putImageData(img, 0, 0)
    ctx.imageSmoothingEnabled = true
    ctx.drawImage(off, 0, 0, width, height)
  }, [chirp, size, win, width])
  const M = Number(size)
  return (
    <Panel title="Spectrogram (STFT) of a linear chirp 20 → 1000 Hz with its second harmonic" className="mt-4">
      <div className="grid gap-4 md:grid-cols-[220px_minmax(0,1fr)]">
        <div className="space-y-3">
          <Select
            label="Window length M"
            value={size}
            onChange={setSize}
            options={['64', '128', '256', '512', '1024'].map((v) => ({ value: v, label: `${v} samples` }))}
          />
          <Select<WindowId> label="Window" value={win} onChange={setWin} options={WINDOWS} />
          <Readout label="Frequency resolution" value={`${fmtFixed(fs / M, 1)} Hz`} />
          <Readout label="Time resolution" value={`${fmtFixed((1000 * M) / fs, 1)} ms`} />
        </div>
        <div ref={wrapRef} className="min-w-0">
          <canvas ref={canvasRef} className="block rounded-xl" aria-label="Spectrogram, time horizontal, frequency vertical" />
          <div className="mt-1 flex justify-between font-mono text-[11px] text-slate-400">
            <span>t = 0 s</span>
            <span>frequency 0 – {fs / 2} Hz (vertical), inferno colormap, −90 … 0 dB</span>
            <span>t = {dur} s</span>
          </div>
        </div>
      </div>
    </Panel>
  )
}

function MicPanel() {
  const [status, setStatus] = useState<'off' | 'on' | 'error'>('off')
  const [err, setErr] = useState('')
  const audio = useRef<{ ctx: AudioContext; an: AnalyserNode; stream: MediaStream; buf: Float32Array<ArrayBuffer> } | null>(null)
  const [spec, setSpec] = useState<number[]>([])
  const [peak, setPeak] = useState(0)
  const [freqs, setFreqs] = useState<number[]>([])
  const freqsRef = useRef<number[]>([])

  const start = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const ctx = new AudioContext()
      const src = ctx.createMediaStreamSource(stream)
      const an = ctx.createAnalyser()
      an.fftSize = 2048
      src.connect(an)
      audio.current = { ctx, an, stream, buf: new Float32Array(an.fftSize) }
      freqsRef.current = Array.from({ length: an.fftSize / 2 + 1 }, (_, k) => (k * ctx.sampleRate) / an.fftSize)
      setFreqs(freqsRef.current)
      setStatus('on')
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
      setStatus('error')
    }
  }
  const stop = () => {
    audio.current?.stream.getTracks().forEach((t) => t.stop())
    void audio.current?.ctx.close()
    audio.current = null
    setStatus('off')
  }
  useEffect(() => () => stop(), [])

  useAnimationLoop(() => {
    const a = audio.current
    if (!a) return
    a.an.getFloatTimeDomainData(a.buf)
    const s = amplitudeSpectrum(a.buf, 'hann')
    let k = 1
    for (let i = 2; i < s.length; i++) if (s[i] > s[k]) k = i
    setPeak(freqsRef.current[k] ?? 0)
    setSpec(Array.from(s, db))
  }, status === 'on')

  return (
    <Panel
      title="Live microphone spectrum (optional)"
      className="mt-4"
      actions={
        status === 'on' ? (
          <button type="button" className="btn" onClick={stop}>
            Stop microphone
          </button>
        ) : (
          <button type="button" className="btn btn-primary" onClick={start}>
            Use microphone
          </button>
        )
      }
    >
      {status === 'on' ? (
        <div className="grid gap-4 md:grid-cols-[200px_minmax(0,1fr)]">
          <Readout label="Dominant frequency" value={`${fmt(peak, 4)} Hz`} />
          <Plot ariaLabel="Live microphone spectrum" height={200} xLabel="f (Hz)" yLabel="dB" yDomain={[-120, 0]} xDomain={[0, 5000]} series={[{ x: freqs, y: spec, color: C.aqua, label: 'mic' }]} />
        </div>
      ) : (
        <p className="text-sm text-slate-400">
          {status === 'error' ? `Microphone unavailable: ${err}` : 'Audio is captured with the Web Audio API and transformed with this page’s own FFT (2048-point, Hann window). Nothing is recorded or sent anywhere.'}
        </p>
      )}
    </Panel>
  )
}
