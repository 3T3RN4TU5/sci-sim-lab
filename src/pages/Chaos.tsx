import { useEffect, useMemo, useRef, useState, type PointerEvent } from 'react'
import { Plot, type Series } from '../components/Plot.tsx'
import { Tex } from '../components/Tex.tsx'
import { MethodSection, PageHeader, Panel, Readout, Slider } from '../components/ui.tsx'
import { fmt, fmtFixed } from '../lib/format.ts'
import { fitCanvas, useAnimationLoop, useElementWidth } from '../lib/hooks.ts'
import { C } from '../lib/palette.ts'
import { bifurcation, LyapunovEstimator, logisticLyapunov } from '../numerics/chaos.ts'
import { colormapLUT } from '../numerics/colormaps.ts'
import { makeStepper } from '../numerics/ode.ts'
import { doublePendulum, lorenz } from '../numerics/systems.ts'

export default function Chaos() {
  return (
    <div>
      <PageHeader eyebrow="Module 2 · Nonlinear dynamics" title="Chaos">
        Deterministic systems whose nearby trajectories separate exponentially. The largest Lyapunov exponent λ₁ measures that rate; it is estimated live with Benettin’s
        algorithm. Drag the Lorenz attractor to rotate it.
      </PageHeader>
      <LorenzPanel />
      <div className="mt-4 grid gap-4 xl:grid-cols-2">
        <DoublePendulumPanel />
        <LogisticPanel />
      </div>
      <MethodSection
        references={[
          { text: 'E. N. Lorenz — Deterministic nonperiodic flow, J. Atmos. Sci. 20 (1963) 130–141.' },
          { text: 'G. Benettin, L. Galgani, A. Giorgilli, J.-M. Strelcyn — Lyapunov characteristic exponents for smooth dynamical systems, Meccanica 15 (1980) 9–30.' },
          { text: 'A. Wolf, J. B. Swift, H. L. Swinney, J. A. Vastano — Determining Lyapunov exponents from a time series, Physica D 16 (1985) 285–317.' },
          { text: 'R. M. May — Simple mathematical models with very complicated dynamics, Nature 261 (1976) 459–467.' },
          { text: 'S. H. Strogatz — Nonlinear Dynamics and Chaos, 2nd ed., Westview (2015).' },
        ]}
      >
        <p>The Lorenz system, integrated with classic RK4 (h = 0.005):</p>
        <Tex block>{String.raw`\dot x = \sigma (y - x),\qquad \dot y = x(\rho - z) - y,\qquad \dot z = xy - \beta z`}</Tex>
        <p>
          <strong>Benettin’s method.</strong> Integrate a reference trajectory <Tex>{String.raw`y(t)`}</Tex> and a shadow <Tex>{String.raw`\tilde y(t)`}</Tex> with{' '}
          <Tex>{String.raw`\|\tilde y(0) - y(0)\| = d_0 = 10^{-8}`}</Tex>. Every <Tex>k</Tex> steps measure the separation <Tex>{String.raw`d_i`}</Tex>, then rescale the shadow back to
          distance <Tex>{String.raw`d_0`}</Tex> along the same direction. The direction aligns with the most unstable one, and
        </p>
        <Tex block>{String.raw`\lambda_1 = \lim_{t\to\infty} \frac{1}{t} \sum_i \ln \frac{d_i}{d_0} \approx 0.906 \quad (\sigma = 10,\ \rho = 28,\ \beta = 8/3)`}</Tex>
        <p>
          The test suite checks the estimate over <Tex>t = 1000</Tex> against the literature value 0.9056. For the logistic map{' '}
          <Tex>{String.raw`x_{n+1} = r x_n (1 - x_n)`}</Tex> the exponent follows from the derivative along the orbit,{' '}
          <Tex>{String.raw`\lambda(r) = \lim \frac1N \sum \ln |r(1 - 2x_n)|`}</Tex>; it is exactly <Tex>{String.raw`\ln 2`}</Tex> at <Tex>r = 4</Tex>, negative in periodic windows and
          zero at each period-doubling bifurcation (Feigenbaum cascade, <Tex>{String.raw`\delta \approx 4.669`}</Tex>).
        </p>
        <p>
          The double pendulum uses the Lagrangian equations of motion for point masses on massless rods; with no dissipation its energy is conserved, which serves as a live check
          on the integrator (RK4, h = 1 ms).
        </p>
      </MethodSection>
    </div>
  )
}

// ─── Lorenz ────────────────────────────────────────────────────────────────

const TRAIL = 5000

function LorenzPanel() {
  const [sigma, setSigma] = useState(10)
  const [rho, setRho] = useState(28)
  const [beta, setBeta] = useState(8 / 3)
  const [running, setRunning] = useState(true)
  const wrapRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const width = useElementWidth(wrapRef)
  const height = Math.min(460, Math.max(300, width * 0.6))
  const view = useRef({ yaw: 0.6, pitch: -0.35, drag: null as null | { x: number; y: number; yaw: number; pitch: number } })

  const sim = useRef<{ est: LyapunovEstimator; twin: Float64Array; twinStep: ReturnType<typeof makeStepper>; trailA: Float32Array; trailB: Float32Array; head: number; count: number; sep: number[]; sepT: number[] } | null>(null)
  const [stats, setStats] = useState({ lambda: 0, t: 0, sep: 1e-9, sepT: [] as number[], sepY: [] as number[] })

  useEffect(() => {
    const p = lorenz(sigma, rho, beta)
    const est = new LyapunovEstimator(p, [1, 1, 1], { h: 0.005, renormEvery: 10 })
    const twin = Float64Array.of(1 + 1e-9, 1, 1)
    sim.current = { est, twin, twinStep: makeStepper('rk4', p), trailA: new Float32Array(TRAIL * 3), trailB: new Float32Array(TRAIL * 3), head: 0, count: 0, sep: [], sepT: [] }
  }, [sigma, rho, beta])

  const draw = () => {
    const s = sim.current
    const canvas = canvasRef.current
    if (!s || !canvas || width < 50) return
    const ctx = fitCanvas(canvas, width, height)
    if (!ctx) return
    ctx.fillStyle = '#020617'
    ctx.fillRect(0, 0, width, height)
    const { yaw, pitch } = view.current
    const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch)
    const scale = Math.min(width, height) / 62
    const project = (x: number, y: number, z: number): [number, number, number] => {
      const zc = z - rho + 3
      const x1 = cy * x - sy * y
      const y1 = sy * x + cy * y
      const y2 = cp * y1 - sp * zc
      const z2 = sp * y1 + cp * zc
      return [width / 2 + x1 * scale, height / 2 - z2 * scale, y2]
    }
    // Axes gizmo.
    ctx.lineWidth = 1
    const axes: [number, number, number, string, string][] = [
      [12, 0, 0, C.red, 'x'],
      [0, 12, 0, C.aqua, 'y'],
      [0, 0, 12, C.blue, 'z'],
    ]
    const o = project(0, 0, rho - 3)
    for (const [x, y, z, col, name] of axes) {
      const q = project(x, y, z + rho - 3)
      ctx.strokeStyle = col
      ctx.beginPath()
      ctx.moveTo(o[0], o[1])
      ctx.lineTo(q[0], q[1])
      ctx.stroke()
      ctx.fillStyle = '#cbd5e1'
      ctx.font = '11px Inter, sans-serif'
      ctx.fillText(name, q[0] + 3, q[1] - 3)
    }
    const drawTrail = (buf: Float32Array, color: [number, number, number]) => {
      const n = s.count
      ctx.lineWidth = 1.2
      const seg = 50
      for (let start = 0; start < n - 1; start += seg) {
        const alpha = 0.15 + 0.85 * (start / n)
        ctx.strokeStyle = `rgba(${color[0]},${color[1]},${color[2]},${alpha.toFixed(3)})`
        ctx.beginPath()
        for (let k = start; k <= Math.min(n - 1, start + seg); k++) {
          const idx = ((s.head - n + k + TRAIL) % TRAIL) * 3
          const p = project(buf[idx], buf[idx + 1], buf[idx + 2])
          if (k === start) ctx.moveTo(p[0], p[1])
          else ctx.lineTo(p[0], p[1])
        }
        ctx.stroke()
      }
      if (n > 0) {
        const idx = ((s.head - 1 + TRAIL) % TRAIL) * 3
        const p = project(buf[idx], buf[idx + 1], buf[idx + 2])
        ctx.fillStyle = `rgb(${color[0]},${color[1]},${color[2]})`
        ctx.beginPath()
        ctx.arc(p[0], p[1], 4, 0, 2 * Math.PI)
        ctx.fill()
      }
    }
    drawTrail(s.trailA, [57, 135, 229])
    drawTrail(s.trailB, [217, 89, 38])
  }

  const lastStats = useRef(0)
  useAnimationLoop(() => {
    const s = sim.current
    if (!s) return
    for (let k = 0; k < 4; k++) {
      s.est.advance(1)
      s.twinStep(s.est.t, s.twin, s.est.h)
      const i = s.head * 3
      s.trailA[i] = s.est.y[0]
      s.trailA[i + 1] = s.est.y[1]
      s.trailA[i + 2] = s.est.y[2]
      s.trailB[i] = s.twin[0]
      s.trailB[i + 1] = s.twin[1]
      s.trailB[i + 2] = s.twin[2]
      s.head = (s.head + 1) % TRAIL
      s.count = Math.min(TRAIL, s.count + 1)
    }
    // Extra Lyapunov-only steps so the estimate converges faster than the animation.
    s.est.advance(40)
    const d = Math.hypot(s.twin[0] - s.est.y[0], s.twin[1] - s.est.y[1], s.twin[2] - s.est.y[2])
    draw()
    const now = performance.now()
    if (now - lastStats.current > 200) {
      lastStats.current = now
      s.sepT.push(s.est.t)
      s.sep.push(d)
      setStats({ lambda: s.est.exponent, t: s.est.t, sep: d, sepT: s.sepT.slice(), sepY: s.sep.slice() })
    }
  }, running)

  useEffect(draw)

  const onDown = (e: PointerEvent<HTMLCanvasElement>) => {
    e.currentTarget.setPointerCapture(e.pointerId)
    view.current.drag = { x: e.clientX, y: e.clientY, yaw: view.current.yaw, pitch: view.current.pitch }
  }
  const onMove = (e: PointerEvent<HTMLCanvasElement>) => {
    const d = view.current.drag
    if (!d) return
    view.current.yaw = d.yaw + (e.clientX - d.x) * 0.01
    view.current.pitch = Math.max(-1.5, Math.min(1.5, d.pitch + (e.clientY - d.y) * 0.01))
    if (!running) draw()
  }

  const sepSeries: Series[] = useMemo(() => {
    const s: Series[] = [{ x: stats.sepT, y: stats.sepY, color: C.orange, label: 'separation |δ(t)|', width: 2 }]
    if (stats.sepT.length > 1) {
      const tt = [0, stats.sepT[stats.sepT.length - 1]]
      s.push({ x: tt, y: tt.map((t) => 1e-9 * Math.exp(0.9056 * t)), color: C.muted, dash: [5, 4], label: 'e^{λ₁t}, λ₁ = 0.906', width: 1.5 })
    }
    return s
  }, [stats.sepT, stats.sepY])

  return (
    <Panel title="Lorenz attractor · two trajectories started 10⁻⁹ apart">
      <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_320px]">
        <div ref={wrapRef} className="min-w-0">
          <canvas
            ref={canvasRef}
            className="sim-canvas block cursor-grab rounded-xl active:cursor-grabbing"
            aria-label="Lorenz attractor, drag to rotate"
            onPointerDown={onDown}
            onPointerMove={onMove}
            onPointerUp={() => (view.current.drag = null)}
          />
        </div>
        <div className="space-y-3">
          <div className="grid grid-cols-2 gap-2">
            <Readout label="λ₁ (Benettin)" value={fmtFixed(stats.lambda, 4)} tone={rho === 28 && sigma === 10 && Math.abs(stats.lambda - 0.906) < 0.05 ? 'good' : undefined} />
            <Readout label="Time t" value={fmtFixed(stats.t, 1)} />
            <Readout label="|δ| blue vs orange" value={fmt(stats.sep)} />
            <Readout label="Literature λ₁" value="0.9056" />
          </div>
          <Slider label="σ" value={sigma} min={1} max={20} step={0.5} onChange={setSigma} />
          <Slider label="ρ" value={rho} min={0.5} max={50} step={0.5} onChange={setRho} hint={rho < 24.74 ? 'ρ < 24.74: the fixed points C± are stable — chaos dies out (λ₁ → ≤ 0).' : undefined} />
          <Slider label="β" value={beta} min={0.5} max={5} step={0.05} onChange={setBeta} format={(v) => v.toFixed(3)} />
          <button type="button" className="btn w-full" onClick={() => setRunning((r) => !r)}>
            {running ? 'Pause' : 'Resume'}
          </button>
        </div>
      </div>
      <div className="mt-3">
        <Plot ariaLabel="Separation of the twin trajectories" series={sepSeries} yLog height={200} xLabel="t" yLabel="|δ|" yDomain={[1e-10, 100]} />
      </div>
    </Panel>
  )
}

// ─── Double pendulum ───────────────────────────────────────────────────────

function DoublePendulumPanel() {
  const [delta, setDelta] = useState(-6)
  const [running, setRunning] = useState(true)
  const wrapRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const width = useElementWidth(wrapRef)
  const height = Math.min(380, Math.max(260, width * 0.75))
  const p = useMemo(() => doublePendulum(), [])
  const sim = useRef<{ a: Float64Array; b: Float64Array; t: number; trailA: number[]; trailB: number[]; E0: number; sepT: number[]; sep: number[] } | null>(null)
  const stepA = useMemo(() => makeStepper('rk4', p), [p])
  const stepB = useMemo(() => makeStepper('rk4', p), [p])
  const [stats, setStats] = useState({ t: 0, dE: 0, sepT: [] as number[], sep: [] as number[] })

  useEffect(() => {
    const a = Float64Array.from(p.y0)
    const b = Float64Array.from(p.y0)
    b[1] += 10 ** delta
    sim.current = { a, b, t: 0, trailA: [], trailB: [], E0: p.energy!(a), sepT: [], sep: [] }
  }, [delta, p])

  const last = useRef(0)
  useAnimationLoop(() => {
    const s = sim.current
    const canvas = canvasRef.current
    if (!s || !canvas || width < 50) return
    const h = 0.001
    for (let k = 0; k < 16; k++) {
      stepA(s.t, s.a, h)
      stepB(s.t, s.b, h)
      s.t += h
    }
    const pa = p.positions(s.a)
    const pb = p.positions(s.b)
    s.trailA.push(pa[2], pa[3])
    s.trailB.push(pb[2], pb[3])
    if (s.trailA.length > 1200) {
      s.trailA.splice(0, 2)
      s.trailB.splice(0, 2)
    }
    const ctx = fitCanvas(canvas, width, height)
    if (!ctx) return
    ctx.fillStyle = '#020617'
    ctx.fillRect(0, 0, width, height)
    const sc = Math.min(width, height) / 4.6
    const ox = width / 2
    const oy = height / 2
    const X = (x: number) => ox + x * sc
    const Y = (y: number) => oy - y * sc
    const trail = (tr: number[], rgb: string) => {
      for (let i = 2; i < tr.length; i += 2) {
        ctx.strokeStyle = `rgba(${rgb},${(0.1 + (0.8 * i) / tr.length).toFixed(3)})`
        ctx.lineWidth = 1.5
        ctx.beginPath()
        ctx.moveTo(X(tr[i - 2]), Y(tr[i - 1]))
        ctx.lineTo(X(tr[i]), Y(tr[i + 1]))
        ctx.stroke()
      }
    }
    trail(s.trailA, '57,135,229')
    trail(s.trailB, '217,89,38')
    const arm = (q: [number, number, number, number], col: string) => {
      ctx.strokeStyle = '#cbd5e1'
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.moveTo(ox, oy)
      ctx.lineTo(X(q[0]), Y(q[1]))
      ctx.lineTo(X(q[2]), Y(q[3]))
      ctx.stroke()
      ctx.fillStyle = col
      for (const [x, y] of [
        [q[0], q[1]],
        [q[2], q[3]],
      ]) {
        ctx.beginPath()
        ctx.arc(X(x), Y(y), 7, 0, 2 * Math.PI)
        ctx.fill()
      }
    }
    arm(pa, C.blue)
    arm(pb, C.orange)
    ctx.fillStyle = '#94a3b8'
    ctx.beginPath()
    ctx.arc(ox, oy, 3, 0, 2 * Math.PI)
    ctx.fill()
    const now = performance.now()
    if (now - last.current > 200) {
      last.current = now
      s.sepT.push(s.t)
      s.sep.push(Math.max(1e-16, Math.hypot(s.a[0] - s.b[0], s.a[1] - s.b[1], 0.1 * (s.a[2] - s.b[2]), 0.1 * (s.a[3] - s.b[3]))))
      setStats({ t: s.t, dE: Math.abs((p.energy!(s.a) - s.E0) / s.E0), sepT: s.sepT.slice(), sep: s.sep.slice() })
    }
  }, running)

  return (
    <Panel title="Double pendulum · sensitivity to initial conditions">
      <div ref={wrapRef}>
        <canvas ref={canvasRef} className="sim-canvas block rounded-xl" aria-label="Two double pendulums with nearly identical initial angles" />
      </div>
      <div className="mt-3 grid grid-cols-3 gap-2">
        <Readout label="Time t (s)" value={fmtFixed(stats.t, 2)} />
        <Readout label="|ΔE/E₀| (RK4)" value={fmt(stats.dE)} tone={stats.dE < 1e-4 ? 'good' : 'warn'} />
        <Readout label="Separation" value={fmt(stats.sep[stats.sep.length - 1] ?? 0)} />
      </div>
      <div className="mt-3 space-y-3">
        <Slider label="Initial offset δθ₂ (rad)" value={delta} min={-12} max={-1} step={1} onChange={setDelta} format={(v) => `1e${v}`} />
        <Plot ariaLabel="Pendulum separation" series={[{ x: stats.sepT, y: stats.sep, color: C.orange, label: 'state separation' }]} yLog height={170} xLabel="t (s)" yLabel="|δ|" />
        <button type="button" className="btn w-full" onClick={() => setRunning((r) => !r)}>
          {running ? 'Pause' : 'Resume'}
        </button>
      </div>
    </Panel>
  )
}

// ─── Logistic map ──────────────────────────────────────────────────────────

function LogisticPanel() {
  const [rMin, setRMin] = useState(2.8)
  const [rMax, setRMax] = useState(4)
  const wrapRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const width = useElementWidth(wrapRef)
  const height = 300
  const lo = Math.min(rMin, rMax - 0.01)
  const hi = Math.max(rMax, lo + 0.01)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || width < 50) return
    const ctx = fitCanvas(canvas, width, height)
    if (!ctx) return
    const W = Math.round(width)
    const H = height
    const keep = 400
    const pts = bifurcation(lo, hi, W, 800, keep)
    const counts = new Float32Array(W * H)
    let max = 0
    for (let c = 0; c < W; c++) {
      for (let k = 0; k < keep; k++) {
        const row = Math.floor((1 - pts[c * keep + k]) * (H - 1))
        if (row < 0 || row >= H) continue
        const v = ++counts[row * W + c]
        if (v > max) max = v
      }
    }
    const img = ctx.createImageData(W, H)
    const lut = colormapLUT('inferno')
    for (let i = 0; i < W * H; i++) {
      const v = counts[i] > 0 ? Math.min(255, Math.round(60 + 195 * (Math.log(1 + counts[i]) / Math.log(1 + max)))) : 0
      img.data[i * 4] = lut[v * 3]
      img.data[i * 4 + 1] = lut[v * 3 + 1]
      img.data[i * 4 + 2] = lut[v * 3 + 2]
      img.data[i * 4 + 3] = 255
    }
    const off = document.createElement('canvas')
    off.width = W
    off.height = H
    off.getContext('2d')!.putImageData(img, 0, 0)
    ctx.imageSmoothingEnabled = false
    ctx.drawImage(off, 0, 0, width, height)
  }, [lo, hi, width])

  const lyap = useMemo(() => {
    const n = 400
    const r: number[] = []
    const l: number[] = []
    for (let i = 0; i < n; i++) {
      const rr = lo + ((hi - lo) * i) / (n - 1)
      r.push(rr)
      l.push(logisticLyapunov(rr, 1500, 300))
    }
    return { r, l }
  }, [lo, hi])

  return (
    <Panel title="Logistic map · bifurcation diagram and λ(r)">
      <div ref={wrapRef}>
        <canvas ref={canvasRef} className="block rounded-xl" aria-label="Bifurcation diagram of the logistic map" />
      </div>
      <div className="mt-1 flex justify-between font-mono text-[11px] text-slate-400">
        <span>r = {lo.toFixed(3)}</span>
        <span>x ∈ [0, 1] (vertical)</span>
        <span>r = {hi.toFixed(3)}</span>
      </div>
      <div className="mt-3">
        <Plot
          ariaLabel="Lyapunov exponent of the logistic map"
          series={[{ x: lyap.r, y: lyap.l, color: C.violet, label: 'λ(r)', width: 1.5 }]}
          height={170}
          xLabel="r"
          yLabel="λ"
          xDomain={[lo, hi]}
          yDomain={[-1.5, 1]}
          hLines={[{ y: 0, color: C.muted }]}
        />
      </div>
      <div className="mt-3 grid grid-cols-2 gap-3">
        <Slider label="r min" value={rMin} min={1} max={3.99} step={0.01} onChange={setRMin} format={(v) => v.toFixed(2)} />
        <Slider label="r max" value={rMax} min={1.01} max={4} step={0.005} onChange={setRMax} format={(v) => v.toFixed(3)} />
      </div>
      <p className="mt-2 text-xs text-slate-500">Tip: zoom to r ∈ [3.82, 3.86] to find the period-3 window and its own cascade of period doublings.</p>
    </Panel>
  )
}
