import { useEffect, useRef, useState } from 'react'
import { Plot, type Series } from '../components/Plot.tsx'
import { Tex } from '../components/Tex.tsx'
import { Check, MethodSection, PageHeader, Panel, Readout, Select, Slider } from '../components/ui.tsx'
import { fmt, fmtFixed } from '../lib/format.ts'
import { fitCanvas, useAnimationLoop, useElementWidth } from '../lib/hooks.ts'
import { C } from '../lib/palette.ts'
import { useStreamingWorker } from '../lib/worker.ts'
import { scalingSlopes, type BenchPoint } from '../numerics/benchmark.ts'
import { cloneBodies, directAccel, energy, makePreset, momentum, NBodySim, PRESET_NAMES, type ForceMethod, type Preset, type PresetId } from '../numerics/nbody.ts'

const DEFAULT_N: Record<PresetId, number> = { galaxies: 2000, figure8: 3, solar: 600, collapse: 1500 }

interface Stats {
  t: number
  forceMs: number
  otherMs: number
  interactions: number
  dE: number
  P: number
  dL: number
  err: number
}

export default function NBody() {
  const [presetId, setPresetId] = useState<PresetId>('galaxies')
  const [n, setN] = useState(DEFAULT_N.galaxies)
  const [method, setMethod] = useState<ForceMethod>('barnes-hut')
  const [theta, setTheta] = useState(0.5)
  const [quadrupole, setQuadrupole] = useState(true)
  const [epsScale, setEpsScale] = useState(1)
  const [speed, setSpeed] = useState(2)
  const [showTree, setShowTree] = useState(false)
  const [running, setRunning] = useState(true)
  const [zoom, setZoom] = useState(1)
  const [seed, setSeed] = useState(0)
  const [stats, setStats] = useState<Stats>({ t: 0, forceMs: 0, otherMs: 0, interactions: 0, dE: 0, P: 0, dL: 0, err: 0 })

  const wrapRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const width = useElementWidth(wrapRef)
  const height = Math.min(560, Math.max(320, width * 0.62))
  const simRef = useRef<{ sim: NBodySim; preset: Preset; E0: number; L0: number; frame: number } | null>(null)
  const lastStats = useRef(0)

  useEffect(() => {
    const preset = makePreset(presetId, presetId === 'figure8' ? undefined : n)
    const sim = new NBodySim(cloneBodies(preset.bodies), { G: preset.params.G, eps: preset.params.eps, theta: 0.5, method: 'barnes-hut', quadrupole: true })
    const e = energy(sim.b, preset.params.G, preset.params.eps).total
    simRef.current = { sim, preset, E0: e, L0: momentum(sim.b).L, frame: 0 }
  }, [presetId, n, seed])

  // Live parameter changes apply without resetting the bodies.
  useEffect(() => {
    const s = simRef.current
    if (!s) return
    s.sim.p.method = method
    s.sim.p.theta = theta
    s.sim.p.quadrupole = quadrupole
    s.sim.p.eps = s.preset.params.eps * epsScale
    s.E0 = energy(s.sim.b, s.sim.p.G, s.sim.p.eps).total
    s.L0 = momentum(s.sim.b).L
    s.sim.computeAccel()
  }, [method, theta, quadrupole, epsScale, presetId, n, seed])

  const draw = () => {
    const s = simRef.current
    const canvas = canvasRef.current
    if (!s || !canvas || width < 50) return
    const ctx = fitCanvas(canvas, width, height)
    if (!ctx) return
    ctx.globalCompositeOperation = 'source-over'
    ctx.fillStyle = '#020617'
    ctx.fillRect(0, 0, width, height)
    const view = s.preset.view / zoom
    const sc = Math.min(width, height) / (2 * view)
    const X = (x: number) => width / 2 + x * sc
    const Y = (y: number) => height / 2 - y * sc
    const b = s.sim.b
    if (showTree && s.sim.p.method === 'barnes-hut') {
      const tr = s.sim.tree
      ctx.strokeStyle = 'rgba(57,135,229,0.22)'
      ctx.lineWidth = 0.6
      ctx.beginPath()
      for (let k = 0; k < tr.count; k++) {
        if (tr.child[k] < 0 && tr.nBodies[k] === 0) continue
        const h = tr.half[k]
        ctx.rect(X(tr.cx[k] - h), Y(tr.cy[k] + h), 2 * h * sc, 2 * h * sc)
      }
      ctx.stroke()
    }
    ctx.globalCompositeOperation = 'lighter'
    const heavy = new Set(s.preset.heavy)
    const small = b.n > 200
    ctx.fillStyle = small ? 'rgba(144, 180, 255, 0.8)' : C.blue
    for (let i = 0; i < b.n; i++) {
      if (heavy.has(i)) continue
      const x = X(b.x[i])
      const y = Y(b.y[i])
      if (x < -2 || y < -2 || x > width + 2 || y > height + 2) continue
      if (small) ctx.fillRect(x - 1, y - 1, 2, 2)
      else {
        ctx.beginPath()
        ctx.arc(x, y, 3, 0, 2 * Math.PI)
        ctx.fill()
      }
    }
    ctx.globalCompositeOperation = 'source-over'
    const colors = [C.yellow, C.orange, C.aqua, C.blue, C.red, C.violet, C.magenta]
    s.preset.heavy.forEach((i, k) => {
      ctx.fillStyle = s.preset.id === 'solar' && k === 0 ? '#fde68a' : colors[k % colors.length]
      ctx.beginPath()
      ctx.arc(X(b.x[i]), Y(b.y[i]), s.preset.id === 'solar' ? (k === 0 ? 7 : 3.5) : 5, 0, 2 * Math.PI)
      ctx.fill()
    })
  }

  useAnimationLoop(() => {
    const s = simRef.current
    if (!s) return
    const t0 = performance.now()
    let forceMs = 0
    const stepsPerFrame = s.preset.id === 'figure8' ? 8 * speed : speed
    for (let k = 0; k < stepsPerFrame; k++) {
      s.sim.step(s.preset.dt)
      forceMs += s.sim.lastForceMs
    }
    const total = performance.now() - t0
    s.frame++
    draw()
    const now = performance.now()
    if (now - lastStats.current > 400) {
      lastStats.current = now
      const b = s.sim.b
      const E = b.n <= 4000 ? energy(b, s.sim.p.G, s.sim.p.eps).total : NaN
      const mom = momentum(b)
      // Force error of the current method vs direct summation, sampled on a copy.
      let err = 0
      if (s.sim.p.method === 'barnes-hut' && b.n <= 4000 && b.n > 3) {
        const d = cloneBodies(b)
        directAccel(d, s.sim.p.G, s.sim.p.eps)
        let ms = 0
        let es = 0
        for (let i = 0; i < b.n; i++) {
          ms += d.ax[i] ** 2 + d.ay[i] ** 2
          es += (b.ax[i] - d.ax[i]) ** 2 + (b.ay[i] - d.ay[i]) ** 2
        }
        err = Math.sqrt(es / ms)
      }
      setStats({
        t: s.sim.t,
        forceMs: forceMs / stepsPerFrame,
        otherMs: (total - forceMs) / stepsPerFrame,
        interactions: s.sim.p.method === 'barnes-hut' ? s.sim.tree.interactions : (b.n * (b.n - 1)) / 2,
        dE: Math.abs((E - s.E0) / s.E0),
        P: Math.hypot(mom.px, mom.py),
        dL: Math.abs((mom.L - s.L0) / (s.L0 || 1)),
        err,
      })
    }
  }, running)

  useEffect(draw)

  const [bench, setBench] = useState<BenchPoint[]>([])
  const [benchBusy, setBenchBusy] = useState(false)
  const resumeAfterBench = useRef(false)
  const worker = useStreamingWorker<{ ns: number[]; theta: number }, { type: 'point'; point: BenchPoint } | { type: 'done' }>(
    () => new Worker(new URL('../workers/nbody.worker.ts', import.meta.url), { type: 'module' }),
    (m) => {
      if (m.type === 'point') setBench((b) => [...b, m.point])
      else {
        setBenchBusy(false)
        setRunning(resumeAfterBench.current)
      }
    },
  )
  const runBench = () => {
    // Pause the live simulation so it does not compete with the worker for CPU time.
    resumeAfterBench.current = running
    setRunning(false)
    setBench([])
    setBenchBusy(true)
    worker.start({ ns: [250, 500, 1000, 2000, 4000, 8000, 16000], theta })
  }
  const slopes = bench.length >= 3 ? scalingSlopes(bench) : null
  const benchSeries: Series[] = [
    { x: bench.map((p) => p.n), y: bench.map((p) => p.directMs), color: C.red, label: `Direct O(N²)${slopes ? `  slope ${slopes.direct.toFixed(2)}` : ''}`, points: true },
    { x: bench.map((p) => p.n), y: bench.map((p) => p.bhMs), color: C.aqua, label: `Barnes–Hut monopole${slopes ? `  slope ${slopes.bh.toFixed(2)}` : ''}`, points: true },
    { x: bench.map((p) => p.n), y: bench.map((p) => p.bhQuadMs), color: C.blue, label: 'Barnes–Hut + quadrupole', points: true, dash: [5, 3] },
  ]
  const errSeries: Series[] = [
    { x: bench.map((p) => p.n), y: bench.map((p) => p.errMono), color: C.aqua, label: 'monopole', points: true },
    { x: bench.map((p) => p.n), y: bench.map((p) => p.errQuad), color: C.blue, label: 'quadrupole', points: true },
  ]

  return (
    <div>
      <PageHeader eyebrow="Module 3 · Gravitational N-body" title="N-body gravity">
        Direct summation costs O(N²) per step. A Barnes–Hut quadtree replaces distant groups of bodies by their multipole expansion, cutting the cost to O(N log N) at a
        controllable accuracy. Leapfrog (kick–drift–kick) integration keeps momentum exact and the energy error bounded.
      </PageHeader>
      <div className="grid gap-4 lg:grid-cols-[280px_minmax(0,1fr)]">
        <Panel title="Simulation">
          <div className="space-y-4">
            <Select<PresetId>
              label="Preset"
              value={presetId}
              onChange={(v) => {
                setPresetId(v)
                setN(DEFAULT_N[v])
                setZoom(1)
              }}
              options={(Object.keys(PRESET_NAMES) as PresetId[]).map((id) => ({ value: id, label: PRESET_NAMES[id] }))}
            />
            {presetId !== 'figure8' && <Slider label="Bodies N" value={n} min={100} max={6000} step={100} onChange={setN} />}
            <Select<ForceMethod>
              label="Force calculation"
              value={method}
              onChange={setMethod}
              options={[
                { value: 'barnes-hut', label: 'Barnes–Hut quadtree' },
                { value: 'direct', label: 'Direct O(N²) summation' },
              ]}
            />
            <Slider label="Opening angle θ" value={theta} min={0.1} max={1.2} step={0.05} onChange={setTheta} disabled={method === 'direct'} format={(v) => v.toFixed(2)} />
            <Check label="Quadrupole moments" checked={quadrupole} onChange={setQuadrupole} />
            <Check label="Show quadtree" checked={showTree} onChange={setShowTree} />
            <Slider label="Softening ε (× preset)" value={epsScale} min={0} max={4} step={0.25} onChange={setEpsScale} format={(v) => `${v.toFixed(2)}×`} />
            <Slider label="Steps per frame" value={speed} min={1} max={8} step={1} onChange={setSpeed} />
            <Slider label="Zoom" value={zoom} min={0.3} max={4} step={0.1} onChange={setZoom} format={(v) => `${v.toFixed(1)}×`} />
            <div className="flex gap-2">
              <button type="button" className="btn flex-1" onClick={() => setRunning((r) => !r)}>
                {running ? 'Pause' : 'Run'}
              </button>
              <button type="button" className="btn flex-1" onClick={() => setSeed((s) => s + 1)}>
                Reset
              </button>
            </div>
          </div>
        </Panel>
        <div className="min-w-0 space-y-4">
          <Panel>
            <div ref={wrapRef}>
              <canvas ref={canvasRef} className="sim-canvas block rounded-xl" aria-label="N-body simulation" />
            </div>
            <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-4">
              <Readout label="Time t" value={fmtFixed(stats.t, 2)} />
              <Readout label="Force eval (ms)" value={fmtFixed(stats.forceMs, 2)} />
              <Readout label="Interactions / step" value={stats.interactions.toLocaleString()} />
              <Readout label="Force error vs direct" value={method === 'direct' ? 'exact' : fmt(stats.err)} tone={stats.err < 0.01 ? 'good' : 'warn'} />
              <Readout label="|ΔE / E₀|" value={fmt(stats.dE)} tone={stats.dE < 1e-2 ? 'good' : 'warn'} />
              <Readout label="|P| total momentum" value={fmt(stats.P)} />
              <Readout label="|ΔL / L₀|" value={fmt(stats.dL)} />
              <Readout label="Integrator" value="Leapfrog KDK" />
            </div>
            <p className="mt-2 text-xs text-slate-500">
              With direct summation, Newton’s third law makes total momentum exact to round-off. Barnes–Hut forces are not pairwise antisymmetric, so momentum drifts at the
              level of the force error — a classic trade-off.
            </p>
          </Panel>
          <Panel
            title="Scaling benchmark (one force evaluation, uniform disk)"
            actions={
              <button type="button" className="btn btn-primary" disabled={benchBusy} onClick={runBench}>
                {benchBusy ? `Running in a worker… (${bench.length}/7)` : 'Run benchmark'}
              </button>
            }
          >
            {bench.length ? (
              <div className="grid gap-4 md:grid-cols-2">
                <Plot ariaLabel="Time per force evaluation vs N" series={benchSeries} xLog yLog height={260} xLabel="N" yLabel="ms / evaluation" />
                <Plot ariaLabel="Force error vs N" series={errSeries} xLog yLog height={260} xLabel="N" yLabel="RMS force error" />
              </div>
            ) : (
              <p className="py-6 text-center text-sm text-slate-400">Times direct summation vs Barnes–Hut for N = 250 … 16 000 at the current θ.</p>
            )}
          </Panel>
        </div>
      </div>
      <MethodSection
        references={[
          { text: 'J. Barnes, P. Hut — A hierarchical O(N log N) force-calculation algorithm, Nature 324 (1986) 446–449.' },
          { text: 'L. Hernquist — Performance characteristics of tree codes, ApJS 64 (1987) 715–734.' },
          { text: 'A. Chenciner, R. Montgomery — A remarkable periodic solution of the three-body problem in the case of equal masses, Ann. Math. 152 (2000) 881–901.' },
          { text: 'J. Binney, S. Tremaine — Galactic Dynamics, 2nd ed., Princeton (2008).' },
        ]}
      >
        <p>Each body feels the Plummer-softened gravity of all others (softening ε removes the 1/r² singularity of close encounters):</p>
        <Tex block>{String.raw`\mathbf a_i = G \sum_{j \ne i} m_j \frac{\mathbf r_j - \mathbf r_i}{\left(|\mathbf r_j - \mathbf r_i|^2 + \varepsilon^2\right)^{3/2}},\qquad E = \sum_i \tfrac12 m_i v_i^2 - \sum_{i<j} \frac{G m_i m_j}{\sqrt{r_{ij}^2 + \varepsilon^2}}`}</Tex>
        <p>
          <strong>Barnes–Hut.</strong> Bodies are inserted into a quadtree whose nodes accumulate mass <Tex>M</Tex>, centre of mass and second moments. Walking the tree from the
          root, a node of width <Tex>s</Tex> at distance <Tex>d</Tex> from the body is accepted when <Tex>{String.raw`s/d < \theta`}</Tex>; its field is the multipole expansion
        </p>
        <Tex block>{String.raw`\mathbf a \approx -\frac{GM\,\mathbf r}{r^3} + G\left(\frac{Q\mathbf r}{r^5} - \frac{5}{2}\frac{(\mathbf r^{\!\top} Q\, \mathbf r)\,\mathbf r}{r^7}\right),\qquad Q_{ij} = \sum_k m_k\left(3 x_i x_j - r^2 \delta_{ij}\right)`}</Tex>
        <p>
          Expanding about the centre of mass kills the dipole, so the monopole error falls like <Tex>{String.raw`\theta^2`}</Tex> and the quadrupole-corrected error like{' '}
          <Tex>{String.raw`\theta^3`}</Tex>; the test suite measures slopes 2.55 and 3.28 and a θ = 0.5 RMS error of 1.4% (monopole) vs 0.24% (quadrupole). Otherwise the node is
          opened. The tree lives in flat typed arrays with an explicit traversal stack — no allocation per step.
        </p>
        <p>
          <strong>Leapfrog</strong> <Tex>{String.raw`\mathbf v \mathrel{+}= \tfrac{\Delta t}{2}\mathbf a,\ \mathbf x \mathrel{+}= \Delta t\,\mathbf v,\ \mathbf v \mathrel{+}= \tfrac{\Delta t}{2}\mathbf a`}</Tex>{' '}
          is symplectic and time-reversible. The figure-eight choreography (Chenciner–Montgomery) returns to its initial state after <Tex>T \approx 6.3259</Tex> to 2×10⁻⁷ in the tests.
        </p>
      </MethodSection>
    </div>
  )
}
