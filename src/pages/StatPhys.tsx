import { useEffect, useMemo, useRef, useState } from 'react'
import { Plot, type Series } from '../components/Plot.tsx'
import { Tex } from '../components/Tex.tsx'
import { MethodSection, PageHeader, Panel, Readout, Segmented, Select, Slider } from '../components/ui.tsx'
import { fmt, fmtFixed } from '../lib/format.ts'
import { fitCanvas, useAnimationLoop, useElementWidth } from '../lib/hooks.ts'
import { C } from '../lib/palette.ts'
import { useStreamingWorker } from '../lib/worker.ts'
import { Ising, onsagerEnergy, onsagerMagnetization, T_CRITICAL, type IsingAlgorithm, type SweepPoint } from '../numerics/ising.ts'
import { RandomWalkers, type ErrorScaling } from '../numerics/montecarlo.ts'
import { Rng } from '../numerics/rng.ts'
import type { StatRequest } from '../workers/statphys.worker.ts'

type StatMsg = { kind: 'point'; point: SweepPoint } | { kind: 'done' } | { kind: 'pi'; result: ErrorScaling }

const makeWorker = () => new Worker(new URL('../workers/statphys.worker.ts', import.meta.url), { type: 'module' })

export default function StatPhys() {
  return (
    <div>
      <PageHeader eyebrow="Module 5 · Statistical physics & Monte Carlo" title="Monte Carlo methods">
        Markov-chain Monte Carlo on the 2D Ising model reproduces Onsager’s exact phase transition at T_c = 2/ln(1+√2) ≈ 2.269. Plain Monte Carlo converges like N^(−1/2);
        quasi-random sequences do better. Random walkers diffuse with MSD ∝ t.
      </PageHeader>
      <IsingSection />
      <div className="mt-4 grid gap-4 xl:grid-cols-2">
        <PiSection />
        <WalkSection />
      </div>
      <MethodSection
        references={[
          { text: 'L. Onsager — Crystal statistics I. A two-dimensional model with an order-disorder transition, Phys. Rev. 65 (1944) 117–149.' },
          { text: 'C. N. Yang — The spontaneous magnetization of a two-dimensional Ising model, Phys. Rev. 85 (1952) 808.' },
          { text: 'N. Metropolis, A. W. Rosenbluth, M. N. Rosenbluth, A. H. Teller, E. Teller — Equation of state calculations by fast computing machines, J. Chem. Phys. 21 (1953) 1087.' },
          { text: 'U. Wolff — Collective Monte Carlo updating for spin systems, Phys. Rev. Lett. 62 (1989) 361–364.' },
          { text: 'M. E. J. Newman, G. T. Barkema — Monte Carlo Methods in Statistical Physics, Oxford (1999).' },
          { text: 'D. Blackman, S. Vigna — Scrambled linear pseudorandom number generators (xoshiro128**), ACM TOMS 47 (2021).' },
        ]}
      >
        <p>
          The Ising Hamiltonian <Tex>{String.raw`H = -J\sum_{\langle ij\rangle} s_i s_j`}</Tex> on an <Tex>L \times L</Tex> periodic lattice. <strong>Metropolis</strong> proposes a
          single spin flip with <Tex>{String.raw`\Delta E = 2 s_i \sum_{j \in \partial i} s_j \in \{-8, -4, 0, 4, 8\}`}</Tex> and accepts with probability{' '}
          <Tex>{String.raw`\min(1, e^{-\beta \Delta E})`}</Tex> (precomputed). Near <Tex>T_c</Tex> it suffers critical slowing down (<Tex>{String.raw`\tau \sim L^{2.17}`}</Tex>).{' '}
          <strong>Wolff</strong> grows a cluster of aligned spins, adding each aligned neighbour with <Tex>{String.raw`p = 1 - e^{-2\beta J}`}</Tex>, and flips it in one move (always
          accepted, detailed balance by construction; <Tex>{String.raw`z \approx 0.25`}</Tex>). Energy changes are computed incrementally from the cluster boundary, O(cluster size).
        </p>
        <p>Fluctuation estimators per spin, with exact results for the infinite lattice:</p>
        <Tex block>{String.raw`\chi = \beta N\big(\langle m^2\rangle - \langle |m| \rangle^2\big),\quad C = \beta^2 N\big(\langle e^2\rangle - \langle e\rangle^2\big),\quad m_0(T) = \big(1 - \sinh^{-4} 2\beta J\big)^{1/8}\ (T < T_c)`}</Tex>
        <Tex block>{String.raw`u(T) = -J\coth 2\beta J\Big[1 + \tfrac{2}{\pi}\big(2\tanh^2 2\beta J - 1\big)\,K(k)\Big],\qquad k = \frac{2\sinh 2\beta J}{\cosh^2 2\beta J}`}</Tex>
        <p>
          with <Tex>K</Tex> the complete elliptic integral of the first kind, evaluated via the arithmetic–geometric mean{' '}
          <Tex>{String.raw`K(k) = \pi / \big(2\,\mathrm{AGM}(1, \sqrt{1 - k^2})\big)`}</Tex> and cross-checked against <code>scipy.special.ellipk</code>.
        </p>
        <p>
          <strong>Monte Carlo integration.</strong> For i.i.d. samples the estimator’s standard error is <Tex>{String.raw`\sigma/\sqrt N`}</Tex> regardless of dimension. Mean-value
          sampling of <Tex>{String.raw`4\int_0^1\sqrt{1 - x^2}\,dx`}</Tex> has smaller <Tex>\sigma</Tex> than hit-or-miss but the same rate. A randomly shifted van der Corput
          (low-discrepancy) sequence achieves <Tex>{String.raw`\mathcal O(N^{-1}\log N)`}</Tex> by the Koksma–Hlawka inequality. The <strong>random walk</strong> with unit lattice steps
          has <Tex>{String.raw`\langle r^2(t)\rangle = t = 2dDt`}</Tex>, i.e. <Tex>{String.raw`D = 1/4`}</Tex> in 2D.
        </p>
      </MethodSection>
    </div>
  )
}

// ─── Ising ─────────────────────────────────────────────────────────────────

function IsingSection() {
  const [L, setL] = useState(128)
  const [T, setT] = useState(2.27)
  const [algo, setAlgo] = useState<IsingAlgorithm>('metropolis')
  const [sweeps, setSweeps] = useState(1)
  const [running, setRunning] = useState(true)
  const model = useMemo(() => new Ising(L, 2.27, 5, 'hot'), [L])
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const wrapRef = useRef<HTMLDivElement>(null)
  const width = useElementWidth(wrapRef)
  const size = Math.max(200, Math.min(460, width))
  const hist = useRef<{ i: number[]; m: number[]; e: number[] }>({ i: [], m: [], e: [] })
  const [live, setLive] = useState({ m: 0, e: 0, hi: [] as number[], hm: [] as number[], he: [] as number[] })
  const lastT = useRef(0)

  useEffect(() => model.setTemperature(T), [T, model])
  useEffect(() => {
    hist.current = { i: [], m: [], e: [] }
  }, [model, algo])

  const draw = () => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = fitCanvas(canvas, size, size)
    if (!ctx) return
    const img = new ImageData(L, L)
    for (let k = 0; k < L * L; k++) {
      const up = model.s[k] > 0
      img.data[k * 4] = up ? 230 : 22
      img.data[k * 4 + 1] = up ? 236 : 64
      img.data[k * 4 + 2] = up ? 244 : 128
      img.data[k * 4 + 3] = 255
    }
    const off = document.createElement('canvas')
    off.width = L
    off.height = L
    off.getContext('2d')!.putImageData(img, 0, 0)
    ctx.imageSmoothingEnabled = false
    ctx.drawImage(off, 0, 0, size, size)
  }

  useAnimationLoop(() => {
    for (let s = 0; s < sweeps; s++) model.sweep(algo)
    draw()
    const h = hist.current
    const n = h.i.length ? h.i[h.i.length - 1] + sweeps : sweeps
    h.i.push(n)
    h.m.push(model.magnetization / model.N)
    h.e.push(model.energy / model.N)
    if (h.i.length > 400) {
      h.i.shift()
      h.m.shift()
      h.e.shift()
    }
    const now = performance.now()
    if (now - lastT.current > 150) {
      lastT.current = now
      setLive({ m: model.magnetization / model.N, e: model.energy / model.N, hi: h.i.slice(), hm: h.m.slice(), he: h.e.slice() })
    }
  }, running)
  useEffect(draw)

  // Temperature sweep (worker).
  const [sweepL, setSweepL] = useState('32')
  const [sweepAlgo, setSweepAlgo] = useState<IsingAlgorithm>('wolff')
  const [points, setPoints] = useState<SweepPoint[]>([])
  const [busy, setBusy] = useState(false)
  const worker = useStreamingWorker<StatRequest, StatMsg>(makeWorker, (m) => {
    if (m.kind === 'point') setPoints((p) => [...p, m.point].sort((a, b) => a.T - b.T))
    else if (m.kind === 'done') setBusy(false)
  })
  const runSweep = () => {
    setPoints([])
    setBusy(true)
    const temps = Array.from({ length: 26 }, (_, i) => 1.5 + i * 0.08)
    worker.start({ kind: 'sweep', L: Number(sweepL), algo: sweepAlgo, temps, equil: 300, samples: Number(sweepL) > 32 ? 1500 : 3000 })
  }

  const exactT = useMemo(() => Array.from({ length: 200 }, (_, i) => 1.4 + i * 0.011), [])
  const tc = [{ x: T_CRITICAL, color: C.muted, label: 'T_c' }]
  const pt = (f: (p: SweepPoint) => number) => ({ x: points.map((p) => p.T), y: points.map(f) })

  return (
    <div className="grid gap-4 lg:grid-cols-[280px_minmax(0,1fr)]">
      <Panel title="2D Ising model">
        <div className="space-y-4">
          <Slider label="Temperature T" value={T} min={0.5} max={5} step={0.01} onChange={setT} format={(v) => v.toFixed(2)} hint={`T_c = ${T_CRITICAL.toFixed(4)}`} />
          <Segmented
            label="Algorithm"
            value={algo}
            onChange={setAlgo}
            options={[
              { value: 'metropolis', label: 'Metropolis' },
              { value: 'wolff', label: 'Wolff' },
            ]}
          />
          <Select
            label="Lattice L × L"
            value={String(L)}
            onChange={(v) => setL(Number(v))}
            options={[
              { value: '64', label: '64 × 64' },
              { value: '128', label: '128 × 128' },
              { value: '256', label: '256 × 256' },
            ]}
          />
          <Slider label="Sweeps per frame" value={sweeps} min={1} max={10} step={1} onChange={setSweeps} />
          <div className="grid grid-cols-2 gap-2">
            <button type="button" className="btn" onClick={() => setRunning((v) => !v)}>
              {running ? 'Pause' : 'Run'}
            </button>
            <button
              type="button"
              className="btn"
              onClick={() => {
                model.reset('hot')
                draw()
              }}
            >
              Randomise
            </button>
          </div>
          <div className="grid grid-cols-2 gap-2">
            <Readout label="m per spin" value={fmtFixed(live.m, 4)} />
            <Readout label="e per spin" value={fmtFixed(live.e, 4)} />
            <Readout label="Onsager |m|" value={fmtFixed(onsagerMagnetization(T), 4)} />
            <Readout label="Onsager e" value={fmtFixed(onsagerEnergy(T), 4)} />
          </div>
        </div>
      </Panel>
      <div className="min-w-0 space-y-4">
        <Panel>
          <div className="grid gap-4 md:grid-cols-[auto_minmax(0,1fr)]">
            <div ref={wrapRef} className="min-w-0 md:w-[460px]">
              <canvas ref={canvasRef} className="block rounded-xl" aria-label="Ising spin lattice (light = up, blue = down)" />
            </div>
            <div className="min-w-0">
              <Plot
                ariaLabel="Magnetisation and energy time series"
                height={size - 30}
                xLabel="sweep"
                yDomain={[-2.05, 1.05]}
                series={[
                  { x: live.hi, y: live.hm, color: C.blue, label: 'm (magnetisation / spin)' },
                  { x: live.hi, y: live.he, color: C.orange, label: 'e (energy / spin)' },
                ]}
              />
            </div>
          </div>
        </Panel>
        <Panel
          title="Temperature sweep vs Onsager’s exact solution"
          actions={
            <div className="flex flex-wrap items-center gap-2">
              <select className="select w-auto" aria-label="Sweep lattice size" value={sweepL} onChange={(e) => setSweepL(e.target.value)}>
                <option value="16">L = 16</option>
                <option value="32">L = 32</option>
                <option value="64">L = 64</option>
              </select>
              <select className="select w-auto" aria-label="Sweep algorithm" value={sweepAlgo} onChange={(e) => setSweepAlgo(e.target.value as IsingAlgorithm)}>
                <option value="wolff">Wolff</option>
                <option value="metropolis">Metropolis</option>
              </select>
              <button type="button" className="btn btn-primary" disabled={busy} onClick={runSweep}>
                {busy ? `Sampling in a worker… ${points.length}/26` : 'Run T sweep'}
              </button>
            </div>
          }
        >
          {points.length ? (
            <div className="grid gap-4 md:grid-cols-2">
              <Plot
                ariaLabel="Magnetisation vs temperature"
                height={210}
                xLabel="T"
                yLabel="⟨|m|⟩"
                vLines={tc}
                xDomain={[1.4, 3.6]}
                series={[
                  { x: exactT, y: exactT.map(onsagerMagnetization), color: C.muted, dash: [5, 4], label: 'Onsager (L = ∞)', width: 1.5 },
                  { ...pt((p) => p.m), color: C.blue, label: `Monte Carlo (L = ${sweepL})`, points: true },
                ]}
              />
              <Plot
                ariaLabel="Energy vs temperature"
                height={210}
                xLabel="T"
                yLabel="⟨e⟩"
                vLines={tc}
                xDomain={[1.4, 3.6]}
                series={[
                  { x: exactT, y: exactT.map(onsagerEnergy), color: C.muted, dash: [5, 4], label: 'Onsager (L = ∞)', width: 1.5 },
                  { ...pt((p) => p.e), color: C.orange, label: `Monte Carlo (L = ${sweepL})`, points: true },
                ]}
              />
              <Plot ariaLabel="Susceptibility vs temperature" height={210} xLabel="T" yLabel="χ" vLines={tc} xDomain={[1.4, 3.6]} series={[{ ...pt((p) => p.chi), color: C.violet, label: 'susceptibility χ', points: true }]} />
              <Plot ariaLabel="Specific heat vs temperature" height={210} xLabel="T" yLabel="C" vLines={tc} xDomain={[1.4, 3.6]} series={[{ ...pt((p) => p.c), color: C.aqua, label: 'specific heat C', points: true }]} />
            </div>
          ) : (
            <p className="py-6 text-center text-sm text-slate-400">Equilibrates and samples 26 temperatures from 3.5 down to 1.5 in a Web Worker, streaming results as they finish.</p>
          )}
        </Panel>
      </div>
    </div>
  )
}

// ─── Monte Carlo π ─────────────────────────────────────────────────────────

function PiSection() {
  const wrapRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const width = useElementWidth(wrapRef)
  const size = Math.max(160, Math.min(260, width / 2 - 8))
  const state = useRef({ rng: new Rng(9), n: 0, hits: 0 })
  const [running, setRunning] = useState(true)
  const [counts, setCounts] = useState({ n: 0, hits: 0 })
  const [scaling, setScaling] = useState<ErrorScaling | null>(null)
  useStreamingWorker<StatRequest, StatMsg>(makeWorker, (m) => m.kind === 'pi' && setScaling(m.result), { kind: 'pi' })

  useEffect(() => {
    const ctx = canvasRef.current && fitCanvas(canvasRef.current, size, size)
    if (!ctx) return
    ctx.fillStyle = '#020617'
    ctx.fillRect(0, 0, size, size)
    ctx.strokeStyle = 'rgba(148,163,184,0.5)'
    ctx.beginPath()
    ctx.arc(0, size, size, -Math.PI / 2, 0)
    ctx.stroke()
  }, [size])

  useAnimationLoop(() => {
    const s = state.current
    if (s.n > 2_000_000) return
    const ctx = canvasRef.current?.getContext('2d')
    const batch = s.n < 2000 ? 20 : 400
    for (let k = 0; k < batch; k++) {
      const x = s.rng.next()
      const y = s.rng.next()
      const inside = x * x + y * y < 1
      if (inside) s.hits++
      s.n++
      if (ctx && s.n < 30000) {
        ctx.fillStyle = inside ? 'rgba(57,135,229,0.85)' : 'rgba(217,89,38,0.85)'
        ctx.fillRect(x * size - 0.75, size - y * size - 0.75, 1.5, 1.5)
      }
    }
    setCounts({ n: s.n, hits: s.hits })
  }, running)

  const s = counts
  const est = s.n ? (4 * s.hits) / s.n : 0
  const scaleSeries: Series[] = scaling
    ? [
        { x: scaling.n, y: scaling.hitMiss, color: C.orange, label: `hit-or-miss MC (slope ${scaling.hitMissSlope.toFixed(2)})`, points: true },
        { x: scaling.n, y: scaling.meanValue, color: C.blue, label: `mean-value MC (slope ${scaling.meanValueSlope.toFixed(2)})`, points: true },
        { x: scaling.n, y: scaling.qmc, color: C.aqua, label: `quasi-MC (slope ${scaling.qmcSlope.toFixed(2)})`, points: true },
        { x: scaling.n, y: scaling.n.map((n) => 1.6 / Math.sqrt(n)), color: C.muted, dash: [5, 4], label: 'N^(−1/2)', width: 1 },
      ]
    : []
  return (
    <Panel title="Monte Carlo estimation of π">
      <div ref={wrapRef} className="flex flex-wrap gap-4">
        <canvas ref={canvasRef} className="block rounded-xl" aria-label="Random points in the unit square; blue points fall inside the quarter disk" />
        <div className="grid flex-1 grid-cols-1 gap-2">
          <Readout label="Samples N" value={s.n.toLocaleString()} />
          <Readout label="Estimate 4·hits/N" value={fmtFixed(est, 6)} />
          <Readout label="Error |π̂ − π|" value={fmt(Math.abs(est - Math.PI))} />
          <Readout label="Expected σ ≈ 1.64/√N" value={s.n ? fmt(1.642 / Math.sqrt(s.n)) : '—'} />
          <button type="button" className="btn" onClick={() => setRunning((v) => !v)}>
            {running ? 'Pause' : 'Run'}
          </button>
        </div>
      </div>
      <div className="mt-3">
        {scaling ? (
          <Plot ariaLabel="RMS error vs number of samples" series={scaleSeries} xLog yLog height={230} xLabel="N" yLabel="RMS error" />
        ) : (
          <p className="py-6 text-center text-sm text-slate-400">Computing error scaling over 48 independent runs per N…</p>
        )}
      </div>
    </Panel>
  )
}

// ─── Random walk ───────────────────────────────────────────────────────────

function WalkSection() {
  const wrapRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const width = useElementWidth(wrapRef)
  const size = Math.max(200, Math.min(320, width))
  const [seed, setSeed] = useState(1)
  const walkers = useMemo(() => new RandomWalkers(3000, seed), [seed])
  const msd = useRef<{ t: number[]; m: number[] }>({ t: [], m: [] })
  const [running, setRunning] = useState(true)
  const [m, setM] = useState<{ steps: number; t: number[]; m: number[] }>({ steps: 0, t: [], m: [] })
  useEffect(() => {
    msd.current = { t: [], m: [] }
  }, [walkers])

  useAnimationLoop(() => {
    if (walkers.t >= 2000) return
    walkers.step()
    walkers.step()
    msd.current.t.push(walkers.t)
    msd.current.m.push(walkers.msd())
    const ctx = canvasRef.current && fitCanvas(canvasRef.current, size, size)
    if (ctx) {
      ctx.fillStyle = '#020617'
      ctx.fillRect(0, 0, size, size)
      const sc = size / 200
      ctx.strokeStyle = 'rgba(148,163,184,0.4)'
      ctx.beginPath()
      ctx.arc(size / 2, size / 2, Math.sqrt(walkers.t) * sc, 0, 2 * Math.PI)
      ctx.stroke()
      ctx.fillStyle = 'rgba(144,133,233,0.7)'
      for (let i = 0; i < walkers.count; i++) ctx.fillRect(size / 2 + walkers.x[i] * sc - 1, size / 2 + walkers.y[i] * sc - 1, 2, 2)
    }
    setM({ steps: walkers.t, t: msd.current.t.slice(), m: msd.current.m.slice() })
  }, running)
  const slope = m.t.length > 10 ? m.m[m.m.length - 1] / m.t[m.t.length - 1] : NaN
  return (
    <Panel title="Random walk diffusion (3000 walkers)">
      <div ref={wrapRef} className="grid gap-3 sm:grid-cols-[auto_minmax(0,1fr)]">
        <canvas ref={canvasRef} className="block rounded-xl" aria-label="Random walkers spreading from the origin; ring at radius √t" />
        <div className="grid h-fit grid-cols-1 gap-2">
          <Readout label="Steps t" value={m.steps.toLocaleString()} />
          <Readout label="MSD ⟨r²⟩" value={fmtFixed(m.m[m.m.length - 1] ?? 0, 1)} />
          <Readout label="MSD / t (theory 1)" value={fmtFixed(slope, 4)} tone={Math.abs(slope - 1) < 0.05 ? 'good' : undefined} />
          <div className="grid grid-cols-2 gap-2">
            <button type="button" className="btn" onClick={() => setRunning((v) => !v)}>
              {running ? 'Pause' : 'Run'}
            </button>
            <button type="button" className="btn" onClick={() => setSeed((s) => s + 1)}>
              Restart
            </button>
          </div>
        </div>
      </div>
      <div className="mt-3">
        <Plot
          ariaLabel="Mean squared displacement vs time"
          height={200}
          xLabel="t (steps)"
          yLabel="⟨r²⟩"
          series={[
            { x: [0, m.t[m.t.length - 1] ?? 1], y: [0, m.t[m.t.length - 1] ?? 1], color: C.muted, dash: [5, 4], label: 'theory ⟨r²⟩ = t', width: 1.5 },
            { x: m.t, y: m.m, color: C.violet, label: 'measured MSD' },
          ]}
        />
      </div>
    </Panel>
  )
}
