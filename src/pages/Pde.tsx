import { useEffect, useRef, useState, type PointerEvent } from 'react'
import { FieldCanvas, type FieldCanvasHandle } from '../components/FieldCanvas.tsx'
import { Plot } from '../components/Plot.tsx'
import { Tex } from '../components/Tex.tsx'
import { MethodSection, PageHeader, Panel, Readout, Segmented, Select, Slider } from '../components/ui.tsx'
import { fmt, fmtFixed } from '../lib/format.ts'
import { useAnimationLoop, useElementWidth } from '../lib/hooks.ts'
import { C } from '../lib/palette.ts'
import { colormapCss, type ColormapId } from '../numerics/colormaps.ts'
import { GRAY_SCOTT_PRESETS, GrayScott, Heat2D, heatModeAmplitude, Wave2D, type HeatScheme, type WaveBoundary, type WaveSource } from '../numerics/pde.ts'

type Tab = 'heat' | 'wave' | 'grayscott'

const CMAPS: { value: ColormapId; label: string }[] = [
  { value: 'viridis', label: 'Viridis' },
  { value: 'inferno', label: 'Inferno' },
  { value: 'coolwarm', label: 'Cool–warm (diverging)' },
]

function Colorbar({ cmap, lo, hi }: { cmap: ColormapId; lo: string; hi: string }) {
  const stops = Array.from({ length: 11 }, (_, i) => `${colormapCss(cmap, i / 10)} ${i * 10}%`).join(', ')
  return (
    <div className="flex items-center gap-2 font-mono text-[11px] text-slate-400">
      <span>{lo}</span>
      <span className="h-2.5 flex-1 rounded" style={{ background: `linear-gradient(to right, ${stops})` }} />
      <span>{hi}</span>
    </div>
  )
}

function useSquareSize() {
  const ref = useRef<HTMLDivElement>(null)
  const w = useElementWidth(ref)
  return { ref, size: Math.max(200, Math.min(560, w)) }
}

export default function Pde() {
  const [tab, setTab] = useState<Tab>('heat')
  return (
    <div>
      <PageHeader eyebrow="Module 4 · Partial differential equations" title="PDE solvers">
        Finite-difference solvers on 2D grids, all CPU typed-array code. Paint initial conditions with the mouse, push the time step past the CFL limit and watch an explicit
        scheme explode while the implicit one does not.
      </PageHeader>
      <div className="mb-4 max-w-xl">
        <Segmented
          label="Equation"
          value={tab}
          onChange={setTab}
          options={[
            { value: 'heat', label: 'Heat (diffusion)' },
            { value: 'wave', label: 'Wave' },
            { value: 'grayscott', label: 'Gray–Scott' },
          ]}
        />
      </div>
      {tab === 'heat' && <HeatPanel />}
      {tab === 'wave' && <WavePanel />}
      {tab === 'grayscott' && <GrayScottPanel />}
      <MethodSection
        references={[
          { text: 'R. J. LeVeque — Finite Difference Methods for Ordinary and Partial Differential Equations, SIAM (2007).' },
          { text: 'D. W. Peaceman, H. H. Rachford — The numerical solution of parabolic and elliptic differential equations, J. SIAM 3 (1955) 28–41.' },
          { text: 'G. Mur — Absorbing boundary conditions for the finite-difference approximation of the time-domain electromagnetic-field equations, IEEE Trans. EMC 23 (1981) 377–382.' },
          { text: 'J. E. Pearson — Complex patterns in a simple system, Science 261 (1993) 189–192.' },
          { text: 'S. van der Walt, N. Smith — mpl colormaps (viridis, inferno); K. Moreland — Diverging color maps for scientific visualization (2009).' },
        ]}
      >
        <p>
          <strong>Heat equation</strong> <Tex>{String.raw`u_t = \alpha \nabla^2 u`}</Tex> on the unit square with <Tex>u = 0</Tex> on the boundary. FTCS (forward time, centred
          space) uses the 5-point Laplacian:
        </p>
        <Tex block>{String.raw`u^{n+1}_{ij} = u^n_{ij} + r\,\big(u^n_{i+1,j} + u^n_{i-1,j} + u^n_{i,j+1} + u^n_{i,j-1} - 4u^n_{ij}\big),\qquad r = \frac{\alpha\,\Delta t}{\Delta x^2}`}</Tex>
        <p>
          Von Neumann analysis gives the amplification factor <Tex>{String.raw`g = 1 - 4r\,(\sin^2\tfrac{k_x\Delta x}{2} + \sin^2\tfrac{k_y\Delta x}{2})`}</Tex>, so stability{' '}
          <Tex>{String.raw`|g| \le 1`}</Tex> requires <Tex>{String.raw`r \le 1/4`}</Tex>. The Peaceman–Rachford ADI scheme splits a Crank–Nicolson step into two half steps that are
          each implicit in one direction, so every half step is a set of tridiagonal solves (Thomas algorithm, O(N) per line):
        </p>
        <Tex block>{String.raw`\Big(I - \tfrac r2 \delta_x^2\Big) u^{*} = \Big(I + \tfrac r2 \delta_y^2\Big) u^{n},\qquad \Big(I - \tfrac r2 \delta_y^2\Big) u^{n+1} = \Big(I + \tfrac r2 \delta_x^2\Big) u^{*}`}</Tex>
        <p>
          It is unconditionally stable and second order in time and space. The tests check both schemes against the analytic decay{' '}
          <Tex>{String.raw`u = e^{-2\pi^2\alpha t}\sin \pi x \sin \pi y`}</Tex> (FTCS error 3×10⁻⁴; ADI 1.5×10⁻⁴ at 40× the FTCS step) and measure ADI’s temporal order as 2.00.
        </p>
        <p>
          <strong>Wave equation</strong> <Tex>{String.raw`u_{tt} = c^2\nabla^2 u`}</Tex> with the explicit leapfrog{' '}
          <Tex>{String.raw`u^{n+1} = 2u^n - u^{n-1} + C^2 \Delta_h u^n`}</Tex>, stable for Courant number <Tex>{String.raw`C = c\Delta t/\Delta x \le 1/\sqrt 2`}</Tex>. Absorbing edges
          use Mur’s first-order condition, a discretisation of the one-way wave equation <Tex>{String.raw`u_t = c\,u_x`}</Tex>:{' '}
          <Tex>{String.raw`u^{n+1}_0 = u^n_1 + \frac{C - 1}{C + 1}\big(u^{n+1}_1 - u^n_0\big)`}</Tex>.
        </p>
        <p>
          <strong>Gray–Scott</strong> reaction–diffusion, <Tex>{String.raw`u_t = D_u\nabla^2 u - uv^2 + F(1 - u)`}</Tex>,{' '}
          <Tex>{String.raw`v_t = D_v\nabla^2 v + uv^2 - (F + k)v`}</Tex>, explicit Euler with a 9-point isotropic Laplacian and periodic boundaries. Small changes in{' '}
          <Tex>(F, k)</Tex> select spots, stripes, mitosis or waves.
        </p>
      </MethodSection>
    </div>
  )
}

// ─── Heat ──────────────────────────────────────────────────────────────────

const HN = 128
const HEAT_DX = 1 / (HN - 1)
const HEAT_DT_MAX = 0.25 * HEAT_DX * HEAT_DX // FTCS limit for α = 1

function seedHeat(heat: Heat2D, kind: 'blobs' | 'mode'): void {
  heat.u.fill(0)
  heat.t = 0
  if (kind === 'mode') heat.setMode(1, 1)
  else {
    // Dirichlet walls: hot (+1) on the left, cold (−1) on the right, insulating-free 0 top/bottom.
    for (let j = 1; j < HN - 1; j++) {
      heat.u[j * HN] = 1
      heat.u[j * HN + HN - 1] = -1
    }
    heat.paint(40, 50, 10, 1)
    heat.paint(85, 80, 14, -0.9)
    heat.paint(90, 30, 6, 1)
  }
}

function modeAmplitude(heat: Heat2D): number {
  let s = 0
  for (let j = 0; j < HN; j++) for (let i = 0; i < HN; i++) s += heat.u[j * HN + i] * Math.sin((Math.PI * i) / (HN - 1)) * Math.sin((Math.PI * j) / (HN - 1))
  return 4 * s * heat.dx * heat.dx
}

function HeatPanel() {
  const { ref, size } = useSquareSize()
  const field = useRef<FieldCanvasHandle>(null)
  const sim = useRef<Heat2D | null>(null)
  const history = useRef<{ t: number[]; a: number[] }>({ t: [], a: [] })
  const [scheme, setScheme] = useState<HeatScheme>('adi')
  const [mult, setMult] = useState(2)
  const [stepsPerFrame, setStepsPerFrame] = useState(1)
  const [cmap, setCmap] = useState<ColormapId>('coolwarm')
  const [running, setRunning] = useState(true)
  const [modeTest, setModeTest] = useState(false)
  const [view, setView] = useState({ t: 0, maxAbs: 1, histT: [] as number[], histA: [] as number[] })
  const dt = mult * HEAT_DT_MAX
  const r = dt / (HEAT_DX * HEAT_DX)
  const unstable = scheme === 'ftcs' && r > 0.25

  const getHeat = () => {
    if (!sim.current) {
      sim.current = new Heat2D(HN, 1)
      seedHeat(sim.current, 'blobs')
    }
    return sim.current
  }
  const publish = (heat: Heat2D) => setView({ t: heat.t, maxAbs: heat.maxAbs(), histT: history.current.t.slice(), histA: history.current.a.slice() })

  const reset = (kind: 'blobs' | 'mode') => {
    const heat = getHeat()
    seedHeat(heat, kind)
    history.current = { t: [], a: [] }
    setModeTest(kind === 'mode')
    if (kind === 'mode') {
      history.current.t.push(0)
      history.current.a.push(modeAmplitude(heat))
    }
    field.current?.render(heat.u, cmap, -1, 1)
    publish(heat)
  }

  useAnimationLoop(() => {
    const heat = getHeat()
    for (let k = 0; k < stepsPerFrame; k++) heat.step(dt, scheme)
    if (modeTest && history.current.t.length < 3000) {
      history.current.t.push(heat.t)
      history.current.a.push(modeAmplitude(heat))
    }
    if (!(heat.maxAbs() < 1e6)) setRunning(false)
    field.current?.render(heat.u, cmap, -1, 1)
    publish(heat)
  }, running)

  useEffect(() => {
    field.current?.render(getHeat().u, cmap, -1, 1)
  })

  const paint = (gi: number, gj: number, e: PointerEvent, phase: string) => {
    if (phase === 'up') return
    const heat = getHeat()
    heat.paint(gi, gj, 5, e.shiftKey ? -0.25 : 0.25)
    setModeTest(false)
    field.current?.render(heat.u, cmap, -1, 1)
  }

  const exactA = heatModeAmplitude(1, view.t)
  const measured = view.histA[view.histA.length - 1] ?? NaN
  const relErr = Math.abs(measured - exactA) / exactA
  return (
    <div className="grid gap-4 lg:grid-cols-[280px_minmax(0,1fr)]">
      <Panel title="Heat equation controls">
        <div className="space-y-4">
          <Segmented
            label="Scheme"
            value={scheme}
            onChange={setScheme}
            options={[
              { value: 'ftcs', label: 'FTCS (explicit)' },
              { value: 'adi', label: 'ADI (implicit)' },
            ]}
          />
          <Slider label="Δt / Δt_FTCS-max" value={mult} min={0.25} max={50} step={0.25} onChange={setMult} format={(v) => `${v.toFixed(2)}×`} />
          <Slider label="Steps per frame" value={stepsPerFrame} min={1} max={10} step={1} onChange={setStepsPerFrame} />
          <Select<ColormapId> label="Colormap" value={cmap} onChange={setCmap} options={CMAPS} />
          <div className="grid grid-cols-2 gap-2">
            <button type="button" className="btn" onClick={() => reset('blobs')}>
              Blobs + hot/cold walls
            </button>
            <button type="button" className="btn" onClick={() => reset('mode')}>
              Fourier-mode test
            </button>
          </div>
          <button type="button" className="btn w-full" onClick={() => setRunning((v) => !v)}>
            {running ? 'Pause' : 'Run'}
          </button>
          <p className="text-xs text-slate-400">Drag on the field to add heat; hold Shift to add cold.</p>
        </div>
      </Panel>
      <div className="min-w-0 space-y-4">
        <Panel>
          <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_260px]">
            <div ref={ref} className="space-y-2">
              <FieldCanvas ref={field} n={HN} size={size} label="Temperature field" onPointer={paint} />
              <div style={{ maxWidth: size }}>
                <Colorbar cmap={cmap} lo="−1" hi="+1" />
              </div>
            </div>
            <div className="grid h-fit grid-cols-2 gap-2 xl:grid-cols-1">
              <Readout label="Mesh ratio r = αΔt/Δx²" value={fmtFixed(r, 3)} tone={unstable ? 'bad' : 'good'} />
              <Readout label="FTCS stability (r ≤ 1/4)" value={scheme === 'adi' ? 'ADI: unconditionally stable' : unstable ? 'VIOLATED — will blow up' : 'satisfied'} tone={unstable ? 'bad' : 'good'} />
              <Readout label="Time t" value={fmt(view.t, 4)} />
              <Readout label="max |u|" value={fmt(view.maxAbs)} tone={view.maxAbs > 2 ? 'bad' : undefined} />
              {modeTest && (
                <>
                  <Readout label="Mode amplitude (measured)" value={fmt(measured, 6)} />
                  <Readout label="Analytic e^{−2π²t}" value={fmt(exactA, 6)} />
                  <Readout label="Relative error" value={fmt(relErr)} tone={relErr < 1e-2 ? 'good' : 'bad'} />
                </>
              )}
            </div>
          </div>
        </Panel>
        {modeTest && (
          <Panel title="Fourier-mode decay: simulation vs analytic">
            <Plot
              ariaLabel="Mode decay"
              yLog
              height={220}
              xLabel="t"
              yLabel="amplitude"
              series={[
                { x: view.histT, y: view.histT.map((t) => heatModeAmplitude(1, t)), color: C.muted, dash: [5, 4], label: 'analytic', width: 1.5 },
                { x: view.histT, y: view.histA, color: scheme === 'adi' ? C.aqua : C.orange, label: scheme === 'adi' ? 'ADI' : 'FTCS' },
              ]}
            />
          </Panel>
        )}
      </div>
    </div>
  )
}

// ─── Wave ──────────────────────────────────────────────────────────────────

const WN = 180

type WavePreset = 'two' | 'slit' | 'one'

function setupWave(wave: Wave2D, preset: WavePreset, omega: number): void {
  wave.clearField()
  wave.wall.fill(0)
  if (preset === 'two') {
    wave.sources = [
      { i: 70, j: 90, omega, amp: 1 },
      { i: 110, j: 90, omega, amp: 1 },
    ]
  } else if (preset === 'slit') {
    for (let i = 0; i < WN; i++) if (Math.abs(i - 75) > 4 && Math.abs(i - 105) > 4) for (let j = 98; j < 102; j++) wave.wall[j * WN + i] = 1
    wave.sources = [{ i: 90, j: 150, omega, amp: 1.5 }]
  } else {
    wave.sources = []
    wave.pulse(90, 90, 5, 2)
  }
}

function WavePanel() {
  const { ref, size } = useSquareSize()
  const field = useRef<FieldCanvasHandle>(null)
  const sim = useRef<Wave2D | null>(null)
  const [boundary, setBoundary] = useState<WaveBoundary>('absorb')
  const [courant, setCourant] = useState(0.5)
  const [omega, setOmega] = useState(0.25)
  const [damping, setDamping] = useState(0)
  const [cmap, setCmap] = useState<ColormapId>('coolwarm')
  const [preset, setPreset] = useState<WavePreset>('two')
  const [running, setRunning] = useState(true)
  const [view, setView] = useState({ steps: 0, histT: [] as number[], histE: [] as number[] })
  const drag = useRef<WaveSource | null>(null)
  const energyHist = useRef<{ t: number[]; e: number[] }>({ t: [], e: [] })

  const getWave = () => {
    if (!sim.current) {
      sim.current = new Wave2D(WN)
      setupWave(sim.current, 'two', 0.25)
    }
    return sim.current
  }

  const choosePreset = (p: WavePreset) => {
    setPreset(p)
    setupWave(getWave(), p, omega)
    energyHist.current = { t: [], e: [] }
  }

  useEffect(() => {
    const wave = getWave()
    wave.boundary = boundary
    wave.courant = courant
    wave.damping = damping
    for (const s of wave.sources) s.omega = omega
  }, [boundary, courant, damping, omega])

  const render = () => {
    const wave = getWave()
    field.current?.render(wave.u, cmap, -0.5, 0.5, wave.wall)
    field.current?.overlay((ctx, cell) => {
      for (const s of wave.sources) {
        ctx.strokeStyle = '#f8fafc'
        ctx.lineWidth = 2
        ctx.beginPath()
        ctx.arc((s.i + 0.5) * cell, (s.j + 0.5) * cell, 7, 0, 2 * Math.PI)
        ctx.stroke()
      }
    })
  }

  useAnimationLoop(() => {
    const wave = getWave()
    for (let k = 0; k < 2; k++) wave.step()
    const h = energyHist.current
    if (wave.steps % 10 < 2) {
      h.t.push(wave.steps)
      h.e.push(wave.energy())
      if (h.t.length > 600) {
        h.t.shift()
        h.e.shift()
      }
    }
    if (!(Math.abs(wave.u[(WN >> 1) * WN + (WN >> 1)]) < 1e6)) setRunning(false)
    render()
    setView({ steps: wave.steps, histT: h.t.slice(), histE: h.e.slice() })
  }, running)
  useEffect(render)

  const onPointer = (gi: number, gj: number, e: PointerEvent, phase: 'down' | 'move' | 'up') => {
    const wave = getWave()
    if (phase === 'down') {
      const hit = wave.sources.find((s) => Math.hypot(s.i - gi, s.j - gj) < 8)
      if (hit) drag.current = hit
      else if (e.shiftKey) wave.sources.push({ i: Math.round(gi), j: Math.round(gj), omega, amp: 1 })
      else wave.pulse(gi, gj, 3, 1.5)
    } else if (phase === 'move' && drag.current) {
      drag.current.i = Math.max(1, Math.min(WN - 2, Math.round(gi)))
      drag.current.j = Math.max(1, Math.min(WN - 2, Math.round(gj)))
    } else if (phase === 'up') drag.current = null
    if (!running) render()
  }

  const unstable = courant > Math.SQRT1_2
  return (
    <div className="grid gap-4 lg:grid-cols-[280px_minmax(0,1fr)]">
      <Panel title="Wave equation controls">
        <div className="space-y-4">
          <Select<WavePreset>
            label="Scenario"
            value={preset}
            onChange={choosePreset}
            options={[
              { value: 'two', label: 'Two-source interference' },
              { value: 'slit', label: 'Double slit' },
              { value: 'one', label: 'Single pulse' },
            ]}
          />
          <Segmented
            label="Boundary"
            value={boundary}
            onChange={setBoundary}
            options={[
              { value: 'absorb', label: 'Absorbing (Mur)' },
              { value: 'reflect', label: 'Reflecting' },
            ]}
          />
          <Slider label="Courant number C = cΔt/Δx" value={courant} min={0.1} max={0.8} step={0.01} onChange={setCourant} format={(v) => v.toFixed(2)} hint={unstable ? 'C > 1/√2 ≈ 0.707: unstable!' : 'Stable for C ≤ 1/√2 ≈ 0.707'} />
          <Slider label="Source frequency ω (rad/step)" value={omega} min={0.05} max={0.6} step={0.01} onChange={setOmega} format={(v) => v.toFixed(2)} />
          <Slider label="Damping γ" value={damping} min={0} max={0.02} step={0.001} onChange={setDamping} format={(v) => v.toFixed(3)} />
          <Select<ColormapId> label="Colormap" value={cmap} onChange={setCmap} options={CMAPS} />
          <button type="button" className="btn w-full" onClick={() => setRunning((v) => !v)}>
            {running ? 'Pause' : 'Run'}
          </button>
          <p className="text-xs text-slate-400">Drag the white rings to move sources. Click to drop a pulse; Shift-click to add a source.</p>
        </div>
      </Panel>
      <Panel>
        <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_260px]">
          <div ref={ref} className="space-y-2">
            <FieldCanvas ref={field} n={WN} size={size} label="Wave field" onPointer={onPointer} />
            <div style={{ maxWidth: size }}>
              <Colorbar cmap={cmap} lo="−0.5" hi="+0.5" />
            </div>
          </div>
          <div className="space-y-2">
            <Readout label="Courant number" value={fmtFixed(courant, 2)} tone={unstable ? 'bad' : 'good'} />
            <Readout label="Time steps" value={view.steps.toLocaleString()} />
            <Readout label="Wavelength (cells)" value={fmtFixed((2 * Math.PI * courant) / omega, 1)} />
            <Plot ariaLabel="Discrete wave energy" series={[{ x: view.histT, y: view.histE, color: C.violet, label: 'energy' }]} height={170} xLabel="step" yLabel="E" />
          </div>
        </div>
      </Panel>
    </div>
  )
}

// ─── Gray–Scott ────────────────────────────────────────────────────────────

const GN = 160

function GrayScottPanel() {
  const { ref, size } = useSquareSize()
  const field = useRef<FieldCanvasHandle>(null)
  const sim = useRef<GrayScott | null>(null)
  const [presetId, setPresetId] = useState('coral')
  const [F, setF] = useState(0.0545)
  const [k, setK] = useState(0.062)
  const [iters, setIters] = useState(12)
  const [cmap, setCmap] = useState<ColormapId>('viridis')
  const [running, setRunning] = useState(true)
  const [view, setView] = useState({ ms: 0, steps: 0 })
  const lastT = useRef(0)

  const getGs = () => {
    if (!sim.current) {
      sim.current = new GrayScott(GN)
      sim.current.reset(1)
    }
    return sim.current
  }

  useEffect(() => {
    const gs = getGs()
    gs.F = F
    gs.k = k
  }, [F, k])

  useAnimationLoop(() => {
    const gs = getGs()
    const t0 = performance.now()
    gs.step(iters)
    const dt = performance.now() - t0
    field.current?.render(gs.v, cmap, 0, 0.42)
    if (t0 - lastT.current > 500) {
      lastT.current = t0
      setView({ ms: dt, steps: gs.steps })
    }
  }, running)
  useEffect(() => {
    field.current?.render(getGs().v, cmap, 0, 0.42)
  })

  return (
    <div className="grid gap-4 lg:grid-cols-[280px_minmax(0,1fr)]">
      <Panel title="Gray–Scott controls">
        <div className="space-y-4">
          <Select
            label="Pattern preset"
            value={presetId}
            onChange={(id) => {
              const p = GRAY_SCOTT_PRESETS.find((q) => q.id === id)!
              setPresetId(id)
              setF(p.F)
              setK(p.k)
              getGs().reset(Math.floor(Math.random() * 1000))
            }}
            options={GRAY_SCOTT_PRESETS.map((p) => ({ value: p.id, label: p.name }))}
          />
          <Slider label="Feed rate F" value={F} min={0.005} max={0.09} step={0.0005} onChange={setF} format={(v) => v.toFixed(4)} />
          <Slider label="Kill rate k" value={k} min={0.03} max={0.075} step={0.0005} onChange={setK} format={(v) => v.toFixed(4)} />
          <Slider label="Iterations per frame" value={iters} min={1} max={40} step={1} onChange={setIters} />
          <Select<ColormapId> label="Colormap" value={cmap} onChange={setCmap} options={CMAPS} />
          <div className="grid grid-cols-2 gap-2">
            <button type="button" className="btn" onClick={() => setRunning((v) => !v)}>
              {running ? 'Pause' : 'Run'}
            </button>
            <button type="button" className="btn" onClick={() => getGs().reset(Math.floor(Math.random() * 1000))}>
              Reseed
            </button>
          </div>
          <p className="text-xs text-slate-400">Paint on the field to inject chemical v.</p>
        </div>
      </Panel>
      <Panel>
        <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_220px]">
          <div ref={ref} className="space-y-2">
            <FieldCanvas ref={field} n={GN} size={size} label="Gray–Scott concentration v" onPointer={(gi, gj, _e, phase) => phase !== 'up' && getGs().paint(gi, gj, 4)} />
            <div style={{ maxWidth: size }}>
              <Colorbar cmap={cmap} lo="v = 0" hi="0.42" />
            </div>
          </div>
          <div className="space-y-2">
            <Readout label="Grid" value={`${GN} × ${GN}, periodic`} />
            <Readout label="Iterations" value={view.steps.toLocaleString()} />
            <Readout label="CPU ms / frame" value={fmtFixed(view.ms, 1)} />
            <Readout label="Cell updates / s" value={view.ms > 0 ? fmt((GN * GN * iters * 1000) / view.ms, 3) : '—'} />
          </div>
        </div>
      </Panel>
    </div>
  )
}
