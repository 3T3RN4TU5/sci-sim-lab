import { useMemo, useState } from 'react'
import { Plot, type Series } from '../components/Plot.tsx'
import { Tex } from '../components/Tex.tsx'
import { Check, MethodSection, PageHeader, Panel, Readout, Segmented, Select, Slider } from '../components/ui.tsx'
import { fmt } from '../lib/format.ts'
import { useAnimationLoop } from '../lib/hooks.ts'
import { C, METHOD_COLOR } from '../lib/palette.ts'
import { useWorkerJob } from '../lib/worker.ts'
import { convergenceStudy, referenceSolution } from '../numerics/convergence.ts'
import { dopri45, integrateFixed, METHODS, type MethodId } from '../numerics/ode.ts'
import type { StiffRequest, StiffResponse } from '../numerics/stiff.ts'
import { harmonic, kepler, pendulum, type ModelProblem } from '../numerics/systems.ts'

type ProblemId = 'kepler' | 'pendulum' | 'harmonic'
type LabMethod = MethodId | 'dopri45'

const FIXED: MethodId[] = ['euler', 'midpoint', 'rk4', 'verlet', 'yoshida4', 'backward-euler']
const ALL: LabMethod[] = [...FIXED, 'dopri45']
const NAME: Record<LabMethod, string> = { ...Object.fromEntries(FIXED.map((m) => [m, METHODS[m].name])), dopri45: 'Dormand–Prince RK45' } as Record<LabMethod, string>
const SHORT: Record<LabMethod, string> = { ...Object.fromEntries(FIXED.map((m) => [m, METHODS[m].short])), dopri45: 'RK45' } as Record<LabMethod, string>
const SAMPLES = 1500

interface Run {
  method: LabMethod
  t: number[]
  a: number[]
  b: number[]
  energy: number[]
  steps: number
  fevals: number
  ms: number
  maxDE: number
  finalErr: number
}

function makeProblem(id: ProblemId, ecc: number, theta0: number): ModelProblem {
  return id === 'kepler' ? kepler(ecc) : id === 'pendulum' ? pendulum(theta0) : harmonic()
}

function runMethod(method: LabMethod, p: ModelProblem, dt: number, T: number, tol: number, ref: Float64Array): Run {
  const t0 = performance.now()
  const E0 = p.energy!(p.y0)
  const run: Run = { method, t: [], a: [], b: [], energy: [], steps: 0, fevals: 0, ms: 0, maxDE: 0, finalErr: 0 }
  const kepler2d = p.id === 'kepler'
  const push = (tt: number, y: Float64Array) => {
    run.t.push(tt)
    run.a.push(y[0])
    run.b.push(kepler2d ? y[1] : y[1])
    const de = Math.abs((p.energy!(y) - E0) / E0)
    run.energy.push(Number.isFinite(de) ? Math.max(de, 1e-17) : NaN)
    if (de > run.maxDE || !Number.isFinite(de)) run.maxDE = Number.isFinite(de) ? Math.max(run.maxDE, de) : Infinity
  }
  let yEnd: Float64Array
  if (method === 'dopri45') {
    const r = dopri45(p, 0, p.y0, T, { rtol: tol, atol: tol * 1e-2 })
    const every = Math.max(1, Math.floor(r.t.length / SAMPLES))
    r.t.forEach((tt, i) => (i % every === 0 || i === r.t.length - 1) && push(tt, r.y[i]))
    run.steps = r.accepted
    run.fevals = r.fevals
    yEnd = r.yEnd
  } else {
    const steps = Math.round(T / dt)
    yEnd = integrateFixed(method, p, p.y0, 0, dt, steps, push, Math.max(1, Math.floor(steps / SAMPLES)))
    run.steps = steps
    run.fevals = METHODS[method].implicit ? NaN : steps * METHODS[method].evalsPerStep
  }
  run.finalErr = Math.max(...Array.from(yEnd, (v, i) => Math.abs(v - ref[i])))
  run.ms = performance.now() - t0
  return run
}

export default function IntegratorLab() {
  const [problemId, setProblemId] = useState<ProblemId>('kepler')
  const [ecc, setEcc] = useState(0.6)
  const [theta0, setTheta0] = useState(2.5)
  const [logDt, setLogDt] = useState(-1.5)
  const [periods, setPeriods] = useState(20)
  const [tol, setTol] = useState(-6)
  const [enabled, setEnabled] = useState<Record<LabMethod, boolean>>({ euler: true, midpoint: false, rk4: true, verlet: true, yoshida4: true, 'backward-euler': false, dopri45: true })
  const [progress, setProgress] = useState(1)
  const [playing, setPlaying] = useState(true)

  const dt = 10 ** logDt
  const problem = useMemo(() => makeProblem(problemId, ecc, theta0), [problemId, ecc, theta0])
  const T = periods * problem.period
  const maxSteps = 400_000
  const dtEff = Math.max(dt, T / maxSteps)

  const runs = useMemo(() => {
    const ref = referenceSolution(problem, T)
    return ALL.filter((m) => enabled[m]).map((m) => runMethod(m, problem, dtEff, T, 10 ** tol, ref))
  }, [problem, T, dtEff, tol, enabled])

  const convergence = useMemo(() => {
    const Tc = problem.period
    const ref = referenceSolution(problem, Tc)
    return FIXED.map((m) => {
      const base = m === 'euler' || m === 'backward-euler' ? 1600 : 100
      return convergenceStudy(m, problem, Tc, [base, base * 2, base * 4, base * 8, base * 16], ref)
    })
  }, [problem])

  useAnimationLoop((ms) => {
    setProgress((p) => {
      const next = p + ms / 6000
      return next > 1.15 ? 0 : next
    })
  }, playing)

  const shown = Math.min(1, progress)
  const cut = (arr: number[], t: number[]) => {
    let k = 0
    const lim = shown * T
    while (k < t.length && t[k] <= lim) k++
    return arr.slice(0, Math.max(1, k))
  }

  const trajSeries: Series[] = runs.map((r) => ({ x: cut(r.a, r.t), y: cut(r.b, r.t), color: METHOD_COLOR[r.method], label: SHORT[r.method], width: 1.5 }))
  if (problem.id === 'kepler') trajSeries.push({ x: [0], y: [0], color: C.yellow, points: true, line: false, radius: 6, label: 'Sun' })
  const energySeries: Series[] = runs.map((r) => ({ x: cut(r.t, r.t), y: cut(r.energy, r.t), color: METHOD_COLOR[r.method], label: SHORT[r.method], width: 1.5 }))
  const convSeries: Series[] = convergence.map((c) => ({ x: c.h, y: c.error, color: METHOD_COLOR[c.method], label: `${METHODS[c.method].short}  p = ${c.order.toFixed(2)}`, points: true }))

  const stiff = useWorkerJob<StiffRequest, StiffResponse>(() => new Worker(new URL('../workers/stiff.worker.ts', import.meta.url), { type: 'module' }))
  const [stiffProblem, setStiffProblem] = useState<StiffRequest['problem']>('vanderpol')
  const [logMu, setLogMu] = useState(2.5)
  const [stiffLogH, setStiffLogH] = useState(-1.3)

  return (
    <div>
      <PageHeader eyebrow="Module 1 · Ordinary differential equations" title="Integrator lab">
        Run seven integrators on the same problem and compare trajectories, long-term energy behaviour and measured order of accuracy. Symplectic methods keep the energy
        error bounded forever; non-symplectic ones drift. A stiff problem shows when explicit methods fail.
      </PageHeader>

      <div className="grid gap-4 lg:grid-cols-[280px_minmax(0,1fr)]">
        <Panel title="Problem & methods">
          <div className="space-y-4">
            <Select<ProblemId>
              label="Problem"
              value={problemId}
              onChange={setProblemId}
              options={[
                { value: 'kepler', label: 'Kepler orbit (2-body)' },
                { value: 'pendulum', label: 'Nonlinear pendulum' },
                { value: 'harmonic', label: 'Harmonic oscillator' },
              ]}
            />
            {problemId === 'kepler' && <Slider label="Eccentricity e" value={ecc} min={0} max={0.9} step={0.05} onChange={setEcc} format={(v) => v.toFixed(2)} />}
            {problemId === 'pendulum' && <Slider label="Initial angle θ₀ (rad)" value={theta0} min={0.1} max={3.1} step={0.05} onChange={setTheta0} format={(v) => v.toFixed(2)} />}
            <Slider label="Step size Δt" value={logDt} min={-3} max={-0.5} step={0.05} onChange={setLogDt} format={() => fmt(dtEff, 3)} />
            <Slider label="Duration (periods)" value={periods} min={1} max={200} step={1} onChange={setPeriods} />
            <Slider label="RK45 tolerance" value={tol} min={-12} max={-2} step={1} onChange={setTol} format={(v) => `1e${v}`} />
            <fieldset className="space-y-1.5">
              <legend className="mb-1 text-sm text-slate-300">Integrators</legend>
              {ALL.map((m) => (
                <Check key={m} label={NAME[m]} color={METHOD_COLOR[m]} checked={enabled[m]} onChange={(v) => setEnabled((s) => ({ ...s, [m]: v }))} />
              ))}
            </fieldset>
            <button type="button" className="btn w-full" onClick={() => setPlaying((p) => !p)}>
              {playing ? 'Pause animation' : 'Play animation'}
            </button>
          </div>
        </Panel>

        <div className="grid min-w-0 gap-4 xl:grid-cols-2">
          <Panel title={problem.id === 'kepler' ? 'Orbit (x, y)' : 'Phase portrait (q, p)'}>
            <Plot ariaLabel="Trajectories" series={trajSeries} equal height={300} xLabel={problem.id === 'kepler' ? 'x' : 'q'} yLabel={problem.id === 'kepler' ? 'y' : 'p'} />
          </Panel>
          <Panel title="Relative energy error |ΔE / E₀|">
            <Plot ariaLabel="Energy drift" series={energySeries} yLog height={300} xLabel="t" yLabel="|ΔE/E₀|" xDomain={[0, T]} />
          </Panel>
          <Panel title="Measured convergence order (global error at t = one period)" className="xl:col-span-2">
            <div className="grid gap-4 md:grid-cols-[minmax(0,1fr)_260px]">
              <Plot ariaLabel="Convergence order log-log plot" series={convSeries} xLog yLog height={280} xLabel="step size h" yLabel="max-norm error" />
              <table className="h-fit w-full text-sm">
                <thead>
                  <tr className="text-left text-xs text-slate-400">
                    <th className="py-1 font-medium">Method</th>
                    <th className="py-1 text-right font-medium">Expected</th>
                    <th className="py-1 text-right font-medium">Measured</th>
                  </tr>
                </thead>
                <tbody className="font-mono">
                  {convergence.map((c) => (
                    <tr key={c.method} className="border-t border-white/5">
                      <td className="py-1 font-sans text-slate-200">
                        <span className="mr-1.5 inline-block h-0.5 w-3 align-middle" style={{ background: METHOD_COLOR[c.method] }} />
                        {METHODS[c.method].short}
                      </td>
                      <td className="py-1 text-right text-slate-400">{METHODS[c.method].order}</td>
                      <td className={`py-1 text-right ${Math.abs(c.order - METHODS[c.method].order) < 0.2 ? 'text-emerald-300' : 'text-amber-300'}`}>{c.order.toFixed(3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Panel>
        </div>
      </div>

      <Panel title="Run statistics" className="mt-4">
        <div className="overflow-x-auto">
          <table className="w-full min-w-[640px] text-sm">
            <thead>
              <tr className="text-left text-xs text-slate-400">
                <th className="py-1.5 font-medium">Integrator</th>
                <th className="py-1.5 text-right font-medium">Steps</th>
                <th className="py-1.5 text-right font-medium">f evaluations</th>
                <th className="py-1.5 text-right font-medium">max |ΔE/E₀|</th>
                <th className="py-1.5 text-right font-medium">Final-state error</th>
                <th className="py-1.5 text-right font-medium">Time (ms)</th>
              </tr>
            </thead>
            <tbody className="font-mono">
              {runs.map((r) => (
                <tr key={r.method} className="border-t border-white/5">
                  <td className="py-1.5 font-sans text-slate-200">
                    <span className="mr-2 inline-block h-0.5 w-3 align-middle" style={{ background: METHOD_COLOR[r.method] }} />
                    {NAME[r.method]}
                  </td>
                  <td className="py-1.5 text-right">{r.steps.toLocaleString()}</td>
                  <td className="py-1.5 text-right">{Number.isFinite(r.fevals) ? r.fevals.toLocaleString() : 'Newton'}</td>
                  <td className="py-1.5 text-right">{fmt(r.maxDE)}</td>
                  <td className="py-1.5 text-right">{fmt(r.finalErr)}</td>
                  <td className="py-1.5 text-right">{r.ms.toFixed(1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-2 text-xs text-slate-500">
          Final-state error is measured against {problem.exact ? 'the analytic solution' : 'a Dormand–Prince reference at rtol = 1e-13'} at t = {fmt(T)}.
          {dtEff > dt && ' Step size was raised to cap the run at 400k steps.'}
        </p>
      </Panel>

      <Panel
        title="Stiffness: explicit vs implicit"
        className="mt-4"
        actions={
          <button type="button" className="btn btn-primary" disabled={stiff.busy} onClick={() => stiff.run({ problem: stiffProblem, mu: 10 ** logMu, h: 10 ** stiffLogH })}>
            {stiff.busy ? 'Solving in a Web Worker…' : 'Run experiment'}
          </button>
        }
      >
        <div className="grid gap-4 lg:grid-cols-[260px_minmax(0,1fr)]">
          <div className="space-y-4">
            <Segmented
              label="Stiff problem"
              value={stiffProblem}
              onChange={setStiffProblem}
              options={[
                { value: 'vanderpol', label: 'Van der Pol' },
                { value: 'robertson', label: 'Robertson' },
              ]}
            />
            {stiffProblem === 'vanderpol' && <Slider label="Stiffness μ" value={logMu} min={0} max={3} step={0.1} onChange={setLogMu} format={(v) => fmt(10 ** v, 3)} />}
            <Slider label="Fixed step h (RK4 & BE)" value={stiffLogH} min={-4} max={0} step={0.1} onChange={setStiffLogH} format={(v) => fmt(10 ** v, 2)} />
            <p className="text-xs leading-relaxed text-slate-400">
              Explicit methods are stable only while h·|λ| stays inside their stability region; for Van der Pol |λ| ≈ 3μ on the slow branch. Backward Euler is L-stable: any h
              works, accuracy permitting.
            </p>
          </div>
          <div className="min-w-0">
            {stiff.result ? (
              <StiffResult res={stiff.result} problem={stiffProblem} />
            ) : (
              <p className="py-10 text-center text-sm text-slate-400">{stiff.busy ? 'Integrating…' : 'Press “Run experiment”. Large μ makes RK45 take millions of steps.'}</p>
            )}
            {stiff.error && <p className="text-sm text-rose-300">{stiff.error}</p>}
          </div>
        </div>
      </Panel>

      <MethodSection
        references={[
          { text: 'E. Hairer, S. P. Nørsett, G. Wanner — Solving Ordinary Differential Equations I: Nonstiff Problems, 2nd ed., Springer (1993).' },
          { text: 'E. Hairer, G. Wanner — Solving Ordinary Differential Equations II: Stiff and Differential-Algebraic Problems, Springer (1996).' },
          { text: 'J. R. Dormand, P. J. Prince — A family of embedded Runge–Kutta formulae, J. Comp. Appl. Math. 6 (1980) 19–26.' },
          { text: 'H. Yoshida — Construction of higher order symplectic integrators, Phys. Lett. A 150 (1990) 262–268.' },
          { text: 'E. Hairer, C. Lubich, G. Wanner — Geometric Numerical Integration, Springer (2006).' },
        ]}
      >
        <p>
          We solve <Tex>{String.raw`\dot y = f(t, y)`}</Tex>. A one-step method of order <Tex>p</Tex> has global error <Tex>{String.raw`\|y_N - y(T)\| = \mathcal O(h^p)`}</Tex>,
          so the slope of <Tex>{String.raw`\log \text{err}`}</Tex> against <Tex>{String.raw`\log h`}</Tex> measures <Tex>p</Tex>. Classic RK4:
        </p>
        <Tex block>{String.raw`k_1 = f(t_n, y_n),\; k_2 = f(t_n + \tfrac h2, y_n + \tfrac h2 k_1),\; k_3 = f(t_n + \tfrac h2, y_n + \tfrac h2 k_2),\; k_4 = f(t_n + h, y_n + h k_3),\quad y_{n+1} = y_n + \tfrac h6 (k_1 + 2k_2 + 2k_3 + k_4)`}</Tex>
        <p>
          <strong>Dormand–Prince RK5(4)</strong> computes a 5th- and an embedded 4th-order solution from the same 7 stages (the last stage is reused — FSAL). Their difference
          estimates the local error, which is scaled per component and the step adapted:
        </p>
        <Tex block>{String.raw`\text{err} = \sqrt{\tfrac1n \sum_i \Big(\frac{y_i - \hat y_i}{\text{atol} + \text{rtol}\,\max(|y_i|, |y_i^{\text{new}}|)}\Big)^2},\qquad h_{\text{new}} = h \cdot \min\!\big(5, \max(0.2,\; 0.9\,\text{err}^{-1/5})\big)`}</Tex>
        <p>
          <strong>Velocity Verlet</strong> for <Tex>{String.raw`H = \tfrac12 p^2 + V(q)`}</Tex> is a composition of exact flows (kick–drift–kick), hence symplectic:
        </p>
        <Tex block>{String.raw`p_{n+\frac12} = p_n - \tfrac h2 \nabla V(q_n),\quad q_{n+1} = q_n + h\, p_{n+\frac12},\quad p_{n+1} = p_{n+\frac12} - \tfrac h2 \nabla V(q_{n+1})`}</Tex>
        <p>
          Backward error analysis shows it exactly conserves a modified Hamiltonian <Tex>{String.raw`\tilde H = H + \mathcal O(h^2)`}</Tex>, so the energy error stays bounded for
          exponentially long times instead of drifting. Yoshida’s triple jump composes three Verlet steps with weights{' '}
          <Tex>{String.raw`w_1 = \frac{1}{2 - 2^{1/3}},\; w_0 = -\frac{2^{1/3}}{2 - 2^{1/3}}`}</Tex> to reach order 4 while staying symplectic.
        </p>
        <p>
          <strong>Backward Euler</strong> solves <Tex>{String.raw`G(z) = z - y_n - h f(t_{n+1}, z) = 0`}</Tex> with Newton’s method,{' '}
          <Tex>{String.raw`(I - h J)\,\delta = -G(z)`}</Tex>, using an analytic Jacobian and an LU factorisation with partial pivoting. Its stability function{' '}
          <Tex>{String.raw`R(z) = 1/(1 - z)`}</Tex> satisfies <Tex>{String.raw`|R| < 1`}</Tex> on the whole left half-plane and <Tex>{String.raw`R(-\infty) = 0`}</Tex> (L-stability).
          The adaptive variant estimates the local error by step doubling.
        </p>
      </MethodSection>
    </div>
  )
}

function StiffResult({ res, problem }: { res: StiffResponse; problem: StiffRequest['problem'] }) {
  const colors: Record<string, string> = { rk4: METHOD_COLOR.rk4, 'backward-euler': METHOD_COLOR['backward-euler'], dopri45: METHOD_COLOR.dopri45, 'be-adaptive': C.aqua }
  const series: Series[] = res.runs.map((r) => ({
    x: problem === 'robertson' ? r.t.map((t) => Math.max(t, 1e-6)) : r.t,
    y: r.y[problem === 'robertson' ? 1 : 0].map((v) => (problem === 'robertson' ? v * 1e4 : v)),
    color: colors[r.id],
    label: r.name,
    width: r.id === 'be-adaptive' ? 2.5 : 1.5,
    dash: r.id === 'backward-euler' ? [6, 4] : undefined,
    points: r.id === 'be-adaptive' && r.t.length < 400,
    radius: 2,
  }))
  return (
    <div className="space-y-3">
      <Plot
        ariaLabel="Stiff solution comparison"
        series={series}
        height={260}
        xLog={problem === 'robertson'}
        yDomain={problem === 'vanderpol' ? [-3, 3] : undefined}
        xLabel="t"
        yLabel={problem === 'robertson' ? 'y₂ × 10⁴' : 'y₁'}
      />
      <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
        {res.runs.map((r) => (
          <Readout
            key={r.id}
            label={r.name.split(' (')[0].replace(', fixed h', ' (fixed)')}
            tone={r.status === 'ok' ? 'good' : 'bad'}
            value={
              <span>
                {r.steps.toLocaleString()} steps · {r.ms.toFixed(0)} ms
                <br />
                <span className="text-xs text-slate-400">{r.status}</span>
              </span>
            }
          />
        ))}
      </div>
    </div>
  )
}
