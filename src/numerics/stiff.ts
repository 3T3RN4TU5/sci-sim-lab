/**
 * The stiffness experiment behind the Integrator lab: the same stiff problem solved by
 * explicit RK4 and implicit backward Euler at one fixed step, and by the adaptive
 * Dormand–Prince and adaptive backward Euler solvers. Pure function, run in a Web Worker.
 */
import { backwardEulerAdaptive, dopri45, integrateFixed, type MethodId } from './ode.ts'
import { robertson, vanDerPol } from './systems.ts'

export interface StiffRequest {
  problem: 'vanderpol' | 'robertson'
  mu: number
  /** Fixed step for the RK4 / backward Euler runs. */
  h: number
}

export interface StiffRun {
  id: 'rk4' | 'backward-euler' | 'dopri45' | 'be-adaptive'
  name: string
  steps: number
  rejected: number
  fevals: number
  ms: number
  status: string
  t: number[]
  /** Sampled components, one array per state variable. */
  y: number[][]
}

export interface StiffResponse {
  T: number
  runs: StiffRun[]
}

const SAMPLES = 600

export function runStiffDemo(req: StiffRequest): StiffResponse {
  const p = req.problem === 'vanderpol' ? vanDerPol(req.mu) : robertson()
  const T = req.problem === 'vanderpol' ? Math.round(1.5 * (3 - 2 * Math.log(2)) * req.mu + 10) : 40
  const dim = p.dim
  const runs: StiffRun[] = []
  const empty = () => Array.from({ length: dim }, () => [] as number[])

  for (const id of ['rk4', 'backward-euler'] as MethodId[]) {
    const t0 = performance.now()
    const steps = Math.ceil(T / req.h)
    const every = Math.max(1, Math.floor(steps / SAMPLES))
    const t: number[] = []
    const y = empty()
    let diverged = -1
    integrateFixed(
      id,
      p,
      p.y0,
      0,
      req.h,
      steps,
      (tt, yy) => {
        const bad = !yy.every((v) => Number.isFinite(v) && Math.abs(v) < 1e8)
        if (bad && diverged < 0) diverged = tt
        if (diverged >= 0 && tt > diverged) return
        t.push(tt)
        for (let i = 0; i < dim; i++) y[i].push(bad ? NaN : yy[i])
      },
      every,
    )
    runs.push({
      id: id as 'rk4' | 'backward-euler',
      name: id === 'rk4' ? 'RK4, fixed h (explicit)' : 'Backward Euler, fixed h (implicit)',
      steps,
      rejected: 0,
      fevals: id === 'rk4' ? 4 * steps : NaN,
      ms: performance.now() - t0,
      status: diverged >= 0 ? `diverged at t ≈ ${diverged.toPrecision(3)}` : 'ok',
      t,
      y,
    })
  }

  {
    const t0 = performance.now()
    const outT = Array.from({ length: SAMPLES + 1 }, (_, i) => (T * i) / SAMPLES)
    const r = dopri45(p, 0, p.y0, T, { rtol: 1e-6, atol: 1e-9, record: false, denseAt: outT, maxSteps: 3_000_000 })
    const y = empty()
    r.dense.forEach((v) => v.forEach((c, i) => y[i].push(c)))
    runs.push({ id: 'dopri45', name: 'Dormand–Prince RK45 (adaptive, explicit)', steps: r.accepted, rejected: r.rejected, fevals: r.fevals, ms: performance.now() - t0, status: r.status, t: outT.slice(0, r.dense.length), y })
  }
  {
    const t0 = performance.now()
    const r = backwardEulerAdaptive(p, 0, p.y0, T, { rtol: 1e-4, atol: 1e-8, record: true, h0: 1e-6, maxSteps: 200_000 })
    const y = empty()
    r.y.forEach((v) => v.forEach((c, i) => y[i].push(c)))
    runs.push({ id: 'be-adaptive', name: 'Backward Euler (adaptive, implicit)', steps: r.accepted, rejected: r.rejected, fevals: r.fevals, ms: performance.now() - t0, status: r.status, t: r.t, y })
  }
  return { T, runs }
}
