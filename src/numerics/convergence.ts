/**
 * Empirical order of accuracy: integrate to time T with a sequence of step sizes,
 * measure the global error against a reference solution and fit the slope of
 * log(error) against log(h).
 */
import { dopri45, integrateFixed, type MethodId } from './ode.ts'
import { logLogSlope } from './stats.ts'
import type { ModelProblem } from './systems.ts'

export interface ConvergenceResult {
  method: MethodId
  h: number[]
  error: number[]
  /** Fitted slope of log error vs log h — the observed order. */
  order: number
  r2: number
}

/** Max-norm global error at T for a list of step counts. */
export function convergenceStudy(method: MethodId, problem: ModelProblem, T: number, stepCounts: number[], reference?: Float64Array): ConvergenceResult {
  const ref = reference ?? referenceSolution(problem, T)
  const h: number[] = []
  const error: number[] = []
  for (const n of stepCounts) {
    const dt = T / n
    const y = integrateFixed(method, problem, problem.y0, 0, dt, n)
    let e = 0
    for (let i = 0; i < y.length; i++) e = Math.max(e, Math.abs(y[i] - ref[i]))
    h.push(dt)
    error.push(Number.isFinite(e) ? e : Infinity)
  }
  const fit = logLogSlope(h, error)
  return { method, h, error, order: fit.slope, r2: fit.r2 }
}

/** Exact solution when known, otherwise Dormand–Prince at near machine-precision tolerances. */
export function referenceSolution(problem: ModelProblem, T: number): Float64Array {
  if (problem.exact) return problem.exact(T)
  return dopri45(problem, 0, problem.y0, T, { rtol: 1e-13, atol: 1e-15, record: false }).yEnd
}

/** Relative energy error |E(t) − E₀|/|E₀| sampled along a fixed-step run. */
export function energyDrift(method: MethodId, problem: ModelProblem, dt: number, steps: number, samples = 400): { t: number[]; rel: number[]; max: number } {
  if (!problem.energy) throw new Error('problem has no energy function')
  const E0 = problem.energy(problem.y0)
  const t: number[] = []
  const rel: number[] = []
  let max = 0
  const every = Math.max(1, Math.floor(steps / samples))
  integrateFixed(
    method,
    problem,
    problem.y0,
    0,
    dt,
    steps,
    (tt, y) => {
      const r = Math.abs((problem.energy!(y) - E0) / E0)
      t.push(tt)
      rel.push(r)
      if (r > max || !Number.isFinite(r)) max = Number.isFinite(r) ? r : Infinity
    },
    every,
  )
  return { t, rel, max }
}
