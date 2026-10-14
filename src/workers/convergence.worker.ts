import { convergenceStudy, referenceSolution } from '../numerics/convergence.ts'
import type { MethodId } from '../numerics/ode.ts'
import { harmonic, kepler, pendulum } from '../numerics/systems.ts'
import { workerScope } from './scope.ts'

export interface ConvergenceRequest {
  problem: 'kepler' | 'pendulum' | 'harmonic'
  ecc: number
  theta0: number
}

const FIXED: MethodId[] = ['euler', 'midpoint', 'rk4', 'verlet', 'yoshida4', 'backward-euler']

const scope = workerScope<ConvergenceRequest>()
scope.onmessage = (e) => {
  const { problem, ecc, theta0 } = e.data
  const p = problem === 'kepler' ? kepler(ecc) : problem === 'pendulum' ? pendulum(theta0) : harmonic()
  // 0.73 of a period: avoids the error cancellation that can occur at whole periods.
  const T = 0.73 * p.period
  const ref = referenceSolution(p, T)
  // Step counts chosen so each method sits in its asymptotic regime but above the
  // reference solution's ~1e-12 accuracy floor. First-order methods need many more steps.
  const results = FIXED.map((m) => {
    const firstOrder = m === 'euler' || m === 'backward-euler'
    const base = firstOrder ? (problem === 'kepler' ? 10000 : 2000) : problem === 'kepler' ? 500 : problem === 'pendulum' ? 250 : 100
    const counts = problem === 'pendulum' && !firstOrder ? [base, base * 2, base * 4, base * 8] : [base, base * 2, base * 4, base * 8, base * 16]
    return convergenceStudy(m, p, T, counts, ref)
  })
  scope.postMessage(results)
}
