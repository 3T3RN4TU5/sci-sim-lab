/**
 * Compartmental epidemic models (SIR, SEIR), integrated with classic RK4, plus synthetic
 * noisy observations and least-squares parameter estimation with Nelder–Mead.
 */
import { integrateFixed, type OdeProblem } from './ode.ts'
import { nelderMead, type NelderMeadResult } from './optimize.ts'
import { Rng } from './rng.ts'

export type EpiModel = 'SIR' | 'SEIR'

export interface EpiParams {
  /** Transmission rate per day. */
  beta: number
  /** Recovery rate per day (1/infectious period). */
  gamma: number
  /** Incubation rate per day (1/latent period), SEIR only. */
  sigma: number
  N: number
  I0: number
}

export interface EpiSeries {
  t: Float64Array
  S: Float64Array
  E: Float64Array
  I: Float64Array
  R: Float64Array
}

export function epiProblem(model: EpiModel, p: EpiParams): OdeProblem {
  const { beta, gamma, sigma, N } = p
  if (model === 'SIR') {
    return {
      dim: 3,
      rhs: (_t, y, out) => {
        const inf = (beta * y[0] * y[1]) / N
        out[0] = -inf
        out[1] = inf - gamma * y[1]
        out[2] = gamma * y[1]
      },
    }
  }
  return {
    dim: 4,
    rhs: (_t, y, out) => {
      const inf = (beta * y[0] * y[2]) / N
      out[0] = -inf
      out[1] = inf - sigma * y[1]
      out[2] = sigma * y[1] - gamma * y[2]
      out[3] = gamma * y[2]
    },
  }
}

/** Integrate for `days` days with RK4 (h = 1/substeps) and sample once per day. */
export function simulateEpidemic(model: EpiModel, p: EpiParams, days: number, substeps = 10): EpiSeries {
  const n = days + 1
  const out: EpiSeries = { t: new Float64Array(n), S: new Float64Array(n), E: new Float64Array(n), I: new Float64Array(n), R: new Float64Array(n) }
  const y0 = model === 'SIR' ? [p.N - p.I0, p.I0, 0] : [p.N - p.I0, 0, p.I0, 0]
  integrateFixed(
    'rk4',
    epiProblem(model, p),
    y0,
    0,
    1 / substeps,
    days * substeps,
    (t, y, step) => {
      const d = Math.round(step / substeps)
      out.t[d] = t
      out.S[d] = y[0]
      if (model === 'SIR') {
        out.I[d] = y[1]
        out.R[d] = y[2]
      } else {
        out.E[d] = y[1]
        out.I[d] = y[2]
        out.R[d] = y[3]
      }
    },
    substeps,
  )
  return out
}

export function basicReproductionNumber(p: Pick<EpiParams, 'beta' | 'gamma'>): number {
  return p.beta / p.gamma
}

/** Final attack rate z solving z = 1 − exp(−R₀z) (Newton; 0 when R₀ ≤ 1). */
export function finalSize(R0: number): number {
  if (R0 <= 1) return 0
  let z = 1 - 1e-6
  for (let i = 0; i < 100; i++) {
    const g = z - 1 + Math.exp(-R0 * z)
    const dg = 1 - R0 * Math.exp(-R0 * z)
    const dz = g / dg
    z -= dz
    if (Math.abs(dz) < 1e-14) break
  }
  return z
}

/** Synthetic observations of prevalence I(t) with multiplicative log-normal noise. */
export function syntheticData(model: EpiModel, p: EpiParams, days: number, noise: number, seed = 42): { t: number[]; I: number[] } {
  const s = simulateEpidemic(model, p, days)
  const rng = new Rng(seed)
  const t: number[] = []
  const I: number[] = []
  for (let d = 0; d <= days; d++) {
    t.push(d)
    I.push(Math.max(0, s.I[d] * Math.exp(noise * rng.normal())))
  }
  return { t, I }
}

export interface FitResult extends NelderMeadResult {
  beta: number
  gamma: number
  R0: number
}

/**
 * Least-squares fit of (β, γ) to prevalence data on a log scale,
 * J(β, γ) = Σ [ln(1 + I_obs) − ln(1 + I_model)]², optimised over (ln β, ln γ) so both stay positive.
 */
export function fitEpidemic(model: EpiModel, data: { t: number[]; I: number[] }, known: Omit<EpiParams, 'beta' | 'gamma'>, guess: { beta: number; gamma: number }, maxIter = 400): FitResult {
  const days = data.t[data.t.length - 1]
  const objective = (x: number[]) => {
    const beta = Math.exp(x[0])
    const gamma = Math.exp(x[1])
    const s = simulateEpidemic(model, { ...known, beta, gamma }, days, 4)
    let sse = 0
    for (let i = 0; i < data.t.length; i++) {
      const r = Math.log1p(data.I[i]) - Math.log1p(s.I[data.t[i]])
      sse += r * r
    }
    return sse
  }
  const res = nelderMead(objective, [Math.log(guess.beta), Math.log(guess.gamma)], { step: [0.4, 0.4], maxIter, history: true, fTol: 1e-12, xTol: 1e-9 })
  const beta = Math.exp(res.x[0])
  const gamma = Math.exp(res.x[1])
  return { ...res, beta, gamma, R0: beta / gamma }
}
