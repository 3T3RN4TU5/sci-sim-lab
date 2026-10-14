/**
 * Tools for chaotic dynamics: Benettin's two-trajectory estimate of the largest
 * Lyapunov exponent for flows, and the logistic map (bifurcation diagram and
 * its exact-derivative Lyapunov exponent).
 */
import { makeStepper, type MethodId, type OdeProblem, type Stepper } from './ode.ts'

export interface LyapunovOptions {
  /** Initial (and renormalised) separation of the shadow trajectory. */
  d0?: number
  /** Integration step. */
  h?: number
  /** Renormalise every k steps. */
  renormEvery?: number
  method?: MethodId
}

/**
 * Benettin et al. (1980): follow a reference and a shadow trajectory separated by d₀;
 * every k steps measure the separation d, accumulate ln(d/d₀) and pull the shadow back
 * to distance d₀ along the current separation direction. λ₁ ≈ Σ ln(dᵢ/d₀) / t.
 */
export class LyapunovEstimator {
  readonly y: Float64Array
  readonly shadow: Float64Array
  t = 0
  sumLog = 0
  readonly d0: number
  readonly h: number
  readonly renormEvery: number
  private readonly step: Stepper
  private readonly stepShadow: Stepper
  private count = 0

  constructor(p: OdeProblem, y0: ArrayLike<number>, opts: LyapunovOptions = {}) {
    this.d0 = opts.d0 ?? 1e-8
    this.h = opts.h ?? 0.005
    this.renormEvery = opts.renormEvery ?? 10
    const method = opts.method ?? 'rk4'
    this.step = makeStepper(method, p)
    this.stepShadow = makeStepper(method, p)
    this.y = Float64Array.from(y0)
    this.shadow = Float64Array.from(y0)
    // Perturb along a fixed generic direction.
    let norm = 0
    for (let i = 0; i < p.dim; i++) norm += (1 + i) ** 2
    norm = Math.sqrt(norm)
    for (let i = 0; i < p.dim; i++) this.shadow[i] += (this.d0 * (1 + i)) / norm
  }

  separation(): number {
    let s = 0
    for (let i = 0; i < this.y.length; i++) s += (this.shadow[i] - this.y[i]) ** 2
    return Math.sqrt(s)
  }

  advance(steps: number): void {
    for (let s = 0; s < steps; s++) {
      this.step(this.t, this.y, this.h)
      this.stepShadow(this.t, this.shadow, this.h)
      this.t += this.h
      if (++this.count % this.renormEvery === 0) {
        const d = this.separation()
        this.sumLog += Math.log(d / this.d0)
        const f = this.d0 / d
        for (let i = 0; i < this.y.length; i++) this.shadow[i] = this.y[i] + (this.shadow[i] - this.y[i]) * f
      }
    }
  }

  /** Current estimate of the largest Lyapunov exponent. */
  get exponent(): number {
    const tr = Math.floor(this.count / this.renormEvery) * this.renormEvery * this.h
    return tr > 0 ? this.sumLog / tr : 0
  }
}

/** Logistic map x ↦ r·x·(1 − x). */
export function logistic(r: number, x: number): number {
  return r * x * (1 - x)
}

/**
 * Bifurcation diagram: for `cols` values of r in [rMin, rMax], iterate past a transient
 * and keep `keep` successive iterates. Returns a cols×keep Float32Array (column-major by r).
 */
export function bifurcation(rMin: number, rMax: number, cols: number, transient = 600, keep = 200, x0 = 0.5): Float32Array {
  const out = new Float32Array(cols * keep)
  for (let c = 0; c < cols; c++) {
    const r = rMin + ((rMax - rMin) * c) / Math.max(1, cols - 1)
    let x = x0
    for (let i = 0; i < transient; i++) x = r * x * (1 - x)
    for (let i = 0; i < keep; i++) {
      x = r * x * (1 - x)
      out[c * keep + i] = x
    }
  }
  return out
}

/** Lyapunov exponent of the logistic map: λ(r) = ⟨ln |r(1 − 2x)|⟩ along the orbit. */
export function logisticLyapunov(r: number, n = 4000, transient = 500, x0 = 0.3): number {
  let x = x0
  for (let i = 0; i < transient; i++) x = r * x * (1 - x)
  let s = 0
  for (let i = 0; i < n; i++) {
    x = r * x * (1 - x)
    s += Math.log(Math.max(1e-300, Math.abs(r * (1 - 2 * x))))
  }
  return s / n
}
