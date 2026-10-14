/**
 * Monte Carlo estimation and random walks: π by hit-or-miss and mean-value sampling,
 * quasi-Monte Carlo on a low-discrepancy sequence, error-scaling studies, and lattice random walks
 * with mean-squared displacement.
 */
import { Rng, radicalInverse } from './rng.ts'
import { logLogSlope } from './stats.ts'

/** Hit-or-miss estimate of π from the first n points of a pseudo-random stream. */
export function estimatePi(n: number, rng: Rng): number {
  let hits = 0
  for (let i = 0; i < n; i++) {
    const x = rng.next()
    const y = rng.next()
    if (x * x + y * y < 1) hits++
  }
  return (4 * hits) / n
}

/** Mean-value Monte Carlo estimate of π = 4∫₀¹ √(1 − x²) dx with pseudo-random x. */
export function estimatePiMeanValue(n: number, rng: Rng): number {
  let s = 0
  for (let i = 0; i < n; i++) {
    const x = rng.next()
    s += Math.sqrt(1 - x * x)
  }
  return (4 * s) / n
}

/**
 * Quasi-Monte Carlo estimate of the same integral on the base-2 van der Corput sequence
 * (the 1D Halton sequence), with a random shift (Cranley–Patterson rotation) so that
 * independent replicas give an unbiased error estimate. Error ~ (log N)/N.
 */
export function estimatePiQmc(n: number, shift: number): number {
  let s = 0
  for (let i = 0; i < n; i++) {
    let x = radicalInverse(i + 1, 2) + shift
    if (x >= 1) x -= 1
    s += Math.sqrt(1 - x * x)
  }
  return (4 * s) / n
}

export interface ErrorScaling {
  n: number[]
  /** RMS error of hit-or-miss sampling of the quarter disk. */
  hitMiss: number[]
  /** RMS error of mean-value sampling of √(1 − x²). */
  meanValue: number[]
  /** RMS error of randomly shifted quasi-Monte Carlo. */
  qmc: number[]
  hitMissSlope: number
  meanValueSlope: number
  qmcSlope: number
}

/** RMS error vs N for N = 2^kMin … 2^kMax, over `runs` independent replicas per N. */
export function piErrorScaling(kMin = 6, kMax = 16, runs = 32, seed = 1): ErrorScaling {
  const n: number[] = []
  const hitMiss: number[] = []
  const meanValue: number[] = []
  const qmc: number[] = []
  const rng = new Rng(seed)
  for (let k = kMin; k <= kMax; k++) {
    const N = 2 ** k
    let a = 0, b = 0, c = 0
    for (let r = 0; r < runs; r++) {
      a += (estimatePi(N, rng) - Math.PI) ** 2
      b += (estimatePiMeanValue(N, rng) - Math.PI) ** 2
      c += (estimatePiQmc(N, rng.next()) - Math.PI) ** 2
    }
    n.push(N)
    hitMiss.push(Math.sqrt(a / runs))
    meanValue.push(Math.sqrt(b / runs))
    qmc.push(Math.sqrt(c / runs))
  }
  return {
    n,
    hitMiss,
    meanValue,
    qmc,
    hitMissSlope: logLogSlope(n, hitMiss).slope,
    meanValueSlope: logLogSlope(n, meanValue).slope,
    qmcSlope: logLogSlope(n, qmc).slope,
  }
}

/**
 * Ensemble of 2D lattice random walkers (unit steps in ±x or ±y).
 * For unit steps MSD(t) = t exactly in expectation, i.e. D = 1/4 in 2D (MSD = 4Dt).
 */
export class RandomWalkers {
  readonly count: number
  readonly x: Int32Array
  readonly y: Int32Array
  t = 0
  readonly rng: Rng

  constructor(count: number, seed = 1) {
    this.count = count
    this.x = new Int32Array(count)
    this.y = new Int32Array(count)
    this.rng = new Rng(seed)
  }

  step(): void {
    const { x, y, rng } = this
    for (let i = 0; i < this.count; i++) {
      const r = rng.nextU32() & 3
      if (r === 0) x[i]++
      else if (r === 1) x[i]--
      else if (r === 2) y[i]++
      else y[i]--
    }
    this.t++
  }

  msd(): number {
    let s = 0
    for (let i = 0; i < this.count; i++) s += this.x[i] * this.x[i] + this.y[i] * this.y[i]
    return s / this.count
  }
}
