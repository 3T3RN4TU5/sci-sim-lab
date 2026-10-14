/**
 * 2D Ising model on an L×L periodic square lattice, H = −J Σ⟨ij⟩ sᵢsⱼ (J = 1, k_B = 1).
 * Metropolis single-spin-flip and Wolff cluster updates, observables, temperature sweeps,
 * and Onsager's exact results for comparison.
 */
import { Rng } from './rng.ts'
import { ellipticK } from './systems.ts'

export const T_CRITICAL = 2 / Math.log(1 + Math.SQRT2) // ≈ 2.269185

export type IsingAlgorithm = 'metropolis' | 'wolff'

export class Ising {
  readonly L: number
  readonly N: number
  readonly s: Int8Array
  readonly rng: Rng
  T: number
  /** Running totals, kept in sync by every update. */
  energy = 0
  magnetization = 0
  private stack: Int32Array
  private inCluster: Uint8Array
  private acc4 = 0
  private acc8 = 0
  private padd = 0

  constructor(L: number, T: number, seed = 1, start: 'hot' | 'cold' = 'cold') {
    this.L = L
    this.N = L * L
    this.s = new Int8Array(this.N)
    this.rng = new Rng(seed)
    this.T = T
    this.stack = new Int32Array(this.N)
    this.inCluster = new Uint8Array(this.N)
    this.reset(start)
    this.setTemperature(T)
  }

  reset(start: 'hot' | 'cold'): void {
    for (let k = 0; k < this.N; k++) this.s[k] = start === 'cold' ? 1 : this.rng.next() < 0.5 ? 1 : -1
    this.recompute()
  }

  setTemperature(T: number): void {
    this.T = T
    const beta = 1 / T
    this.acc4 = Math.exp(-4 * beta)
    this.acc8 = Math.exp(-8 * beta)
    this.padd = 1 - Math.exp(-2 * beta)
  }

  /** Recompute E and M from scratch. */
  recompute(): void {
    const { L, s } = this
    let E = 0
    let M = 0
    for (let y = 0; y < L; y++) {
      for (let x = 0; x < L; x++) {
        const k = y * L + x
        M += s[k]
        E -= s[k] * (s[y * L + ((x + 1) % L)] + s[((y + 1) % L) * L + x])
      }
    }
    this.energy = E
    this.magnetization = M
  }

  private neighbourSum(k: number): number {
    const L = this.L
    const x = k % L
    const y = (k - x) / L
    const s = this.s
    return s[y * L + (x + 1 === L ? 0 : x + 1)] + s[y * L + (x === 0 ? L - 1 : x - 1)] + s[(y + 1 === L ? 0 : y + 1) * L + x] + s[(y === 0 ? L - 1 : y - 1) * L + x]
  }

  /** One Metropolis sweep = N attempted single-spin flips. Returns the acceptance rate. */
  metropolisSweep(): number {
    const { N, s, rng } = this
    let accepted = 0
    for (let n = 0; n < N; n++) {
      const k = rng.int(N)
      const dE = 2 * s[k] * this.neighbourSum(k)
      if (dE <= 0 || (dE === 4 ? rng.next() < this.acc4 : rng.next() < this.acc8)) {
        s[k] = -s[k]
        this.energy += dE
        this.magnetization += 2 * s[k]
        accepted++
      }
    }
    return accepted / N
  }

  private neighbours(k: number, out: Int32Array): void {
    const L = this.L
    const x = k % L
    const y = (k - x) / L
    out[0] = y * L + (x + 1 === L ? 0 : x + 1)
    out[1] = y * L + (x === 0 ? L - 1 : x - 1)
    out[2] = (y + 1 === L ? 0 : y + 1) * L + x
    out[3] = (y === 0 ? L - 1 : y - 1) * L + x
  }

  private nb = new Int32Array(4)
  private members: Int32Array | null = null

  /**
   * One Wolff cluster flip (bond probability p = 1 − e^(−2β)). Grows the cluster with a
   * stack, flips it, and updates E and M incrementally from the cluster boundary, so the
   * cost is O(cluster size). Returns the cluster size.
   */
  wolffStep(): number {
    const { N, s, rng, stack, inCluster, nb } = this
    const members = (this.members ??= new Int32Array(N))
    const seed = rng.int(N)
    const spin = s[seed]
    const p = this.padd
    let sp = 0
    let size = 0
    stack[sp++] = seed
    inCluster[seed] = 1
    members[size++] = seed
    while (sp > 0) {
      const k = stack[--sp]
      this.neighbours(k, nb)
      for (let q = 0; q < 4; q++) {
        const j = nb[q]
        if (!inCluster[j] && s[j] === spin && rng.next() < p) {
          inCluster[j] = 1
          members[size++] = j
          stack[sp++] = j
        }
      }
    }
    // Energy change comes only from bonds that cross the cluster boundary.
    let dE = 0
    for (let c = 0; c < size; c++) {
      this.neighbours(members[c], nb)
      for (let q = 0; q < 4; q++) if (!inCluster[nb[q]]) dE += 2 * spin * s[nb[q]]
    }
    for (let c = 0; c < size; c++) {
      const k = members[c]
      s[k] = -spin
      inCluster[k] = 0
    }
    this.energy += dE
    this.magnetization -= 2 * spin * size
    return size
  }

  /**
   * Advance by roughly one sweep's worth of work. For Wolff, flips clusters until the
   * total number of flipped spins reaches N (so both algorithms are compared per sweep).
   */
  sweep(algo: IsingAlgorithm): void {
    if (algo === 'metropolis') {
      this.metropolisSweep()
      return
    }
    let flipped = 0
    while (flipped < this.N) flipped += this.wolffStep()
  }
}

export interface SweepPoint {
  T: number
  /** ⟨|m|⟩ per spin */
  m: number
  /** ⟨e⟩ per spin */
  e: number
  /** χ = βN(⟨m²⟩ − ⟨|m|⟩²) */
  chi: number
  /** C = β²N(⟨e²⟩ − ⟨e⟩²) */
  c: number
  /** Binder cumulant U = 1 − ⟨m⁴⟩/(3⟨m²⟩²) */
  binder: number
}

/** Equilibrate then sample at one temperature. */
export function measure(model: Ising, algo: IsingAlgorithm, equil: number, samples: number): SweepPoint {
  for (let i = 0; i < equil; i++) model.sweep(algo)
  const N = model.N
  let sm = 0, sm2 = 0, sm4 = 0, se = 0, se2 = 0
  for (let i = 0; i < samples; i++) {
    model.sweep(algo)
    const m = Math.abs(model.magnetization) / N
    const e = model.energy / N
    sm += m
    sm2 += m * m
    sm4 += m * m * m * m
    se += e
    se2 += e * e
  }
  sm /= samples
  sm2 /= samples
  sm4 /= samples
  se /= samples
  se2 /= samples
  const beta = 1 / model.T
  return {
    T: model.T,
    m: sm,
    e: se,
    chi: beta * N * (sm2 - sm * sm),
    c: beta * beta * N * (se2 - se * se),
    binder: 1 - sm4 / (3 * sm2 * sm2),
  }
}

/** Onsager/Yang spontaneous magnetisation per spin (0 above T_c). */
export function onsagerMagnetization(T: number): number {
  if (T >= T_CRITICAL) return 0
  const s = Math.sinh(2 / T)
  return Math.pow(1 - Math.pow(s, -4), 1 / 8)
}

/** Exact internal energy per spin of the infinite lattice (Onsager 1944). */
export function onsagerEnergy(T: number): number {
  const b = 1 / T
  const k = (2 * Math.sinh(2 * b)) / Math.cosh(2 * b) ** 2
  const kk = Math.min(k, 1 - 1e-15)
  const K = ellipticK(kk)
  const t = Math.tanh(2 * b)
  return -(1 / Math.tanh(2 * b)) * (1 + (2 / Math.PI) * (2 * t * t - 1) * K)
}
