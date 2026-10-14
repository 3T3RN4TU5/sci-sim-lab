/**
 * Finite-difference solvers for 2D PDEs on uniform N×N grids (row-major, index j·N + i).
 *
 *  • Heat equation u_t = α∇²u, Dirichlet boundaries:
 *      – FTCS (explicit, stable only for r = αΔt/Δx² ≤ 1/4),
 *      – Peaceman–Rachford ADI (Crank–Nicolson-type, unconditionally stable, O(Δt², Δx²)).
 *  • Wave equation u_tt = c²∇²u: explicit leapfrog in time (CFL: CΔt/Δx ≤ 1/√2),
 *    reflecting (Dirichlet) or absorbing (first-order Mur) boundaries, walls, point sources.
 *  • Gray–Scott reaction–diffusion with periodic boundaries.
 */
import { thomasConst } from './linalg.ts'
import { Rng } from './rng.ts'

// ─── Heat ──────────────────────────────────────────────────────────────────

export type HeatScheme = 'ftcs' | 'adi'

export class Heat2D {
  readonly n: number
  u: Float64Array
  private tmp: Float64Array
  private line: Float64Array
  private scratch: Float64Array
  alpha: number
  /** Grid spacing; the domain is [0, (n−1)·dx]². */
  dx: number
  t = 0

  constructor(n: number, alpha = 1, length = 1) {
    this.n = n
    this.alpha = alpha
    this.dx = length / (n - 1)
    this.u = new Float64Array(n * n)
    this.tmp = new Float64Array(n * n)
    this.line = new Float64Array(n)
    this.scratch = new Float64Array(n)
  }

  /** Mesh ratio r = αΔt/Δx². FTCS is stable iff r ≤ 1/4 in 2D. */
  ratio(dt: number): number {
    return (this.alpha * dt) / (this.dx * this.dx)
  }

  /** The largest stable FTCS time step. */
  maxStableDt(): number {
    return (0.25 * this.dx * this.dx) / this.alpha
  }

  step(dt: number, scheme: HeatScheme): void {
    if (scheme === 'ftcs') this.stepFTCS(dt)
    else this.stepADI(dt)
    this.t += dt
  }

  stepFTCS(dt: number): void {
    const { n, u, tmp } = this
    const r = this.ratio(dt)
    tmp.set(u)
    for (let j = 1; j < n - 1; j++) {
      const row = j * n
      for (let i = 1; i < n - 1; i++) {
        const k = row + i
        u[k] = tmp[k] + r * (tmp[k - 1] + tmp[k + 1] + tmp[k - n] + tmp[k + n] - 4 * tmp[k])
      }
    }
  }

  /**
   * Peaceman–Rachford ADI:
   *   (I − r/2 δx²) u* = (I + r/2 δy²) uⁿ
   *   (I − r/2 δy²) uⁿ⁺¹ = (I + r/2 δx²) u*
   * Each half step is a set of independent tridiagonal solves (Thomas algorithm).
   */
  stepADI(dt: number): void {
    const { n, u, tmp, line, scratch } = this
    const h = 0.5 * this.ratio(dt)
    const m = n - 2
    // Boundary values are held fixed, so the intermediate field shares them.
    tmp.set(u)
    // Sweep 1: implicit in x for each interior row j.
    for (let j = 1; j < n - 1; j++) {
      const row = j * n
      for (let i = 1; i < n - 1; i++) {
        const k = row + i
        line[i - 1] = u[k] + h * (u[k - n] - 2 * u[k] + u[k + n])
      }
      line[0] += h * u[row]
      line[m - 1] += h * u[row + n - 1]
      thomasConst(-h, 1 + 2 * h, -h, line, m, scratch)
      for (let i = 1; i < n - 1; i++) tmp[row + i] = line[i - 1]
    }
    // Sweep 2: implicit in y for each interior column i.
    for (let i = 1; i < n - 1; i++) {
      for (let j = 1; j < n - 1; j++) {
        const k = j * n + i
        line[j - 1] = tmp[k] + h * (tmp[k - 1] - 2 * tmp[k] + tmp[k + 1])
      }
      line[0] += h * u[i]
      line[m - 1] += h * u[(n - 1) * n + i]
      thomasConst(-h, 1 + 2 * h, -h, line, m, scratch)
      for (let j = 1; j < n - 1; j++) u[j * n + i] = line[j - 1]
    }
  }

  /** Initialise with the (p, q) Fourier mode sin(pπx)·sin(qπy) of the unit square. */
  setMode(p = 1, q = 1, amp = 1): void {
    const { n } = this
    for (let j = 0; j < n; j++) {
      for (let i = 0; i < n; i++) {
        this.u[j * n + i] = amp * Math.sin((p * Math.PI * i) / (n - 1)) * Math.sin((q * Math.PI * j) / (n - 1))
      }
    }
    this.t = 0
  }

  /** Add a Gaussian blob of heat (or cold, amp < 0) centred at grid coordinates (ci, cj). */
  paint(ci: number, cj: number, radius: number, amp: number): void {
    const { n } = this
    const r2 = radius * radius
    const lo = Math.ceil(3 * radius)
    for (let j = Math.max(1, Math.floor(cj - lo)); j <= Math.min(n - 2, Math.ceil(cj + lo)); j++) {
      for (let i = Math.max(1, Math.floor(ci - lo)); i <= Math.min(n - 2, Math.ceil(ci + lo)); i++) {
        const d2 = (i - ci) ** 2 + (j - cj) ** 2
        this.u[j * n + i] += amp * Math.exp(-d2 / r2)
      }
    }
  }

  /** ∫u dA (trapezoid-free: plain sum × Δx²). */
  total(): number {
    let s = 0
    for (let k = 0; k < this.u.length; k++) s += this.u[k]
    return s * this.dx * this.dx
  }

  maxAbs(): number {
    let m = 0
    for (let k = 0; k < this.u.length; k++) m = Math.max(m, Math.abs(this.u[k]))
    return m
  }
}

/** Analytic amplitude of the (p, q) mode of the heat equation on the unit square. */
export function heatModeAmplitude(alpha: number, t: number, p = 1, q = 1): number {
  return Math.exp(-alpha * Math.PI * Math.PI * (p * p + q * q) * t)
}

// ─── Wave ──────────────────────────────────────────────────────────────────

export type WaveBoundary = 'reflect' | 'absorb'

export interface WaveSource {
  i: number
  j: number
  /** Angular frequency in radians per time step. */
  omega: number
  amp: number
}

export class Wave2D {
  readonly n: number
  u: Float64Array
  prev: Float64Array
  next: Float64Array
  /** Walls (reflecting obstacles), 1 = wall. */
  wall: Uint8Array
  /** Courant number C = cΔt/Δx (grid units). Stable for C ≤ 1/√2. */
  courant = 0.5
  damping = 0
  boundary: WaveBoundary = 'absorb'
  sources: WaveSource[] = []
  steps = 0

  constructor(n: number) {
    this.n = n
    this.u = new Float64Array(n * n)
    this.prev = new Float64Array(n * n)
    this.next = new Float64Array(n * n)
    this.wall = new Uint8Array(n * n)
  }

  clearField(): void {
    this.u.fill(0)
    this.prev.fill(0)
    this.next.fill(0)
    this.steps = 0
  }

  step(): void {
    const { n, u, prev, next, wall } = this
    const c2 = this.courant * this.courant
    const g = this.damping
    for (let j = 1; j < n - 1; j++) {
      const row = j * n
      for (let i = 1; i < n - 1; i++) {
        const k = row + i
        const lap = u[k - 1] + u[k + 1] + u[k - n] + u[k + n] - 4 * u[k]
        next[k] = (2 - g) * u[k] - (1 - g) * prev[k] + c2 * lap
      }
    }
    if (this.boundary === 'absorb') {
      // First-order Mur: one-way wave equation u_t = ±c u_x discretised at each edge.
      const C = this.courant
      const kk = (C - 1) / (C + 1)
      for (let j = 1; j < n - 1; j++) {
        const row = j * n
        next[row] = u[row + 1] + kk * (next[row + 1] - u[row])
        next[row + n - 1] = u[row + n - 2] + kk * (next[row + n - 2] - u[row + n - 1])
      }
      for (let i = 1; i < n - 1; i++) {
        next[i] = u[n + i] + kk * (next[n + i] - u[i])
        const b = (n - 1) * n + i
        next[b] = u[b - n] + kk * (next[b - n] - u[b])
      }
      next[0] = 0.5 * (next[1] + next[n])
      next[n - 1] = 0.5 * (next[n - 2] + next[2 * n - 1])
      next[(n - 1) * n] = 0.5 * (next[(n - 2) * n] + next[(n - 1) * n + 1])
      next[n * n - 1] = 0.5 * (next[n * n - 2] + next[(n - 1) * n - 1])
    } else {
      for (let i = 0; i < n; i++) {
        next[i] = 0
        next[(n - 1) * n + i] = 0
        next[i * n] = 0
        next[i * n + n - 1] = 0
      }
    }
    for (let k = 0; k < n * n; k++) if (wall[k]) next[k] = 0
    this.steps++
    for (const s of this.sources) {
      if (s.i > 0 && s.j > 0 && s.i < n - 1 && s.j < n - 1) next[s.j * n + s.i] = s.amp * Math.sin(s.omega * this.steps)
    }
    // Rotate buffers.
    this.prev = u
    this.u = next
    this.next = prev
  }

  /** Discrete energy ½Σ[(u−u_prev)²/Δt² + c²|∇u|²] in grid units (Δt = C). */
  energy(): number {
    const { n, u, prev } = this
    const C = this.courant
    let kin = 0
    let pot = 0
    for (let j = 0; j < n - 1; j++) {
      for (let i = 0; i < n - 1; i++) {
        const k = j * n + i
        const v = u[k] - prev[k]
        kin += v * v
        const gx = u[k + 1] - u[k]
        const gy = u[k + n] - u[k]
        pot += gx * gx + gy * gy
      }
    }
    return 0.5 * (kin / (C * C) + pot)
  }

  /** Gaussian pulse initial condition (at rest). */
  pulse(ci: number, cj: number, radius: number, amp: number): void {
    const { n } = this
    for (let j = 1; j < n - 1; j++) {
      for (let i = 1; i < n - 1; i++) {
        const d2 = (i - ci) ** 2 + (j - cj) ** 2
        const v = amp * Math.exp(-d2 / (radius * radius))
        this.u[j * n + i] += v
        this.prev[j * n + i] += v
      }
    }
  }
}

// ─── Gray–Scott ────────────────────────────────────────────────────────────

export interface GrayScottPreset {
  id: string
  name: string
  F: number
  k: number
}

/** Classic parameter regimes (Pearson 1993; Munafo's map). */
export const GRAY_SCOTT_PRESETS: GrayScottPreset[] = [
  { id: 'coral', name: 'Coral growth', F: 0.0545, k: 0.062 },
  { id: 'mitosis', name: 'Mitosis (dividing spots)', F: 0.0367, k: 0.0649 },
  { id: 'spots', name: 'Solitons / spots', F: 0.03, k: 0.062 },
  { id: 'worms', name: 'Worms / labyrinth', F: 0.046, k: 0.063 },
  { id: 'waves', name: 'Pulsating waves', F: 0.014, k: 0.045 },
]

/**
 * u_t = D_u∇²u − uv² + F(1 − u),  v_t = D_v∇²v + uv² − (F + k)v.
 * 9-point isotropic Laplacian (weights −1, 0.2, 0.05), Δt = 1, Δx = 1, D_u = 1, D_v = 0.5.
 */
export class GrayScott {
  readonly n: number
  u: Float32Array
  v: Float32Array
  private u2: Float32Array
  private v2: Float32Array
  F = 0.0545
  k = 0.062
  Du = 1.0
  Dv = 0.5
  dt = 1.0
  steps = 0

  constructor(n: number) {
    this.n = n
    this.u = new Float32Array(n * n).fill(1)
    this.v = new Float32Array(n * n)
    this.u2 = new Float32Array(n * n)
    this.v2 = new Float32Array(n * n)
  }

  reset(seed = 1): void {
    this.u.fill(1)
    this.v.fill(0)
    this.steps = 0
    const rng = new Rng(seed)
    const n = this.n
    for (let s = 0; s < 14; s++) {
      const ci = rng.int(n)
      const cj = rng.int(n)
      const r = 2 + rng.int(5)
      for (let dj = -r; dj <= r; dj++) {
        for (let di = -r; di <= r; di++) {
          const k = ((cj + dj + n) % n) * n + ((ci + di + n) % n)
          this.u[k] = 0.5
          this.v[k] = 0.25 + 0.05 * rng.next()
        }
      }
    }
  }

  paint(ci: number, cj: number, r: number): void {
    const n = this.n
    for (let dj = -r; dj <= r; dj++) {
      for (let di = -r; di <= r; di++) {
        if (di * di + dj * dj > r * r) continue
        const k = ((Math.round(cj) + dj + n) % n) * n + ((Math.round(ci) + di + n) % n)
        this.u[k] = 0.5
        this.v[k] = 0.25
      }
    }
  }

  step(iters = 1): void {
    const n = this.n
    const { F, k, Du, Dv, dt } = this
    for (let it = 0; it < iters; it++) {
      const u = this.u, v = this.v, un = this.u2, vn = this.v2
      for (let j = 0; j < n; j++) {
        const jm = ((j - 1 + n) % n) * n
        const j0 = j * n
        const jp = ((j + 1) % n) * n
        for (let i = 0; i < n; i++) {
          const im = (i - 1 + n) % n
          const ip = (i + 1) % n
          const c = j0 + i
          const lu = 0.2 * (u[j0 + im] + u[j0 + ip] + u[jm + i] + u[jp + i]) + 0.05 * (u[jm + im] + u[jm + ip] + u[jp + im] + u[jp + ip]) - u[c]
          const lv = 0.2 * (v[j0 + im] + v[j0 + ip] + v[jm + i] + v[jp + i]) + 0.05 * (v[jm + im] + v[jm + ip] + v[jp + im] + v[jp + ip]) - v[c]
          const uvv = u[c] * v[c] * v[c]
          un[c] = u[c] + dt * (Du * lu - uvv + F * (1 - u[c]))
          vn[c] = v[c] + dt * (Dv * lv + uvv - (F + k) * v[c])
        }
      }
      this.u = un
      this.v = vn
      this.u2 = u
      this.v2 = v
      this.steps++
    }
  }
}
