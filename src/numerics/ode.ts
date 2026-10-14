/**
 * Ordinary differential equation solvers, written from scratch.
 *
 * Fixed-step:  explicit Euler (order 1), explicit midpoint / RK2 (2), classic RK4 (4),
 *              velocity Verlet / leapfrog (2, symplectic), Yoshida 4th order (4, symplectic),
 *              backward (implicit) Euler with Newton iterations (1, L-stable).
 * Adaptive:    Dormand–Prince RK5(4) with embedded error estimate, PI-free classic step-size
 *              control, FSAL and optional dense output (Hairer's continuous extension),
 *              and an adaptive backward Euler driven by step doubling for stiff problems.
 *
 * All state lives in Float64Arrays and every stepper preallocates its work vectors,
 * so the inner loops do not allocate.
 */
import { luFactor, luSolve } from './linalg.ts'

/** dy/dt = f(t, y), written into `out`. */
export type Rhs = (t: number, y: Float64Array, out: Float64Array) => void
/** Acceleration a(q) for separable Hamiltonians with unit mass (q' = v, v' = a(q)). */
export type Accel = (q: Float64Array, out: Float64Array) => void
/** Jacobian ∂f/∂y as a row-major dim×dim matrix. */
export type Jacobian = (t: number, y: Float64Array, out: Float64Array) => void

export interface OdeProblem {
  /** Length of the state vector. For problems with `accel`, y = [q₁…q_d, v₁…v_d] and dim = 2d. */
  dim: number
  rhs: Rhs
  accel?: Accel
  jac?: Jacobian
}

export type MethodId = 'euler' | 'midpoint' | 'rk4' | 'verlet' | 'yoshida4' | 'backward-euler'

export interface MethodInfo {
  id: MethodId
  name: string
  short: string
  order: number
  symplectic: boolean
  implicit: boolean
  /** Right-hand-side (or force) evaluations per step for explicit methods. */
  evalsPerStep: number
}

export const METHODS: Record<MethodId, MethodInfo> = {
  euler: { id: 'euler', name: 'Explicit Euler', short: 'Euler', order: 1, symplectic: false, implicit: false, evalsPerStep: 1 },
  midpoint: { id: 'midpoint', name: 'Midpoint (RK2)', short: 'RK2', order: 2, symplectic: false, implicit: false, evalsPerStep: 2 },
  rk4: { id: 'rk4', name: 'Classic RK4', short: 'RK4', order: 4, symplectic: false, implicit: false, evalsPerStep: 4 },
  verlet: { id: 'verlet', name: 'Velocity Verlet', short: 'Verlet', order: 2, symplectic: true, implicit: false, evalsPerStep: 1 },
  yoshida4: { id: 'yoshida4', name: 'Yoshida 4 (symplectic)', short: 'Yoshida4', order: 4, symplectic: true, implicit: false, evalsPerStep: 3 },
  'backward-euler': { id: 'backward-euler', name: 'Backward Euler (Newton)', short: 'BE', order: 1, symplectic: false, implicit: true, evalsPerStep: NaN },
}

/** Advances y (in place) from t to t + h. */
export type Stepper = (t: number, y: Float64Array, h: number) => void

export function makeStepper(id: MethodId, p: OdeProblem): Stepper {
  switch (id) {
    case 'euler':
      return eulerStepper(p)
    case 'midpoint':
      return midpointStepper(p)
    case 'rk4':
      return rk4Stepper(p)
    case 'verlet':
      return verletStepper(p)
    case 'yoshida4':
      return yoshidaStepper(p)
    case 'backward-euler': {
      const be = new BackwardEuler(p)
      const out = new Float64Array(p.dim)
      return (t, y, h) => {
        if (be.step(t, y, h, out)) y.set(out)
        else y.fill(NaN)
      }
    }
  }
}

function eulerStepper(p: OdeProblem): Stepper {
  const k = new Float64Array(p.dim)
  return (t, y, h) => {
    p.rhs(t, y, k)
    for (let i = 0; i < y.length; i++) y[i] += h * k[i]
  }
}

function midpointStepper(p: OdeProblem): Stepper {
  const k1 = new Float64Array(p.dim)
  const k2 = new Float64Array(p.dim)
  const tmp = new Float64Array(p.dim)
  return (t, y, h) => {
    const n = y.length
    p.rhs(t, y, k1)
    for (let i = 0; i < n; i++) tmp[i] = y[i] + 0.5 * h * k1[i]
    p.rhs(t + 0.5 * h, tmp, k2)
    for (let i = 0; i < n; i++) y[i] += h * k2[i]
  }
}

function rk4Stepper(p: OdeProblem): Stepper {
  const n = p.dim
  const k1 = new Float64Array(n)
  const k2 = new Float64Array(n)
  const k3 = new Float64Array(n)
  const k4 = new Float64Array(n)
  const tmp = new Float64Array(n)
  return (t, y, h) => {
    p.rhs(t, y, k1)
    for (let i = 0; i < n; i++) tmp[i] = y[i] + 0.5 * h * k1[i]
    p.rhs(t + 0.5 * h, tmp, k2)
    for (let i = 0; i < n; i++) tmp[i] = y[i] + 0.5 * h * k2[i]
    p.rhs(t + 0.5 * h, tmp, k3)
    for (let i = 0; i < n; i++) tmp[i] = y[i] + h * k3[i]
    p.rhs(t + h, tmp, k4)
    const h6 = h / 6
    for (let i = 0; i < n; i++) y[i] += h6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])
  }
}

function requireAccel(p: OdeProblem): Accel {
  if (!p.accel) throw new Error('Symplectic integrators need a separable problem with an `accel` function')
  return p.accel
}

/** Kick–drift–kick velocity Verlet (equivalent to leapfrog). */
function verletStepper(p: OdeProblem): Stepper {
  const accel = requireAccel(p)
  const d = p.dim / 2
  const q = new Float64Array(d)
  const a = new Float64Array(d)
  return (_t, y, h) => {
    for (let i = 0; i < d; i++) q[i] = y[i]
    accel(q, a)
    for (let i = 0; i < d; i++) {
      y[d + i] += 0.5 * h * a[i]
      y[i] += h * y[d + i]
      q[i] = y[i]
    }
    accel(q, a)
    for (let i = 0; i < d; i++) y[d + i] += 0.5 * h * a[i]
  }
}

// Yoshida (1990) triple-jump composition of leapfrog.
const CBRT2 = Math.cbrt(2)
const W1 = 1 / (2 - CBRT2)
const W0 = -CBRT2 / (2 - CBRT2)
const YC = [W1 / 2, (W0 + W1) / 2, (W0 + W1) / 2, W1 / 2]
const YD = [W1, W0, W1]

function yoshidaStepper(p: OdeProblem): Stepper {
  const accel = requireAccel(p)
  const d = p.dim / 2
  const q = new Float64Array(d)
  const a = new Float64Array(d)
  return (_t, y, h) => {
    for (let s = 0; s < 4; s++) {
      const c = YC[s] * h
      for (let i = 0; i < d; i++) y[i] += c * y[d + i]
      if (s < 3) {
        for (let i = 0; i < d; i++) q[i] = y[i]
        accel(q, a)
        const dd = YD[s] * h
        for (let i = 0; i < d; i++) y[d + i] += dd * a[i]
      }
    }
  }
}

/**
 * Backward Euler: solve G(z) = z − y − h·f(t+h, z) = 0 with full Newton iterations,
 * J_G = I − h·∂f/∂y. Uses the analytic Jacobian when the problem provides one and
 * forward differences otherwise.
 */
export class BackwardEuler {
  readonly p: OdeProblem
  fevals = 0
  jevals = 0
  newtonIters = 0
  private readonly n: number
  private readonly J: Float64Array
  private readonly M: Float64Array
  private readonly piv: Int32Array
  private readonly f: Float64Array
  private readonly g: Float64Array
  private readonly z: Float64Array
  private readonly zp: Float64Array
  private readonly fp: Float64Array
  maxIter = 12
  tol = 1e-10

  constructor(p: OdeProblem) {
    this.p = p
    const n = (this.n = p.dim)
    this.J = new Float64Array(n * n)
    this.M = new Float64Array(n * n)
    this.piv = new Int32Array(n)
    this.f = new Float64Array(n)
    this.g = new Float64Array(n)
    this.z = new Float64Array(n)
    this.zp = new Float64Array(n)
    this.fp = new Float64Array(n)
  }

  jacobian(t: number, y: Float64Array): void {
    this.jevals++
    if (this.p.jac) {
      this.p.jac(t, y, this.J)
      return
    }
    const n = this.n
    this.p.rhs(t, y, this.f)
    this.fevals++
    for (let j = 0; j < n; j++) {
      const eps = 1e-8 * Math.max(1, Math.abs(y[j]))
      this.zp.set(y)
      this.zp[j] += eps
      this.p.rhs(t, this.zp, this.fp)
      this.fevals++
      for (let i = 0; i < n; i++) this.J[i * n + j] = (this.fp[i] - this.f[i]) / eps
    }
  }

  /** One step from (t, y) with size h into `out`. Returns false if Newton fails to converge. */
  step(t: number, y: Float64Array, h: number, out: Float64Array): boolean {
    const n = this.n
    const { z, f, g, M, J, piv } = this
    z.set(y)
    const t1 = t + h
    for (let it = 0; it < this.maxIter; it++) {
      this.newtonIters++
      this.p.rhs(t1, z, f)
      this.fevals++
      for (let i = 0; i < n; i++) g[i] = -(z[i] - y[i] - h * f[i])
      this.jacobian(t1, z)
      for (let i = 0; i < n * n; i++) M[i] = -h * J[i]
      for (let i = 0; i < n; i++) M[i * n + i] += 1
      if (!luFactor(M, n, piv)) return false
      luSolve(M, n, piv, g)
      let dn = 0
      let zn = 0
      for (let i = 0; i < n; i++) {
        z[i] += g[i]
        dn = Math.max(dn, Math.abs(g[i]))
        zn = Math.max(zn, Math.abs(z[i]))
      }
      if (!Number.isFinite(dn)) return false
      if (dn <= this.tol * (1 + zn)) {
        out.set(z)
        return true
      }
    }
    return false
  }
}

// ─── Adaptive solvers ───────────────────────────────────────────────────────

export interface AdaptiveOptions {
  rtol?: number
  atol?: number
  /** Initial step; chosen automatically (Hairer's heuristic) when omitted. */
  h0?: number
  hMax?: number
  maxSteps?: number
  /** Store every accepted step (default true). */
  record?: boolean
  /** Sorted output times evaluated with the dense-output interpolant. */
  denseAt?: ArrayLike<number>
}

export type AdaptiveStatus = 'ok' | 'max-steps' | 'step-too-small' | 'newton-failure'

export interface AdaptiveResult {
  t: number[]
  y: Float64Array[]
  /** Final state (always set, even when `record` is false). */
  yEnd: Float64Array
  tEnd: number
  accepted: number
  rejected: number
  fevals: number
  status: AdaptiveStatus
  /** Values at `denseAt`, when requested. */
  dense: Float64Array[]
}

function errNorm(err: Float64Array, y0: Float64Array, y1: Float64Array, rtol: number, atol: number): number {
  let s = 0
  const n = err.length
  for (let i = 0; i < n; i++) {
    const sc = atol + rtol * Math.max(Math.abs(y0[i]), Math.abs(y1[i]))
    const e = err[i] / sc
    s += e * e
  }
  return Math.sqrt(s / n)
}

// Dormand–Prince 5(4) tableau.
const A21 = 1 / 5
const A31 = 3 / 40, A32 = 9 / 40
const A41 = 44 / 45, A42 = -56 / 15, A43 = 32 / 9
const A51 = 19372 / 6561, A52 = -25360 / 2187, A53 = 64448 / 6561, A54 = -212 / 729
const A61 = 9017 / 3168, A62 = -355 / 33, A63 = 46732 / 5247, A64 = 49 / 176, A65 = -5103 / 18656
const B1 = 35 / 384, B3 = 500 / 1113, B4 = 125 / 192, B5 = -2187 / 6784, B6 = 11 / 84
// e = b − b̂ (difference between the 5th- and embedded 4th-order weights)
const E1 = 71 / 57600, E3 = -71 / 16695, E4 = 71 / 1920, E5 = -17253 / 339200, E6 = 22 / 525, E7 = -1 / 40
// Dense output coefficients (Hairer, Nørsett & Wanner, contd5)
const D1 = -12715105075 / 11282082432, D3 = 87487479700 / 32700410799, D4 = -10690763975 / 1880347072
const D5 = 701980252875 / 199316789632, D6 = -1453857185 / 822651844, D7 = 69997945 / 29380423

/**
 * Dormand–Prince RK5(4)7M with FSAL. Local extrapolation (advances with the 5th-order
 * solution); step size h_new = h·clamp(0.9·err^(−1/5), 0.2, 5).
 */
export function dopri45(p: OdeProblem, t0: number, y0: ArrayLike<number>, t1: number, opts: AdaptiveOptions = {}): AdaptiveResult {
  const n = p.dim
  const rtol = opts.rtol ?? 1e-6
  const atol = opts.atol ?? 1e-9
  const maxSteps = opts.maxSteps ?? 1_000_000
  const record = opts.record ?? true
  const dir = Math.sign(t1 - t0) || 1
  const hMax = opts.hMax ?? Math.abs(t1 - t0)
  const y = Float64Array.from(y0)
  const ynew = new Float64Array(n)
  const k1 = new Float64Array(n), k2 = new Float64Array(n), k3 = new Float64Array(n), k4 = new Float64Array(n)
  const k5 = new Float64Array(n), k6 = new Float64Array(n), k7 = new Float64Array(n)
  const tmp = new Float64Array(n)
  const err = new Float64Array(n)
  const res: AdaptiveResult = { t: [], y: [], yEnd: y, tEnd: t0, accepted: 0, rejected: 0, fevals: 0, status: 'ok', dense: [] }
  const denseAt = opts.denseAt
  let di = 0
  if (record) {
    res.t.push(t0)
    res.y.push(Float64Array.from(y))
  }
  let t = t0
  p.rhs(t, y, k1)
  res.fevals++

  let h: number
  if (opts.h0) h = Math.abs(opts.h0)
  else {
    // Hairer's starting step heuristic.
    let d0 = 0, d1 = 0
    for (let i = 0; i < n; i++) {
      const sc = atol + rtol * Math.abs(y[i])
      d0 += (y[i] / sc) ** 2
      d1 += (k1[i] / sc) ** 2
    }
    d0 = Math.sqrt(d0 / n)
    d1 = Math.sqrt(d1 / n)
    let hh = d0 < 1e-5 || d1 < 1e-5 ? 1e-6 : (0.01 * d0) / d1
    for (let i = 0; i < n; i++) tmp[i] = y[i] + dir * hh * k1[i]
    p.rhs(t + dir * hh, tmp, k2)
    res.fevals++
    let d2 = 0
    for (let i = 0; i < n; i++) {
      const sc = atol + rtol * Math.abs(y[i])
      d2 += ((k2[i] - k1[i]) / sc) ** 2
    }
    d2 = Math.sqrt(d2 / n) / hh
    const dm = Math.max(d1, d2)
    const h1 = dm <= 1e-15 ? Math.max(1e-6, hh * 1e-3) : Math.pow(0.01 / dm, 1 / 5)
    hh = Math.min(100 * hh, h1)
    h = hh
  }
  h = Math.min(h, hMax)

  let steps = 0
  while (dir * (t1 - t) > 0) {
    if (steps++ >= maxSteps) {
      res.status = 'max-steps'
      break
    }
    const clamped = dir * (t + dir * h - t1) >= 0
    if (clamped) h = Math.abs(t1 - t)
    const hs = dir * h
    for (let i = 0; i < n; i++) tmp[i] = y[i] + hs * A21 * k1[i]
    p.rhs(t + hs / 5, tmp, k2)
    for (let i = 0; i < n; i++) tmp[i] = y[i] + hs * (A31 * k1[i] + A32 * k2[i])
    p.rhs(t + (3 * hs) / 10, tmp, k3)
    for (let i = 0; i < n; i++) tmp[i] = y[i] + hs * (A41 * k1[i] + A42 * k2[i] + A43 * k3[i])
    p.rhs(t + (4 * hs) / 5, tmp, k4)
    for (let i = 0; i < n; i++) tmp[i] = y[i] + hs * (A51 * k1[i] + A52 * k2[i] + A53 * k3[i] + A54 * k4[i])
    p.rhs(t + (8 * hs) / 9, tmp, k5)
    for (let i = 0; i < n; i++) tmp[i] = y[i] + hs * (A61 * k1[i] + A62 * k2[i] + A63 * k3[i] + A64 * k4[i] + A65 * k5[i])
    p.rhs(t + hs, tmp, k6)
    for (let i = 0; i < n; i++) ynew[i] = y[i] + hs * (B1 * k1[i] + B3 * k3[i] + B4 * k4[i] + B5 * k5[i] + B6 * k6[i])
    p.rhs(t + hs, ynew, k7)
    res.fevals += 6
    for (let i = 0; i < n; i++) err[i] = hs * (E1 * k1[i] + E3 * k3[i] + E4 * k4[i] + E5 * k5[i] + E6 * k6[i] + E7 * k7[i])
    const e = errNorm(err, y, ynew, rtol, atol)

    if (e <= 1 && Number.isFinite(e)) {
      const tNew = clamped ? t1 : t + hs
      if (denseAt) {
        while (di < denseAt.length && dir * (denseAt[di] - tNew) <= 0) {
          const th = (denseAt[di] - t) / hs
          const th1 = 1 - th
          const out = new Float64Array(n)
          for (let i = 0; i < n; i++) {
            const r2 = ynew[i] - y[i]
            const r3 = hs * k1[i] - r2
            const r4 = r2 - hs * k7[i] - r3
            const r5 = hs * (D1 * k1[i] + D3 * k3[i] + D4 * k4[i] + D5 * k5[i] + D6 * k6[i] + D7 * k7[i])
            out[i] = y[i] + th * (r2 + th1 * (r3 + th * (r4 + th1 * r5)))
          }
          res.dense.push(out)
          di++
        }
      }
      t = tNew
      y.set(ynew)
      k1.set(k7) // FSAL
      res.accepted++
      if (record) {
        res.t.push(t)
        res.y.push(Float64Array.from(y))
      }
      const fac = e === 0 ? 5 : Math.min(5, Math.max(0.2, 0.9 * Math.pow(e, -0.2)))
      h = Math.min(h * fac, hMax)
    } else {
      res.rejected++
      const fac = Number.isFinite(e) ? Math.max(0.2, 0.9 * Math.pow(e, -0.2)) : 0.1
      h *= fac
    }
    if (h < 1e-14 * Math.max(1, Math.abs(t))) {
      res.status = 'step-too-small'
      break
    }
  }
  res.tEnd = t
  res.yEnd = y
  return res
}

/**
 * Adaptive backward Euler for stiff problems. The local error is estimated by step
 * doubling (one step of h vs two steps of h/2): for a first-order method the difference
 * is ≈ the local error of the half-step solution. Newton failure halves… quarters the step.
 */
export function backwardEulerAdaptive(p: OdeProblem, t0: number, y0: ArrayLike<number>, t1: number, opts: AdaptiveOptions = {}): AdaptiveResult & { jevals: number } {
  const n = p.dim
  const rtol = opts.rtol ?? 1e-4
  const atol = opts.atol ?? 1e-7
  const maxSteps = opts.maxSteps ?? 200_000
  const record = opts.record ?? true
  const be = new BackwardEuler(p)
  const y = Float64Array.from(y0)
  const yFull = new Float64Array(n)
  const yHalf = new Float64Array(n)
  const yTwo = new Float64Array(n)
  const err = new Float64Array(n)
  const res: AdaptiveResult & { jevals: number } = { t: [], y: [], yEnd: y, tEnd: t0, accepted: 0, rejected: 0, fevals: 0, status: 'ok', dense: [], jevals: 0 }
  if (record) {
    res.t.push(t0)
    res.y.push(Float64Array.from(y))
  }
  let t = t0
  let h = opts.h0 ?? Math.abs(t1 - t0) * 1e-6
  const hMax = opts.hMax ?? Math.abs(t1 - t0) / 10
  let steps = 0
  while (t < t1) {
    if (steps++ >= maxSteps) {
      res.status = 'max-steps'
      break
    }
    const clamped = t + h >= t1
    if (clamped) h = t1 - t
    const ok = be.step(t, y, h, yFull) && be.step(t, y, h / 2, yHalf) && be.step(t + h / 2, yHalf, h / 2, yTwo)
    if (!ok) {
      res.rejected++
      h *= 0.25
      if (h < 1e-14 * Math.max(1, Math.abs(t))) {
        res.status = 'newton-failure'
        break
      }
      continue
    }
    for (let i = 0; i < n; i++) err[i] = yTwo[i] - yFull[i]
    const e = errNorm(err, y, yTwo, rtol, atol)
    if (e <= 1) {
      t = clamped ? t1 : t + h
      y.set(yTwo)
      res.accepted++
      if (record) {
        res.t.push(t)
        res.y.push(Float64Array.from(y))
      }
      h = Math.min(hMax, h * (e === 0 ? 4 : Math.min(4, Math.max(0.2, 0.9 / Math.sqrt(e)))))
    } else {
      res.rejected++
      h *= Math.max(0.2, 0.9 / Math.sqrt(e))
    }
  }
  res.tEnd = t
  res.fevals = be.fevals
  res.jevals = be.jevals
  return res
}

/** Integrate with a fixed-step method; calls `observe` every `every` steps (and at step 0). */
export function integrateFixed(
  method: MethodId,
  p: OdeProblem,
  y0: ArrayLike<number>,
  t0: number,
  h: number,
  steps: number,
  observe?: (t: number, y: Float64Array, step: number) => void,
  every = 1,
): Float64Array {
  const step = makeStepper(method, p)
  const y = Float64Array.from(y0)
  let t = t0
  observe?.(t, y, 0)
  for (let s = 1; s <= steps; s++) {
    step(t, y, h)
    t = t0 + s * h
    if (observe && (s % every === 0 || s === steps)) observe(t, y, s)
  }
  return y
}
