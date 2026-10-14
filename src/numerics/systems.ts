/**
 * Model problems used across the lab, each with its right-hand side, an
 * acceleration (for separable Hamiltonians, enabling symplectic integrators),
 * an analytic Jacobian where useful, conserved quantities and, when known,
 * the exact solution.
 */
import type { OdeProblem } from './ode.ts'

export interface ModelProblem extends OdeProblem {
  id: string
  name: string
  y0: Float64Array
  /** Natural period or time scale, used to pick sensible horizons. */
  period: number
  energy?: (y: Float64Array) => number
  exact?: (t: number) => Float64Array
}

/** Harmonic oscillator q'' = −ω²q, y = [q, v]. Exact: q = cos ωt (for q₀=1, v₀=0). */
export function harmonic(omega = 1): ModelProblem {
  const w2 = omega * omega
  return {
    id: 'harmonic',
    name: 'Harmonic oscillator',
    dim: 2,
    y0: Float64Array.of(1, 0),
    period: (2 * Math.PI) / omega,
    rhs: (_t, y, out) => {
      out[0] = y[1]
      out[1] = -w2 * y[0]
    },
    accel: (q, a) => {
      a[0] = -w2 * q[0]
    },
    jac: (_t, _y, J) => {
      J[0] = 0
      J[1] = 1
      J[2] = -w2
      J[3] = 0
    },
    energy: (y) => 0.5 * y[1] * y[1] + 0.5 * w2 * y[0] * y[0],
    exact: (t) => Float64Array.of(Math.cos(omega * t), -omega * Math.sin(omega * t)),
  }
}

/** Nonlinear pendulum θ'' = −(g/L) sin θ, y = [θ, ω]. */
export function pendulum(theta0 = 2.0, gOverL = 1): ModelProblem {
  // Exact period: T = 4K(sin(θ₀/2)) / √(g/L), with K via the AGM.
  const k = Math.sin(theta0 / 2)
  const period = (4 * ellipticK(k)) / Math.sqrt(gOverL)
  return {
    id: 'pendulum',
    name: 'Nonlinear pendulum',
    dim: 2,
    y0: Float64Array.of(theta0, 0),
    period,
    rhs: (_t, y, out) => {
      out[0] = y[1]
      out[1] = -gOverL * Math.sin(y[0])
    },
    accel: (q, a) => {
      a[0] = -gOverL * Math.sin(q[0])
    },
    jac: (_t, y, J) => {
      J[0] = 0
      J[1] = 1
      J[2] = -gOverL * Math.cos(y[0])
      J[3] = 0
    },
    energy: (y) => 0.5 * y[1] * y[1] - gOverL * Math.cos(y[0]),
  }
}

/**
 * Kepler two-body problem in the reduced (GM = 1) form, y = [x, y, vx, vy].
 * Starts at perihelion of an orbit with semi-major axis 1 and eccentricity e,
 * so the period is exactly 2π and the state after whole periods is y₀.
 */
export function kepler(e = 0.5): ModelProblem {
  const y0 = Float64Array.of(1 - e, 0, 0, Math.sqrt((1 + e) / (1 - e)))
  return {
    id: 'kepler',
    name: 'Kepler orbit',
    dim: 4,
    y0,
    period: 2 * Math.PI,
    rhs: (_t, y, out) => {
      const r2 = y[0] * y[0] + y[1] * y[1]
      const r3 = r2 * Math.sqrt(r2)
      out[0] = y[2]
      out[1] = y[3]
      out[2] = -y[0] / r3
      out[3] = -y[1] / r3
    },
    accel: (q, a) => {
      const r2 = q[0] * q[0] + q[1] * q[1]
      const r3 = r2 * Math.sqrt(r2)
      a[0] = -q[0] / r3
      a[1] = -q[1] / r3
    },
    energy: (y) => 0.5 * (y[2] * y[2] + y[3] * y[3]) - 1 / Math.hypot(y[0], y[1]),
    exact: (t) => keplerExact(e, t),
  }
}

/** Exact Kepler position/velocity by solving Kepler's equation M = E − e sin E with Newton. */
export function keplerExact(e: number, t: number): Float64Array {
  const M = t // mean motion n = 1 for a = 1, GM = 1
  let E = e > 0.8 ? Math.PI : M
  for (let i = 0; i < 50; i++) {
    const dE = (E - e * Math.sin(E) - M) / (1 - e * Math.cos(E))
    E -= dE
    if (Math.abs(dE) < 1e-15) break
  }
  const c = Math.cos(E)
  const s = Math.sin(E)
  const b = Math.sqrt(1 - e * e)
  const Edot = 1 / (1 - e * c)
  return Float64Array.of(c - e, b * s, -s * Edot, b * c * Edot)
}

/** Van der Pol oscillator: y₁' = y₂, y₂' = μ(1 − y₁²)y₂ − y₁. Stiff for large μ (period ≈ (3 − 2 ln 2)μ). */
export function vanDerPol(mu: number): ModelProblem {
  return {
    id: 'vanderpol',
    name: 'Van der Pol',
    dim: 2,
    y0: Float64Array.of(2, 0),
    period: Math.max(2 * Math.PI, (3 - 2 * Math.log(2)) * mu),
    rhs: (_t, y, out) => {
      out[0] = y[1]
      out[1] = mu * (1 - y[0] * y[0]) * y[1] - y[0]
    },
    jac: (_t, y, J) => {
      J[0] = 0
      J[1] = 1
      J[2] = -2 * mu * y[0] * y[1] - 1
      J[3] = mu * (1 - y[0] * y[0])
    },
  }
}

/** Robertson's chemical kinetics — the canonical stiff test problem (rates 0.04, 3e7, 1e4). */
export function robertson(): ModelProblem {
  return {
    id: 'robertson',
    name: 'Robertson kinetics',
    dim: 3,
    y0: Float64Array.of(1, 0, 0),
    period: 40,
    rhs: (_t, y, out) => {
      const a = 0.04 * y[0]
      const b = 1e4 * y[1] * y[2]
      const c = 3e7 * y[1] * y[1]
      out[0] = -a + b
      out[1] = a - b - c
      out[2] = c
    },
    jac: (_t, y, J) => {
      J[0] = -0.04
      J[1] = 1e4 * y[2]
      J[2] = 1e4 * y[1]
      J[3] = 0.04
      J[4] = -1e4 * y[2] - 6e7 * y[1]
      J[5] = -1e4 * y[1]
      J[6] = 0
      J[7] = 6e7 * y[1]
      J[8] = 0
    },
  }
}

/** Lorenz (1963) system. */
export function lorenz(sigma = 10, rho = 28, beta = 8 / 3): ModelProblem {
  return {
    id: 'lorenz',
    name: 'Lorenz system',
    dim: 3,
    y0: Float64Array.of(1, 1, 1),
    period: 1,
    rhs: (_t, y, out) => {
      out[0] = sigma * (y[1] - y[0])
      out[1] = y[0] * (rho - y[2]) - y[1]
      out[2] = y[0] * y[1] - beta * y[2]
    },
    jac: (_t, y, J) => {
      J[0] = -sigma
      J[1] = sigma
      J[2] = 0
      J[3] = rho - y[2]
      J[4] = -1
      J[5] = -y[0]
      J[6] = y[1]
      J[7] = y[0]
      J[8] = -beta
    },
  }
}

/**
 * Planar double pendulum with point masses m₁, m₂ on massless rods L₁, L₂.
 * y = [θ₁, θ₂, ω₁, ω₂]; equations from the Lagrangian.
 */
export function doublePendulum(m1 = 1, m2 = 1, l1 = 1, l2 = 1, g = 9.81): ModelProblem & { positions: (y: Float64Array) => [number, number, number, number] } {
  return {
    id: 'double-pendulum',
    name: 'Double pendulum',
    dim: 4,
    y0: Float64Array.of(2.2, 2.6, 0, 0),
    period: 2 * Math.PI * Math.sqrt(l1 / g),
    rhs: (_t, y, out) => {
      const [t1, t2, w1, w2] = y
      const d = t1 - t2
      const sd = Math.sin(d)
      const cd = Math.cos(d)
      const den = 2 * m1 + m2 - m2 * Math.cos(2 * d)
      out[0] = w1
      out[1] = w2
      out[2] = (-g * (2 * m1 + m2) * Math.sin(t1) - m2 * g * Math.sin(t1 - 2 * t2) - 2 * sd * m2 * (w2 * w2 * l2 + w1 * w1 * l1 * cd)) / (l1 * den)
      out[3] = (2 * sd * (w1 * w1 * l1 * (m1 + m2) + g * (m1 + m2) * Math.cos(t1) + w2 * w2 * l2 * m2 * cd)) / (l2 * den)
    },
    energy: (y) => {
      const [t1, t2, w1, w2] = y
      const T = 0.5 * (m1 + m2) * l1 * l1 * w1 * w1 + 0.5 * m2 * l2 * l2 * w2 * w2 + m2 * l1 * l2 * w1 * w2 * Math.cos(t1 - t2)
      const V = -(m1 + m2) * g * l1 * Math.cos(t1) - m2 * g * l2 * Math.cos(t2)
      return T + V
    },
    positions: (y) => {
      const x1 = l1 * Math.sin(y[0])
      const y1 = -l1 * Math.cos(y[0])
      return [x1, y1, x1 + l2 * Math.sin(y[1]), y1 - l2 * Math.cos(y[1])]
    },
  }
}

/** Complete elliptic integral of the first kind K(k) via the arithmetic–geometric mean. */
export function ellipticK(k: number): number {
  let a = 1
  let b = Math.sqrt(1 - k * k)
  for (let i = 0; i < 40 && Math.abs(a - b) > 1e-16 * a; i++) {
    const an = 0.5 * (a + b)
    b = Math.sqrt(a * b)
    a = an
  }
  return Math.PI / (2 * a)
}
