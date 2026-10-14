import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import { logLogSlope } from '../src/numerics/stats.ts'
import { cloneBodies, directAccel, energy, figure8, momentum, NBodySim, QuadTree, randomCloud, galaxies } from '../src/numerics/nbody.ts'

function forceErrors(n: number, theta: number, quadrupole = true, eps = 0.01) {
  const b = randomCloud(n, 5)
  const d = cloneBodies(b)
  directAccel(d, 1, eps)
  const tree = new QuadTree()
  tree.build(b)
  tree.accel(b, theta, 1, eps, quadrupole)
  // Per-body error relative to the RMS force (per-body relative errors blow up where
  // the net force nearly cancels, e.g. at the centre of a uniform disk).
  let ms = 0
  for (let i = 0; i < n; i++) ms += d.ax[i] ** 2 + d.ay[i] ** 2
  const aRms = Math.sqrt(ms / n)
  const rel: number[] = []
  for (let i = 0; i < n; i++) rel.push(Math.hypot(b.ax[i] - d.ax[i], b.ay[i] - d.ay[i]) / aRms)
  rel.sort((p, q) => p - q)
  return { median: rel[n >> 1], p99: rel[Math.floor(0.99 * n)], rms: Math.sqrt(rel.reduce((s, r) => s + r * r, 0) / n), interactions: tree.interactions }
}

describe('Barnes–Hut quadtree', () => {
  it('θ = 0 reproduces direct summation to round-off', () => {
    const e = forceErrors(500, 0)
    assert.ok(e.p99 < 1e-12)
  })

  it('θ = 0.5 force error vs direct summation (uniform disk, N = 4000): < 2% monopole, < 0.5% quadrupole', (t) => {
    const mono = forceErrors(4000, 0.5, false)
    const quad = forceErrors(4000, 0.5, true)
    t.diagnostic(`monopole: median ${mono.median.toExponential(2)}, 99th pct ${mono.p99.toExponential(2)}, RMS ${mono.rms.toExponential(2)}`)
    t.diagnostic(`quadrupole: median ${quad.median.toExponential(2)}, 99th pct ${quad.p99.toExponential(2)}, RMS ${quad.rms.toExponential(2)}; ${quad.interactions} interactions vs ${4000 * 3999} direct`)
    assert.ok(mono.rms < 2e-2)
    assert.ok(quad.rms < 5e-3)
    assert.ok(quad.rms < mono.rms / 3)
    assert.ok(quad.interactions < 0.2 * 4000 * 3999)
  })

  it('error falls as a power of θ: ≈ θ² (monopole) and ≈ θ³ (quadrupole)', (t) => {
    const thetas = [0.2, 0.4, 0.8]
    const slope = (q: boolean) => logLogSlope(thetas, thetas.map((th) => forceErrors(2000, th, q).rms)).slope
    const m = slope(false)
    const q = slope(true)
    t.diagnostic(`d log err / d log θ: monopole ${m.toFixed(2)}, quadrupole ${q.toFixed(2)}`)
    assert.ok(m > 1.8 && m < 3)
    assert.ok(q > 2.8 && q > m + 0.6)
  })

  it('handles coincident bodies without infinite subdivision', () => {
    const b = randomCloud(50, 1)
    for (let i = 0; i < 10; i++) {
      b.x[i] = 0.123
      b.y[i] = -0.456
    }
    const tree = new QuadTree()
    tree.build(b)
    tree.accel(b, 0.5, 1, 0.05)
    assert.ok(b.ax.every(Number.isFinite))
  })
})

describe('leapfrog N-body integration', () => {
  it('figure-8 choreography returns to its initial state after one period (T ≈ 6.3259)', (t) => {
    const p = figure8()
    const sim = new NBodySim(cloneBodies(p.bodies), { G: 1, eps: 0, theta: 0, method: 'direct' })
    const T = 6.32591398
    const steps = 20000
    for (let s = 0; s < steps; s++) sim.step(T / steps)
    let err = 0
    for (let i = 0; i < 3; i++) err = Math.max(err, Math.hypot(sim.b.x[i] - p.bodies.x[i], sim.b.y[i] - p.bodies.y[i]))
    t.diagnostic(`position error after one period: ${err.toExponential(2)}`)
    assert.ok(err < 1e-4)
  })

  it('direct summation conserves linear momentum to round-off and energy to O(dt²)', (t) => {
    const p = galaxies(200)
    const sim = new NBodySim(cloneBodies(p.bodies), { ...p.params, theta: 0.5, method: 'direct' })
    const E0 = energy(sim.b, p.params.G, p.params.eps).total
    const L0 = momentum(sim.b).L
    for (let s = 0; s < 500; s++) sim.step(p.dt)
    const mom = momentum(sim.b)
    const dE = Math.abs((energy(sim.b, p.params.G, p.params.eps).total - E0) / E0)
    t.diagnostic(`|P| = ${Math.hypot(mom.px, mom.py).toExponential(2)}, |ΔE/E| = ${dE.toExponential(2)}, |ΔL/L| = ${Math.abs((mom.L - L0) / L0).toExponential(2)}`)
    assert.ok(Math.hypot(mom.px, mom.py) < 1e-13)
    assert.ok(Math.abs((mom.L - L0) / L0) < 1e-10)
    assert.ok(dE < 1e-3)
  })
})
