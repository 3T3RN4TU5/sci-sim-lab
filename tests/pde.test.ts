import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import { GrayScott, Heat2D, heatModeAmplitude, Wave2D } from '../src/numerics/pde.ts'

function modeAmplitude(h: Heat2D): number {
  // Project onto sin(πx) sin(πy): a = 4 ∫∫ u sin sin.
  const n = h.n
  let s = 0
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) s += h.u[j * n + i] * Math.sin((Math.PI * i) / (n - 1)) * Math.sin((Math.PI * j) / (n - 1))
  return 4 * s * h.dx * h.dx
}

describe('2D heat equation', () => {
  it('FTCS reproduces the analytic decay of the (1,1) Fourier mode', (t) => {
    const h = new Heat2D(65, 1)
    h.setMode(1, 1)
    const dt = 0.9 * h.maxStableDt()
    const T = 0.05
    const steps = Math.round(T / dt)
    for (let s = 0; s < steps; s++) h.step(dt, 'ftcs')
    const exact = heatModeAmplitude(1, h.t)
    const rel = Math.abs(modeAmplitude(h) - exact) / exact
    t.diagnostic(`t=${h.t.toFixed(4)} amplitude ${modeAmplitude(h).toFixed(6)} vs exact ${exact.toFixed(6)} (rel ${rel.toExponential(2)})`)
    assert.ok(rel < 2e-3)
  })

  it('ADI (Peaceman–Rachford) matches the decay with steps 40× above the FTCS limit', (t) => {
    const h = new Heat2D(65, 1)
    h.setMode(1, 1)
    const dt = 40 * h.maxStableDt()
    const steps = Math.round(0.05 / dt)
    for (let s = 0; s < steps; s++) h.step(dt, 'adi')
    const exact = heatModeAmplitude(1, h.t)
    const rel = Math.abs(modeAmplitude(h) - exact) / exact
    t.diagnostic(`r = ${h.ratio(dt).toFixed(1)}, rel error ${rel.toExponential(2)}`)
    assert.ok(rel < 1e-2)
  })

  it('ADI is second-order accurate in time', (t) => {
    const errs: number[] = []
    const dts = [0.004, 0.002, 0.001]
    for (const dt of dts) {
      const h = new Heat2D(129, 1)
      h.setMode(2, 1)
      const steps = Math.round(0.02 / dt)
      for (let s = 0; s < steps; s++) h.step(dt, 'adi')
      // Compare with the semi-discrete exact solution (spatial eigenvalue of the 5-point Laplacian).
      const n = h.n
      const lam = (4 / h.dx ** 2) * (Math.sin((2 * Math.PI * h.dx) / 2) ** 2 + Math.sin((Math.PI * h.dx) / 2) ** 2)
      const ex = Math.exp(-lam * h.t)
      const k = Math.round((n - 1) / 4) * n + Math.round((n - 1) / 4) // a grid point with sin(2πx)sin(πy) ≠ 0
      const shape = Math.sin((2 * Math.PI * (k % n)) / (n - 1)) * Math.sin((Math.PI * Math.floor(k / n)) / (n - 1))
      errs.push(Math.abs(h.u[k] / shape - ex))
    }
    const order = Math.log2(errs[0] / errs[1])
    const order2 = Math.log2(errs[1] / errs[2])
    t.diagnostic(`errors ${errs.map((e) => e.toExponential(2)).join(', ')} → order ${order.toFixed(2)}, ${order2.toFixed(2)}`)
    assert.ok(order > 1.8 && order2 > 1.8)
  })

  it('FTCS is unstable above the CFL limit r = 1/4', () => {
    const h = new Heat2D(33, 1)
    h.setMode(1, 1)
    // Seed round-off-size noise in the highest mode, which grows when r > 1/4.
    for (let k = 0; k < h.u.length; k++) h.u[k] += 1e-12 * (k % 2 ? 1 : -1)
    const dt = 1.2 * h.maxStableDt()
    for (let s = 0; s < 400; s++) h.step(dt, 'ftcs')
    assert.ok(!(h.maxAbs() < 10), 'should blow up')
  })
})

describe('2D wave equation', () => {
  it('standing (1,1) mode oscillates at ω = cπ√2', (t) => {
    const n = 97
    const w = new Wave2D(n)
    w.boundary = 'reflect'
    w.courant = 0.5
    const L = n - 1 // grid units, c = 1
    const omega = (Math.PI * Math.SQRT2) / L
    const dt = w.courant
    for (let j = 0; j < n; j++)
      for (let i = 0; i < n; i++) {
        const s = Math.sin((Math.PI * i) / L) * Math.sin((Math.PI * j) / L)
        w.u[j * n + i] = s
        w.prev[j * n + i] = s * Math.cos(omega * dt)
      }
    const steps = 800
    for (let s = 0; s < steps; s++) w.step()
    const centre = w.u[((n - 1) / 2) * n + (n - 1) / 2]
    const expected = Math.cos(omega * dt * steps)
    t.diagnostic(`u(centre) = ${centre.toFixed(5)}, exact ${expected.toFixed(5)}`)
    assert.ok(Math.abs(centre - expected) < 5e-3)
  })

  it('reflecting box conserves energy; absorbing (Mur) boundary removes most of it', (t) => {
    const make = (b: 'reflect' | 'absorb') => {
      const w = new Wave2D(129)
      w.boundary = b
      w.courant = 0.5
      w.pulse(64, 64, 4, 1)
      return w
    }
    const r = make('reflect')
    const a = make('absorb')
    for (let s = 0; s < 10; s++) {
      r.step()
      a.step()
    }
    const e0r = r.energy()
    const e0a = a.energy()
    for (let s = 0; s < 800; s++) {
      r.step()
      a.step()
    }
    const keptR = r.energy() / e0r
    const keptA = a.energy() / e0a
    t.diagnostic(`energy kept: reflecting ${(100 * keptR).toFixed(2)}%, absorbing ${(100 * keptA).toFixed(2)}%`)
    assert.ok(Math.abs(keptR - 1) < 0.02)
    assert.ok(keptA < 0.05)
  })
})

describe('Gray–Scott reaction–diffusion', () => {
  it('the homogeneous state (u, v) = (1, 0) is a fixed point', () => {
    const g = new GrayScott(32)
    g.step(50)
    assert.ok(g.u.every((v) => Math.abs(v - 1) < 1e-6) && g.v.every((v) => v === 0))
  })

  it('seeded patterns stay bounded in [0, 1] and evolve', () => {
    const g = new GrayScott(96)
    g.reset(3)
    const before = g.v.slice()
    g.step(2000)
    let changed = 0
    for (let k = 0; k < g.v.length; k++) {
      assert.ok(g.u[k] >= -1e-6 && g.u[k] <= 1 + 1e-6 && g.v[k] >= -1e-6 && g.v[k] <= 1 + 1e-6)
      if (Math.abs(g.v[k] - before[k]) > 0.05) changed++
    }
    assert.ok(changed > 100)
  })
})
