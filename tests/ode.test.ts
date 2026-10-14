import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import { convergenceStudy, energyDrift } from '../src/numerics/convergence.ts'
import { backwardEulerAdaptive, dopri45, integrateFixed, METHODS, type MethodId, type OdeProblem } from '../src/numerics/ode.ts'
import { logLogSlope } from '../src/numerics/stats.ts'
import { harmonic, kepler, pendulum, robertson, vanDerPol } from '../src/numerics/systems.ts'

const STEPS = [200, 400, 800, 1600, 3200]

describe('fixed-step integrators: measured convergence order', () => {
  const cases: [MethodId, number][] = [
    ['euler', 1],
    ['midpoint', 2],
    ['rk4', 4],
    ['verlet', 2],
    ['yoshida4', 4],
    ['backward-euler', 1],
  ]
  for (const [method, expected] of cases) {
    it(`${METHODS[method].name} is order ${expected} on the harmonic oscillator`, (t) => {
      const res = convergenceStudy(method, harmonic(), 10, STEPS)
      t.diagnostic(`order=${res.order.toFixed(3)} r2=${res.r2.toFixed(5)}`)
      assert.ok(Math.abs(res.order - expected) < 0.15, `observed order ${res.order}`)
    })
    it(`${METHODS[method].name} is order ${expected} on the Kepler problem (e = 0.5)`, (t) => {
      const k = kepler(0.5)
      const steps = method === 'euler' || method === 'backward-euler' ? [20000, 40000, 80000, 160000] : [1000, 2000, 4000, 8000]
      const res = convergenceStudy(method, k, 2 * Math.PI, steps)
      t.diagnostic(`order=${res.order.toFixed(3)}`)
      assert.ok(Math.abs(res.order - expected) < 0.2, `observed order ${res.order}`)
    })
  }
})

describe('Dormand–Prince RK45 (adaptive)', () => {
  it('meets the requested tolerance on problems with analytic solutions', (t) => {
    // Harmonic oscillator and logistic growth y' = y(1 − y), y = 1/(1 + 9e^{−t}).
    const logisticGrowth: OdeProblem = { dim: 1, rhs: (_t, y, o) => (o[0] = y[0] * (1 - y[0])) }
    for (const tol of [1e-4, 1e-6, 1e-8, 1e-10]) {
      const h = harmonic()
      const rh = dopri45(h, 0, h.y0, 10, { rtol: tol, atol: tol, record: false })
      const ex = h.exact!(10)
      const eh = Math.max(Math.abs(rh.yEnd[0] - ex[0]), Math.abs(rh.yEnd[1] - ex[1]))
      const rl = dopri45(logisticGrowth, 0, [0.1], 10, { rtol: tol, atol: tol, record: false })
      const el = Math.abs(rl.yEnd[0] - 1 / (1 + 9 * Math.exp(-10)))
      t.diagnostic(`tol=${tol.toExponential(0)}  harmonic err=${eh.toExponential(2)} (${rh.accepted} steps)  logistic err=${el.toExponential(2)} (${rl.accepted} steps)`)
      assert.equal(rh.status, 'ok')
      assert.ok(eh < 10 * tol, `harmonic error ${eh} for tol ${tol}`)
      assert.ok(el < 10 * tol, `logistic error ${el} for tol ${tol}`)
    }
  })

  it('global error is proportional to the tolerance (Kepler, e = 0.6)', (t) => {
    const k = kepler(0.6)
    const tols = [1e-5, 1e-7, 1e-9, 1e-11]
    const errs = tols.map((tol) => {
      const r = dopri45(k, 0, k.y0, 2 * Math.PI, { rtol: tol, atol: tol, record: false })
      const ex = k.exact!(2 * Math.PI)
      return Math.max(...Array.from(r.yEnd, (v, i) => Math.abs(v - ex[i])))
    })
    const slope = logLogSlope(tols, errs).slope
    t.diagnostic(`errors ${errs.map((e) => e.toExponential(2)).join(' → ')}; d log err / d log tol = ${slope.toFixed(3)}`)
    assert.ok(slope > 0.8 && slope < 1.25)
  })

  it('exponential decay y\' = −λy matches e^{−λt}', () => {
    const p: OdeProblem = { dim: 1, rhs: (_t, y, o) => (o[0] = -3 * y[0]) }
    const r = dopri45(p, 0, [1], 2, { rtol: 1e-9, atol: 1e-12 })
    assert.ok(Math.abs(r.yEnd[0] - Math.exp(-6)) < 1e-10)
    assert.equal(r.t[r.t.length - 1], 2)
  })

  it('tighter tolerances take more steps (step-size control responds)', () => {
    const k = kepler(0.8)
    const loose = dopri45(k, 0, k.y0, 2 * Math.PI, { rtol: 1e-4, atol: 1e-6, record: false })
    const tight = dopri45(k, 0, k.y0, 2 * Math.PI, { rtol: 1e-10, atol: 1e-12, record: false })
    assert.ok(tight.accepted > 4 * loose.accepted)
  })

  it('dense output interpolates to near-solver accuracy', () => {
    const h = harmonic()
    const ts = Array.from({ length: 101 }, (_, i) => (i * 20) / 100)
    const r = dopri45(h, 0, h.y0, 20, { rtol: 1e-9, atol: 1e-12, denseAt: ts })
    assert.equal(r.dense.length, ts.length)
    let err = 0
    ts.forEach((tt, i) => (err = Math.max(err, Math.abs(r.dense[i][0] - Math.cos(tt)))))
    assert.ok(err < 1e-7, `dense error ${err}`)
  })
})

describe('symplectic integrators: long-term energy behaviour', () => {
  it('velocity Verlet energy error stays bounded over 10⁵ steps (pendulum)', (t) => {
    const p = pendulum(2.0)
    const d = energyDrift('verlet', p, 0.05, 100_000, 2000)
    // Bounded, oscillating error of size O(h²) — no secular growth.
    const firstHalf = Math.max(...d.rel.slice(0, d.rel.length / 2))
    const secondHalf = Math.max(...d.rel.slice(d.rel.length / 2))
    t.diagnostic(`max |ΔE/E| = ${d.max.toExponential(3)} (1st half ${firstHalf.toExponential(2)}, 2nd half ${secondHalf.toExponential(2)})`)
    assert.ok(d.max < 2e-3)
    assert.ok(secondHalf < 1.5 * firstHalf)
  })

  it('Yoshida-4 energy error is bounded and far smaller than Verlet at the same step', () => {
    const p = pendulum(2.0)
    const v = energyDrift('verlet', p, 0.05, 100_000, 500)
    const y = energyDrift('yoshida4', p, 0.05, 100_000, 500)
    assert.ok(y.max < v.max / 10)
  })

  it('RK4 (non-symplectic) shows secular energy drift where Verlet does not', (t) => {
    const k = kepler(0.5)
    const dt = 0.05
    const steps = 100_000
    const rk = energyDrift('rk4', k, dt, steps, 200)
    const vv = energyDrift('verlet', k, dt, steps, 200)
    const rkEnd = rk.rel[rk.rel.length - 1]
    const rkEarly = rk.rel[Math.floor(rk.rel.length / 10)]
    t.diagnostic(`RK4 drift ${rkEarly.toExponential(2)} → ${rkEnd.toExponential(2)}; Verlet max ${vv.max.toExponential(2)}`)
    assert.ok(rkEnd > 5 * rkEarly, 'RK4 energy error should keep growing')
    assert.ok(vv.max < 0.05)
  })
})

describe('stiff problems', () => {
  it('explicit RK4 blows up on stiff Van der Pol (μ = 100) at a step backward Euler handles', () => {
    const p = vanDerPol(100)
    const h = 0.01
    const steps = 1000
    const rk = integrateFixed('rk4', p, p.y0, 0, h, steps)
    const be = integrateFixed('backward-euler', p, p.y0, 0, h, steps)
    assert.ok(!rk.every(Number.isFinite) || Math.abs(rk[0]) > 1e6, 'RK4 should diverge')
    assert.ok(be.every(Number.isFinite) && Math.abs(be[0]) < 2.5)
  })

  it('adaptive backward Euler solves Robertson to t = 40 in few steps; reference values match', (t) => {
    const p = robertson()
    const r = backwardEulerAdaptive(p, 0, p.y0, 40, { rtol: 1e-4, atol: 1e-8, record: false })
    t.diagnostic(`steps=${r.accepted} rejected=${r.rejected} y=${Array.from(r.yEnd).map((v) => v.toExponential(4)).join(', ')}`)
    assert.equal(r.status, 'ok')
    // Reference (Hairer & Wanner): y(40) ≈ (0.7158, 9.185e-6, 0.2842)
    assert.ok(Math.abs(r.yEnd[0] - 0.7158) < 5e-3)
    assert.ok(Math.abs(r.yEnd[1] - 9.185e-6) / 9.185e-6 < 0.05)
    assert.ok(Math.abs(r.yEnd[0] + r.yEnd[1] + r.yEnd[2] - 1) < 1e-8, 'mass conservation')
    assert.ok(r.accepted < 2000)
  })

  it('Dormand–Prince needs orders of magnitude more steps than implicit on stiff VdP', (t) => {
    const p = vanDerPol(300)
    const T = 150
    const ex = dopri45(p, 0, p.y0, T, { rtol: 1e-5, atol: 1e-8, record: false })
    const im = backwardEulerAdaptive(p, 0, p.y0, T, { rtol: 1e-3, atol: 1e-6, record: false })
    t.diagnostic(`DOPRI steps=${ex.accepted}+${ex.rejected}rej, BE steps=${im.accepted}+${im.rejected}rej`)
    assert.equal(ex.status, 'ok')
    assert.equal(im.status, 'ok')
    assert.ok(ex.accepted > 10 * im.accepted)
    // Both land on the same limit cycle branch.
    assert.ok(Math.abs(ex.yEnd[0] - im.yEnd[0]) < 0.05)
  })
})
