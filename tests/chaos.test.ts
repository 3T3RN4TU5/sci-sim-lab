import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import { bifurcation, LyapunovEstimator, logisticLyapunov } from '../src/numerics/chaos.ts'
import { lorenz } from '../src/numerics/systems.ts'

describe('chaos', () => {
  it('Benettin estimate of the largest Lorenz Lyapunov exponent ≈ 0.906', (t) => {
    const est = new LyapunovEstimator(lorenz(), [1, 1, 1], { h: 0.005, renormEvery: 10 })
    est.advance(4000) // discard transient
    est.sumLog = 0
    const t0 = est.t
    const steps = 200_000
    est.advance(steps)
    const lambda = est.sumLog / (est.t - t0)
    t.diagnostic(`λ₁ = ${lambda.toFixed(4)} over t = ${(est.t - t0).toFixed(0)}`)
    assert.ok(Math.abs(lambda - 0.9056) < 0.03)
  })

  it('logistic map: λ(4) = ln 2, λ < 0 in the period-2 window, period doubling at r = 3.2', () => {
    assert.ok(Math.abs(logisticLyapunov(4, 200_000) - Math.LN2) < 0.01)
    assert.ok(logisticLyapunov(3.2) < 0)
    const pts = bifurcation(3.2, 3.2, 1, 2000, 16)
    const distinct = new Set(Array.from(pts, (x) => x.toFixed(5)))
    assert.equal(distinct.size, 2)
    const pts4 = bifurcation(3.5, 3.5, 1, 4000, 16)
    assert.equal(new Set(Array.from(pts4, (x) => x.toFixed(5))).size, 4)
  })
})
