import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import { piErrorScaling, RandomWalkers } from '../src/numerics/montecarlo.ts'
import { linearFit } from '../src/numerics/stats.ts'

describe('Monte Carlo', () => {
  it('Monte Carlo error scales as N^(−1/2); quasi-Monte Carlo as ≈ N^(−1)', (t) => {
    const s = piErrorScaling(6, 15, 64, 2)
    t.diagnostic(`hit-or-miss slope ${s.hitMissSlope.toFixed(3)}, mean-value slope ${s.meanValueSlope.toFixed(3)}, QMC slope ${s.qmcSlope.toFixed(3)}`)
    assert.ok(Math.abs(s.hitMissSlope + 0.5) < 0.1)
    assert.ok(Math.abs(s.meanValueSlope + 0.5) < 0.1)
    // Mean-value sampling has lower variance than hit-or-miss at every N.
    s.n.forEach((_, i) => assert.ok(s.meanValue[i] < s.hitMiss[i]))
    assert.ok(s.qmcSlope < -0.85)
  })

  it('2D lattice random walk: MSD(t) = t (diffusive, D = 1/4)', (t) => {
    const w = new RandomWalkers(20000, 4)
    const ts: number[] = []
    const msd: number[] = []
    for (let s = 1; s <= 400; s++) {
      w.step()
      if (s % 20 === 0) {
        ts.push(s)
        msd.push(w.msd())
      }
    }
    const fit = linearFit(ts, msd)
    t.diagnostic(`MSD slope ${fit.slope.toFixed(4)} (expected 1), R² ${fit.r2.toFixed(5)}`)
    assert.ok(Math.abs(fit.slope - 1) < 0.05)
  })
})
