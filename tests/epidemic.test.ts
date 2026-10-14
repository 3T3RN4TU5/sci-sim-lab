import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import { finalSize, fitEpidemic, simulateEpidemic, syntheticData } from '../src/numerics/epidemic.ts'
import { nelderMead } from '../src/numerics/optimize.ts'

describe('Nelder–Mead', () => {
  it('minimises the Rosenbrock function to (1, 1)', (t) => {
    const r = nelderMead((x) => 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2, [-1.2, 1], { maxIter: 2000 })
    t.diagnostic(`x = (${r.x.map((v) => v.toFixed(6)).join(', ')}), ${r.iterations} iterations, ${r.evaluations} evaluations`)
    assert.ok(r.converged)
    assert.ok(Math.abs(r.x[0] - 1) < 1e-4 && Math.abs(r.x[1] - 1) < 1e-4)
  })

  it('minimises a 4-D quadratic', () => {
    const c = [1, -2, 3, 0.5]
    const r = nelderMead((x) => x.reduce((s, v, i) => s + (i + 1) * (v - c[i]) ** 2, 0), [0, 0, 0, 0], { step: 1, maxIter: 5000 })
    r.x.forEach((v, i) => assert.ok(Math.abs(v - c[i]) < 1e-4))
  })
})

describe('SIR / SEIR epidemics', () => {
  const known = { N: 1e6, I0: 10, sigma: 1 / 5.2 }

  it('conserves the population and matches the final-size relation', () => {
    const p = { ...known, beta: 0.3, gamma: 0.1 }
    const s = simulateEpidemic('SIR', p, 400)
    const last = s.t.length - 1
    for (let d = 0; d < s.t.length; d++) assert.ok(Math.abs(s.S[d] + s.I[d] + s.R[d] - p.N) < 1e-6 * p.N)
    const z = finalSize(3)
    assert.ok(Math.abs(s.R[last] / p.N - z) < 2e-3, `${s.R[last] / p.N} vs ${z}`)
  })

  for (const model of ['SIR', 'SEIR'] as const) {
    it(`${model}: Nelder–Mead recovers β and γ from noisy synthetic data`, (t) => {
      const truth = { ...known, beta: 0.45, gamma: 0.15 }
      const data = syntheticData(model, truth, 120, 0.1, 7)
      const fit = fitEpidemic(model, data, known, { beta: 0.2, gamma: 0.3 })
      t.diagnostic(`β = ${fit.beta.toFixed(4)} (true 0.45), γ = ${fit.gamma.toFixed(4)} (true 0.15), R₀ = ${fit.R0.toFixed(3)} (true 3), ${fit.iterations} iterations`)
      assert.ok(Math.abs(fit.beta - 0.45) / 0.45 < 0.03)
      assert.ok(Math.abs(fit.gamma - 0.15) / 0.15 < 0.03)
      assert.ok(Math.abs(fit.R0 - 3) / 3 < 0.03)
    })
  }
})
