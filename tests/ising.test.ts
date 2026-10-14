import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import { Ising, measure, onsagerEnergy, onsagerMagnetization, T_CRITICAL } from '../src/numerics/ising.ts'

describe('2D Ising model', () => {
  it('critical temperature constant is 2/ln(1+√2) ≈ 2.269185', () => {
    assert.ok(Math.abs(T_CRITICAL - 2.269185314) < 1e-8)
  })

  it('incremental E and M bookkeeping matches a full recomputation', () => {
    for (const algo of ['metropolis', 'wolff'] as const) {
      const m = new Ising(24, 2.4, 3, 'hot')
      for (let i = 0; i < 50; i++) m.sweep(algo)
      const { energy, magnetization } = m
      m.recompute()
      assert.equal(energy, m.energy)
      assert.equal(magnetization, m.magnetization)
    }
  })

  it('low temperature (T = 0.5): ground state, e = −2, |m| = 1', (t) => {
    const m = new Ising(32, 0.5, 1, 'cold')
    const p = measure(m, 'metropolis', 100, 200)
    t.diagnostic(`e = ${p.e.toFixed(6)}, |m| = ${p.m.toFixed(6)}`)
    assert.ok(Math.abs(p.e + 2) < 1e-3)
    assert.ok(Math.abs(p.m - 1) < 1e-3)
  })

  it('high temperature (T = 100): disordered, e ≈ −2·tanh(1/T), |m| ≈ O(1/L)', (t) => {
    const m = new Ising(32, 100, 2, 'hot')
    const p = measure(m, 'metropolis', 100, 400)
    const eHigh = -2 * Math.tanh(1 / 100)
    t.diagnostic(`e = ${p.e.toFixed(5)} (high-T series ${eHigh.toFixed(5)}), |m| = ${p.m.toFixed(4)}`)
    assert.ok(Math.abs(p.e - eHigh) < 5e-3)
    assert.ok(p.m < 0.05)
  })

  for (const algo of ['metropolis', 'wolff'] as const) {
    it(`${algo}: energy and magnetisation match Onsager's exact solution at T = 2.0 (L = 32)`, (t) => {
      const m = new Ising(32, 2.0, 7, 'cold')
      const p = measure(m, algo, 300, 3000)
      const eEx = onsagerEnergy(2.0)
      const mEx = onsagerMagnetization(2.0)
      t.diagnostic(`e = ${p.e.toFixed(4)} (exact ${eEx.toFixed(4)}), |m| = ${p.m.toFixed(4)} (exact ${mEx.toFixed(4)})`)
      assert.ok(Math.abs(p.e - eEx) < 0.01)
      assert.ok(Math.abs(p.m - mEx) < 0.01)
    })
  }

  it('Onsager energy at T_c equals −√2', () => {
    assert.ok(Math.abs(onsagerEnergy(T_CRITICAL) + Math.SQRT2) < 1e-6)
  })

  it('susceptibility and specific heat peak near T_c (Wolff, L = 24)', (t) => {
    const Ts = [1.8, 2.1, 2.3, 2.5, 2.9]
    const m = new Ising(24, Ts[0], 11, 'cold')
    const pts = Ts.map((T) => {
      m.setTemperature(T)
      return measure(m, 'wolff', 200, 1500)
    })
    const iChi = pts.reduce((b, p, i) => (p.chi > pts[b].chi ? i : b), 0)
    const iC = pts.reduce((b, p, i) => (p.c > pts[b].c ? i : b), 0)
    t.diagnostic(pts.map((p) => `T=${p.T}: χ=${p.chi.toFixed(2)} C=${p.c.toFixed(2)}`).join('; '))
    assert.ok(Ts[iChi] >= 2.1 && Ts[iChi] <= 2.5)
    assert.ok(Ts[iC] >= 2.1 && Ts[iC] <= 2.5)
  })
})
