import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import { amplitudeSpectrum, applyFir, designFir, dft, fft, frequencyResponse, ifft, stft } from '../src/numerics/fft.ts'
import { Rng } from '../src/numerics/rng.ts'

function randomSignal(n: number, seed: number) {
  const rng = new Rng(seed)
  return { re: Float64Array.from({ length: n }, () => rng.normal()), im: Float64Array.from({ length: n }, () => rng.normal()) }
}

describe('radix-2 FFT', () => {
  it('matches the naive O(N²) DFT for N = 1 … 1024', (t) => {
    let worst = 0
    for (let k = 0; k <= 10; k++) {
      const n = 2 ** k
      const { re, im } = randomSignal(n, 100 + k)
      const ref = dft(re, im)
      const r2 = re.slice()
      const i2 = im.slice()
      fft(r2, i2)
      for (let i = 0; i < n; i++) worst = Math.max(worst, Math.abs(r2[i] - ref.re[i]) / Math.sqrt(n), Math.abs(i2[i] - ref.im[i]) / Math.sqrt(n))
    }
    t.diagnostic(`max |FFT − DFT| / √N = ${worst.toExponential(2)}`)
    assert.ok(worst < 1e-12)
  })

  it('inverse FFT round-trips to machine precision', () => {
    for (const n of [2, 64, 4096, 65536]) {
      const { re, im } = randomSignal(n, n)
      const r2 = re.slice()
      const i2 = im.slice()
      fft(r2, i2)
      ifft(r2, i2)
      for (let i = 0; i < n; i++) {
        assert.ok(Math.abs(r2[i] - re[i]) < 1e-12)
        assert.ok(Math.abs(i2[i] - im[i]) < 1e-12)
      }
    }
  })

  it("satisfies Parseval's theorem Σ|x|² = (1/N) Σ|X|²", (t) => {
    const n = 8192
    const { re, im } = randomSignal(n, 9)
    let time = 0
    for (let i = 0; i < n; i++) time += re[i] ** 2 + im[i] ** 2
    fft(re, im)
    let freq = 0
    for (let i = 0; i < n; i++) freq += re[i] ** 2 + im[i] ** 2
    freq /= n
    t.diagnostic(`relative difference ${(Math.abs(time - freq) / time).toExponential(2)}`)
    assert.ok(Math.abs(time - freq) / time < 1e-12)
  })

  it('is linear and maps a pure tone to a single bin', () => {
    const n = 256
    const re = Float64Array.from({ length: n }, (_, i) => Math.cos((2 * Math.PI * 17 * i) / n))
    const im = new Float64Array(n)
    fft(re, im)
    for (let k = 0; k < n; k++) {
      const mag = Math.hypot(re[k], im[k])
      if (k === 17 || k === n - 17) assert.ok(Math.abs(mag - n / 2) < 1e-9)
      else assert.ok(mag < 1e-9)
    }
  })

  it('rejects lengths that are not powers of two', () => {
    assert.throws(() => fft(new Float64Array(6), new Float64Array(6)))
  })
})

describe('spectral analysis and filtering', () => {
  it('amplitude spectrum recovers the amplitude of bin-centred sinusoids', () => {
    const fs = 1024
    const x = Float64Array.from({ length: 1024 }, (_, i) => 1.5 * Math.sin((2 * Math.PI * 50 * i) / fs) + 0.5 * Math.sin((2 * Math.PI * 200 * i) / fs))
    const a = amplitudeSpectrum(x, 'rect')
    assert.ok(Math.abs(a[50] - 1.5) < 1e-9)
    assert.ok(Math.abs(a[200] - 0.5) < 1e-9)
  })

  it('STFT of a linear chirp tracks the instantaneous frequency', () => {
    const fs = 2048
    const n = 8192
    const f0 = 50
    const f1 = 800
    const T = n / fs
    const x = Float64Array.from({ length: n }, (_, i) => {
      const t = i / fs
      return Math.sin(2 * Math.PI * (f0 * t + ((f1 - f0) / (2 * T)) * t * t))
    })
    const size = 256
    const hop = 128
    const s = stft(x, size, hop, 'hann')
    for (let f = 2; f < s.frames - 2; f += 7) {
      let best = 0
      for (let k = 1; k < s.bins; k++) if (s.db[f * s.bins + k] > s.db[f * s.bins + best]) best = k
      const tc = (f * hop + size / 2) / fs
      const expected = f0 + ((f1 - f0) * tc) / T
      assert.ok(Math.abs((best * fs) / size - expected) <= 2 * (fs / size), `frame ${f}: ${(best * fs) / size} vs ${expected}`)
    }
  })

  it('windowed-sinc low-pass FIR: unity passband, > 60 dB stopband attenuation (Blackman)', (t) => {
    const fs = 1000
    const h = designFir('lowpass', 101, fs, 100, 100, 'blackman')
    const H = frequencyResponse(h, 4096)
    const bin = (f: number) => Math.round((f / fs) * 4096)
    let passDev = 0
    for (let f = 0; f <= 60; f++) passDev = Math.max(passDev, Math.abs(H[bin(f)] - 1))
    let stop = 0
    for (let f = 160; f <= 500; f++) stop = Math.max(stop, H[bin(f)])
    t.diagnostic(`passband ripple ${passDev.toExponential(2)}, stopband ${(20 * Math.log10(stop)).toFixed(1)} dB`)
    assert.ok(passDev < 1e-3)
    assert.ok(20 * Math.log10(stop) < -60)
  })

  it('filtering removes an out-of-band tone', () => {
    const fs = 1000
    const n = 2048
    const lo = Float64Array.from({ length: n }, (_, i) => Math.sin((2 * Math.PI * 20 * i) / fs))
    const x = lo.map((v, i) => v + Math.sin((2 * Math.PI * 300 * i) / fs))
    const y = applyFir(x, designFir('lowpass', 151, fs, 80))
    let err = 0
    for (let i = 200; i < n - 200; i++) err = Math.max(err, Math.abs(y[i] - lo[i]))
    assert.ok(err < 1e-3, `residual ${err}`)
  })
})
