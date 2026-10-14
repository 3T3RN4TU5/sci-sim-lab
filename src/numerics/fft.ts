/**
 * Spectral tools: iterative radix-2 Cooley–Tukey FFT (in place, bit reversal + butterflies
 * with precomputed twiddles), inverse FFT, a naive O(N²) DFT for verification, window
 * functions, the short-time Fourier transform, and windowed-sinc FIR filter design.
 */

export function isPowerOfTwo(n: number): boolean {
  return n > 0 && (n & (n - 1)) === 0
}

const twiddleCache = new Map<number, { cos: Float64Array; sin: Float64Array }>()

function twiddles(n: number): { cos: Float64Array; sin: Float64Array } {
  let t = twiddleCache.get(n)
  if (!t) {
    const half = n >> 1
    t = { cos: new Float64Array(half), sin: new Float64Array(half) }
    for (let k = 0; k < half; k++) {
      t.cos[k] = Math.cos((2 * Math.PI * k) / n)
      t.sin[k] = Math.sin((2 * Math.PI * k) / n)
    }
    twiddleCache.set(n, t)
  }
  return t
}

/**
 * In-place forward FFT, X_k = Σ x_n e^{−2πikn/N}. `re` and `im` must have the same
 * power-of-two length. With `inverse`, computes Σ X_k e^{+2πikn/N} / N.
 */
export function fft(re: Float64Array, im: Float64Array, inverse = false): void {
  const n = re.length
  if (im.length !== n) throw new Error('fft: re and im lengths differ')
  if (!isPowerOfTwo(n)) throw new Error(`fft: length ${n} is not a power of two`)
  // Bit-reversal permutation.
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1
    for (; j & bit; bit >>= 1) j ^= bit
    j ^= bit
    if (i < j) {
      let t = re[i]
      re[i] = re[j]
      re[j] = t
      t = im[i]
      im[i] = im[j]
      im[j] = t
    }
  }
  const { cos, sin } = twiddles(n)
  const sign = inverse ? 1 : -1
  for (let size = 2; size <= n; size <<= 1) {
    const half = size >> 1
    const stride = n / size
    for (let start = 0; start < n; start += size) {
      for (let k = 0; k < half; k++) {
        const wr = cos[k * stride]
        const wi = sign * sin[k * stride]
        const a = start + k
        const b = a + half
        const xr = re[b] * wr - im[b] * wi
        const xi = re[b] * wi + im[b] * wr
        re[b] = re[a] - xr
        im[b] = im[a] - xi
        re[a] += xr
        im[a] += xi
      }
    }
  }
  if (inverse) {
    for (let i = 0; i < n; i++) {
      re[i] /= n
      im[i] /= n
    }
  }
}

export function ifft(re: Float64Array, im: Float64Array): void {
  fft(re, im, true)
}

/** Naive O(N²) DFT (any length) — the reference the FFT is tested against. */
export function dft(re: ArrayLike<number>, im: ArrayLike<number>): { re: Float64Array; im: Float64Array } {
  const n = re.length
  const outRe = new Float64Array(n)
  const outIm = new Float64Array(n)
  for (let k = 0; k < n; k++) {
    let sr = 0
    let si = 0
    for (let t = 0; t < n; t++) {
      const a = (-2 * Math.PI * ((k * t) % n)) / n
      const c = Math.cos(a)
      const s = Math.sin(a)
      sr += re[t] * c - im[t] * s
      si += re[t] * s + im[t] * c
    }
    outRe[k] = sr
    outIm[k] = si
  }
  return { re: outRe, im: outIm }
}

export type WindowId = 'rect' | 'hann' | 'hamming' | 'blackman'

export const WINDOW_NAMES: Record<WindowId, string> = {
  rect: 'Rectangular',
  hann: 'Hann',
  hamming: 'Hamming',
  blackman: 'Blackman',
}

/** Symmetric window of length n. */
export function windowFn(id: WindowId, n: number): Float64Array {
  const w = new Float64Array(n)
  const d = n - 1 || 1
  for (let i = 0; i < n; i++) {
    const x = (2 * Math.PI * i) / d
    switch (id) {
      case 'rect':
        w[i] = 1
        break
      case 'hann':
        w[i] = 0.5 - 0.5 * Math.cos(x)
        break
      case 'hamming':
        w[i] = 0.54 - 0.46 * Math.cos(x)
        break
      case 'blackman':
        w[i] = 0.42 - 0.5 * Math.cos(x) + 0.08 * Math.cos(2 * x)
        break
    }
  }
  return w
}

/**
 * One-sided amplitude spectrum of a real signal: |X_k|·2/Σw for k = 0…N/2,
 * so a sinusoid of amplitude A on a bin centre reads ≈ A regardless of window.
 */
export function amplitudeSpectrum(signal: ArrayLike<number>, window: WindowId = 'hann'): Float64Array {
  const n = signal.length
  const w = windowFn(window, n)
  const re = new Float64Array(n)
  const im = new Float64Array(n)
  let sw = 0
  for (let i = 0; i < n; i++) {
    re[i] = signal[i] * w[i]
    sw += w[i]
  }
  fft(re, im)
  const out = new Float64Array(n / 2 + 1)
  for (let k = 0; k <= n / 2; k++) out[k] = (Math.hypot(re[k], im[k]) * (k === 0 || k === n / 2 ? 1 : 2)) / sw
  return out
}

/**
 * Short-time Fourier transform magnitude in dB. Returns frames×(size/2+1) values,
 * frame-major, plus the frame count.
 */
export function stft(signal: ArrayLike<number>, size: number, hop: number, window: WindowId = 'hann'): { frames: number; bins: number; db: Float32Array } {
  const w = windowFn(window, size)
  const frames = Math.max(0, Math.floor((signal.length - size) / hop) + 1)
  const bins = size / 2 + 1
  const db = new Float32Array(frames * bins)
  const re = new Float64Array(size)
  const im = new Float64Array(size)
  let sw = 0
  for (let i = 0; i < size; i++) sw += w[i]
  for (let f = 0; f < frames; f++) {
    const off = f * hop
    for (let i = 0; i < size; i++) {
      re[i] = signal[off + i] * w[i]
      im[i] = 0
    }
    fft(re, im)
    for (let k = 0; k < bins; k++) {
      const a = (2 * Math.hypot(re[k], im[k])) / sw
      db[f * bins + k] = 20 * Math.log10(a + 1e-12)
    }
  }
  return { frames, bins, db }
}

export type FilterType = 'lowpass' | 'highpass' | 'bandpass'

/**
 * Windowed-sinc FIR design (odd number of taps, linear phase). Cutoffs are in Hz,
 * `fs` is the sample rate. High-pass by spectral inversion; band-pass as the
 * difference of two low-pass kernels.
 */
export function designFir(type: FilterType, taps: number, fs: number, f1: number, f2 = f1, window: WindowId = 'blackman'): Float64Array {
  const M = taps % 2 === 0 ? taps + 1 : taps
  const w = windowFn(window, M)
  const mid = (M - 1) / 2
  const lowpass = (fc: number) => {
    const h = new Float64Array(M)
    const wc = fc / fs // normalised cutoff (cycles/sample)
    let sum = 0
    for (let i = 0; i < M; i++) {
      const k = i - mid
      h[i] = (k === 0 ? 2 * wc : Math.sin(2 * Math.PI * wc * k) / (Math.PI * k)) * w[i]
      sum += h[i]
    }
    for (let i = 0; i < M; i++) h[i] /= sum // unit DC gain
    return h
  }
  if (type === 'lowpass') return lowpass(f1)
  if (type === 'highpass') {
    const h = lowpass(f1)
    for (let i = 0; i < M; i++) h[i] = -h[i]
    h[mid] += 1
    return h
  }
  const lo = lowpass(Math.min(f1, f2))
  const hi = lowpass(Math.max(f1, f2))
  const h = new Float64Array(M)
  for (let i = 0; i < M; i++) h[i] = hi[i] - lo[i]
  return h
}

/** |H(f)| sampled at nfft/2+1 frequencies (zero-padded FFT of the taps). */
export function frequencyResponse(h: ArrayLike<number>, nfft = 1024): Float64Array {
  const re = new Float64Array(nfft)
  const im = new Float64Array(nfft)
  for (let i = 0; i < h.length && i < nfft; i++) re[i] = h[i]
  fft(re, im)
  const out = new Float64Array(nfft / 2 + 1)
  for (let k = 0; k <= nfft / 2; k++) out[k] = Math.hypot(re[k], im[k])
  return out
}

/** Linear-phase FIR filtering with the group delay removed ("same" convolution). */
export function applyFir(x: ArrayLike<number>, h: ArrayLike<number>): Float64Array {
  const n = x.length
  const M = h.length
  const mid = (M - 1) >> 1
  const y = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    let s = 0
    for (let k = 0; k < M; k++) {
      const j = i + mid - k
      if (j >= 0 && j < n) s += h[k] * x[j]
    }
    y[i] = s
  }
  return y
}
