/**
 * Small, fast, seedable pseudo-random number generators.
 *
 * `Rng` is xoshiro128** (Blackman & Vigna): 128-bit state, period 2^128 − 1,
 * passes BigCrush. Seeding goes through SplitMix32 so that nearby seeds give
 * uncorrelated streams. Everything is deterministic, which keeps tests and
 * Monte Carlo experiments reproducible.
 */
export class Rng {
  private s0: number
  private s1: number
  private s2: number
  private s3: number
  private spare = NaN

  constructor(seed = 0x9e3779b9) {
    let x = seed >>> 0
    const next = () => {
      x = (x + 0x9e3779b9) >>> 0
      let z = x
      z = Math.imul(z ^ (z >>> 16), 0x85ebca6b) >>> 0
      z = Math.imul(z ^ (z >>> 13), 0xc2b2ae35) >>> 0
      return (z ^ (z >>> 16)) >>> 0
    }
    this.s0 = next()
    this.s1 = next()
    this.s2 = next()
    this.s3 = next()
    if ((this.s0 | this.s1 | this.s2 | this.s3) === 0) this.s0 = 1
  }

  /** Uniform 32-bit unsigned integer. */
  nextU32(): number {
    const result = Math.imul(rotl(Math.imul(this.s1, 5), 7), 9) >>> 0
    const t = this.s1 << 9
    this.s2 ^= this.s0
    this.s3 ^= this.s1
    this.s1 ^= this.s2
    this.s0 ^= this.s3
    this.s2 ^= t
    this.s3 = rotl(this.s3, 11)
    return result
  }

  /** Uniform in [0, 1) with 32 bits of resolution. */
  next(): number {
    return this.nextU32() / 4294967296
  }

  /** Uniform integer in [0, n). */
  int(n: number): number {
    return Math.floor(this.next() * n)
  }

  /** Uniform in [a, b). */
  uniform(a: number, b: number): number {
    return a + (b - a) * this.next()
  }

  /** Standard normal deviate (Marsaglia polar method, caches the spare). */
  normal(): number {
    if (!Number.isNaN(this.spare)) {
      const s = this.spare
      this.spare = NaN
      return s
    }
    let u: number, v: number, s: number
    do {
      u = 2 * this.next() - 1
      v = 2 * this.next() - 1
      s = u * u + v * v
    } while (s >= 1 || s === 0)
    const f = Math.sqrt((-2 * Math.log(s)) / s)
    this.spare = v * f
    return u * f
  }
}

function rotl(x: number, k: number): number {
  return (x << k) | (x >>> (32 - k))
}

/** Van der Corput radical inverse in base `b` — the building block of Halton sequences. */
export function radicalInverse(i: number, b: number): number {
  let f = 1
  let r = 0
  while (i > 0) {
    f /= b
    r += f * (i % b)
    i = Math.floor(i / b)
  }
  return r
}
