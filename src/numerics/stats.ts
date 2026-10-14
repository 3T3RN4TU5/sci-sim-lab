/** Ordinary least-squares fit y = a + b·x. Returns slope, intercept and R². */
export function linearFit(xs: ArrayLike<number>, ys: ArrayLike<number>): { slope: number; intercept: number; r2: number } {
  const n = xs.length
  let sx = 0
  let sy = 0
  for (let i = 0; i < n; i++) {
    sx += xs[i]
    sy += ys[i]
  }
  const mx = sx / n
  const my = sy / n
  let sxx = 0
  let sxy = 0
  let syy = 0
  for (let i = 0; i < n; i++) {
    const dx = xs[i] - mx
    const dy = ys[i] - my
    sxx += dx * dx
    sxy += dx * dy
    syy += dy * dy
  }
  const slope = sxy / sxx
  const intercept = my - slope * mx
  const r2 = syy === 0 ? 1 : (sxy * sxy) / (sxx * syy)
  return { slope, intercept, r2 }
}

/** Slope of log10(y) against log10(x): the empirical order / scaling exponent. */
export function logLogSlope(xs: ArrayLike<number>, ys: ArrayLike<number>): { slope: number; intercept: number; r2: number } {
  const lx: number[] = []
  const ly: number[] = []
  for (let i = 0; i < xs.length; i++) {
    if (xs[i] > 0 && ys[i] > 0 && Number.isFinite(ys[i])) {
      lx.push(Math.log10(xs[i]))
      ly.push(Math.log10(ys[i]))
    }
  }
  return linearFit(lx, ly)
}

export function mean(xs: ArrayLike<number>): number {
  let s = 0
  for (let i = 0; i < xs.length; i++) s += xs[i]
  return s / xs.length
}

export function median(xs: ArrayLike<number>): number {
  const a = Array.from(xs).sort((p, q) => p - q)
  const m = a.length >> 1
  return a.length % 2 ? a[m] : 0.5 * (a[m - 1] + a[m])
}
