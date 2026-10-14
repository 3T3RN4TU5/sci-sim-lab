/**
 * Derivative-free minimisation with the Nelder–Mead simplex method
 * (reflection α = 1, expansion γ = 2, contraction ρ = ½, shrink σ = ½; Lagarias et al. 1998).
 */

export interface NelderMeadOptions {
  /** Initial simplex step per coordinate (default: 5% of |x₀ᵢ|, or 0.00025 when x₀ᵢ = 0). */
  step?: number | number[]
  maxIter?: number
  /** Stop when the spread of function values and the simplex diameter fall below these. */
  fTol?: number
  xTol?: number
  /** Record the best point of each iteration (for animating convergence). */
  history?: boolean
}

export interface NelderMeadResult {
  x: number[]
  fx: number
  iterations: number
  evaluations: number
  converged: boolean
  history: { x: number[]; fx: number }[]
}

export function nelderMead(f: (x: number[]) => number, x0: number[], opts: NelderMeadOptions = {}): NelderMeadResult {
  const n = x0.length
  const maxIter = opts.maxIter ?? 200 * n
  const fTol = opts.fTol ?? 1e-10
  const xTol = opts.xTol ?? 1e-10
  let evals = 0
  const F = (x: number[]) => {
    evals++
    const v = f(x)
    return Number.isFinite(v) ? v : Infinity
  }
  const simplex: number[][] = [x0.slice()]
  for (let i = 0; i < n; i++) {
    const x = x0.slice()
    const s = Array.isArray(opts.step) ? opts.step[i] : opts.step
    x[i] = s !== undefined ? x[i] + s : x[i] !== 0 ? x[i] * 1.05 : 0.00025
    simplex.push(x)
  }
  let values = simplex.map(F)
  const history: { x: number[]; fx: number }[] = []
  let iter = 0
  let converged = false

  const order = () => {
    const idx = values.map((_, i) => i).sort((a, b) => values[a] - values[b])
    const s2 = idx.map((i) => simplex[i])
    const v2 = idx.map((i) => values[i])
    for (let i = 0; i <= n; i++) {
      simplex[i] = s2[i]
      values[i] = v2[i]
    }
  }

  order()
  while (iter < maxIter) {
    iter++
    if (opts.history) history.push({ x: simplex[0].slice(), fx: values[0] })
    // Convergence: function spread and simplex size.
    let diam = 0
    for (let i = 1; i <= n; i++) for (let j = 0; j < n; j++) diam = Math.max(diam, Math.abs(simplex[i][j] - simplex[0][j]))
    if (Math.abs(values[n] - values[0]) <= fTol * (Math.abs(values[0]) + 1e-30) + 1e-300 && diam <= xTol * (1 + Math.max(...simplex[0].map(Math.abs)))) {
      converged = true
      break
    }
    const c = new Array<number>(n).fill(0)
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) c[j] += simplex[i][j] / n
    const worst = simplex[n]
    const xr = c.map((cj, j) => cj + (cj - worst[j]))
    const fr = F(xr)
    if (fr < values[0]) {
      const xe = c.map((cj, j) => cj + 2 * (cj - worst[j]))
      const fe = F(xe)
      if (fe < fr) {
        simplex[n] = xe
        values[n] = fe
      } else {
        simplex[n] = xr
        values[n] = fr
      }
    } else if (fr < values[n - 1]) {
      simplex[n] = xr
      values[n] = fr
    } else {
      const outside = fr < values[n]
      const xc = outside ? c.map((cj, j) => cj + 0.5 * (xr[j] - cj)) : c.map((cj, j) => cj + 0.5 * (worst[j] - cj))
      const fc = F(xc)
      if (fc < (outside ? fr : values[n])) {
        simplex[n] = xc
        values[n] = fc
      } else {
        for (let i = 1; i <= n; i++) {
          simplex[i] = simplex[i].map((v, j) => simplex[0][j] + 0.5 * (v - simplex[0][j]))
        }
        values = [values[0], ...simplex.slice(1).map(F)]
      }
    }
    order()
  }
  if (opts.history) history.push({ x: simplex[0].slice(), fx: values[0] })
  return { x: simplex[0], fx: values[0], iterations: iter, evaluations: evals, converged, history }
}
