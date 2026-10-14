/**
 * Dense LU factorisation with partial pivoting for the small systems that
 * appear inside implicit ODE steps (Newton iterations), plus the Thomas
 * algorithm for the tridiagonal systems of ADI / Crank–Nicolson schemes.
 * Matrices are row-major Float64Arrays.
 */

/** In-place LU of an n×n matrix. Returns false when the matrix is singular to working precision. */
export function luFactor(a: Float64Array, n: number, piv: Int32Array): boolean {
  for (let k = 0; k < n; k++) {
    let p = k
    let max = Math.abs(a[k * n + k])
    for (let i = k + 1; i < n; i++) {
      const v = Math.abs(a[i * n + k])
      if (v > max) {
        max = v
        p = i
      }
    }
    piv[k] = p
    if (max === 0 || !Number.isFinite(max)) return false
    if (p !== k) {
      for (let j = 0; j < n; j++) {
        const t = a[k * n + j]
        a[k * n + j] = a[p * n + j]
        a[p * n + j] = t
      }
    }
    const inv = 1 / a[k * n + k]
    for (let i = k + 1; i < n; i++) {
      const l = (a[i * n + k] *= inv)
      if (l !== 0) for (let j = k + 1; j < n; j++) a[i * n + j] -= l * a[k * n + j]
    }
  }
  return true
}

/** Solve LU·x = b in place (b becomes x) using the output of luFactor. */
export function luSolve(lu: Float64Array, n: number, piv: Int32Array, b: Float64Array): void {
  for (let k = 0; k < n; k++) {
    const p = piv[k]
    if (p !== k) {
      const t = b[k]
      b[k] = b[p]
      b[p] = t
    }
  }
  for (let i = 1; i < n; i++) {
    let s = b[i]
    for (let j = 0; j < i; j++) s -= lu[i * n + j] * b[j]
    b[i] = s
  }
  for (let i = n - 1; i >= 0; i--) {
    let s = b[i]
    for (let j = i + 1; j < n; j++) s -= lu[i * n + j] * b[j]
    b[i] = s / lu[i * n + i]
  }
}

/**
 * Thomas algorithm for a tridiagonal system with constant coefficients
 * (sub = a, diag = b, super = c): solves in place in `d`, using `scratch`.
 * O(n), stable for diagonally dominant systems such as (I − r·Δ).
 */
export function thomasConst(a: number, b: number, c: number, d: Float64Array, n: number, scratch: Float64Array): void {
  scratch[0] = c / b
  d[0] = d[0] / b
  for (let i = 1; i < n; i++) {
    const m = 1 / (b - a * scratch[i - 1])
    scratch[i] = c * m
    d[i] = (d[i] - a * d[i - 1]) * m
  }
  for (let i = n - 2; i >= 0; i--) d[i] -= scratch[i] * d[i + 1]
}
