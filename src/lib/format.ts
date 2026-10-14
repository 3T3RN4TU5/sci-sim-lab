/** Compact number formatting for readouts and axis ticks. */
export function fmt(v: number, digits = 3): string {
  if (!Number.isFinite(v)) return Number.isNaN(v) ? 'NaN' : v > 0 ? '∞' : '−∞'
  const a = Math.abs(v)
  if (a === 0) return '0'
  if (a >= 1e5 || a < 1e-3) return v.toExponential(Math.max(0, digits - 1)).replace('e+', 'e').replace('-', '−')
  return Number(v.toPrecision(digits)).toString().replace('-', '−')
}

export function fmtFixed(v: number, d = 2): string {
  return Number.isFinite(v) ? v.toFixed(d).replace('-', '−') : fmt(v)
}
