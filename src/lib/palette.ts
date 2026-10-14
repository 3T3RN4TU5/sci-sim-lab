import type { MethodId } from '../numerics/ode.ts'

/** Categorical palette validated for colour-vision-deficiency separation on the dark surface. */
export const C = {
  blue: '#3987e5',
  orange: '#d95926',
  aqua: '#199e70',
  yellow: '#c98500',
  magenta: '#d55181',
  violet: '#9085e9',
  red: '#e66767',
  ink: '#e2e8f0',
  muted: '#94a3b8',
  grid: 'rgba(148, 163, 184, 0.14)',
} as const

/** Each integrator keeps the same colour on every chart. */
export const METHOD_COLOR: Record<MethodId | 'dopri45', string> = {
  euler: C.red,
  midpoint: C.yellow,
  rk4: C.blue,
  verlet: C.aqua,
  yoshida4: C.violet,
  'backward-euler': C.magenta,
  dopri45: C.orange,
}
