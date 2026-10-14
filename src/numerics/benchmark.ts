/**
 * Scaling benchmark: wall time of one force evaluation for direct summation vs
 * Barnes–Hut (monopole and quadrupole) as N grows, plus the RMS force error.
 */
import { cloneBodies, directAccel, QuadTree, randomCloud } from './nbody.ts'
import { logLogSlope } from './stats.ts'

export interface BenchPoint {
  n: number
  directMs: number
  bhMs: number
  bhQuadMs: number
  errMono: number
  errQuad: number
}

function time(fn: () => void, reps: number): number {
  fn() // warm-up: let the JIT compile the hot loops before timing
  const ts: number[] = []
  for (let r = 0; r < reps; r++) {
    const t0 = performance.now()
    fn()
    ts.push(performance.now() - t0)
  }
  ts.sort((a, b) => a - b)
  return ts[Math.floor(ts.length / 2)]
}

export function benchmarkPoint(n: number, theta: number, eps = 0.01): BenchPoint {
  const b = randomCloud(n, 3)
  const d = cloneBodies(b)
  const reps = n <= 2000 ? 5 : 3
  const directMs = time(() => directAccel(d, 1, eps), n > 8000 ? 1 : reps)
  const tree = new QuadTree()
  const mono = cloneBodies(b)
  const bhMs = time(() => {
    tree.build(mono)
    tree.accel(mono, theta, 1, eps, false)
  }, reps)
  const quad = cloneBodies(b)
  const bhQuadMs = time(() => {
    tree.build(quad)
    tree.accel(quad, theta, 1, eps, true)
  }, reps)
  let ms = 0
  let em = 0
  let eq = 0
  for (let i = 0; i < n; i++) {
    ms += d.ax[i] ** 2 + d.ay[i] ** 2
    em += (mono.ax[i] - d.ax[i]) ** 2 + (mono.ay[i] - d.ay[i]) ** 2
    eq += (quad.ax[i] - d.ax[i]) ** 2 + (quad.ay[i] - d.ay[i]) ** 2
  }
  return { n, directMs, bhMs, bhQuadMs, errMono: Math.sqrt(em / ms), errQuad: Math.sqrt(eq / ms) }
}

export function scalingSlopes(pts: BenchPoint[]): { direct: number; bh: number } {
  const big = pts.filter((p) => p.n >= 1000)
  const use = big.length >= 2 ? big : pts
  return { direct: logLogSlope(use.map((p) => p.n), use.map((p) => p.directMs)).slope, bh: logLogSlope(use.map((p) => p.n), use.map((p) => p.bhMs)).slope }
}
