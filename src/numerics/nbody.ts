/**
 * 2D gravitational N-body: direct O(N²) summation and a Barnes–Hut quadtree
 * (O(N log N)), Plummer softening, kick–drift–kick leapfrog, conserved quantities
 * and initial-condition presets. Structure-of-arrays storage throughout.
 */
import { Rng } from './rng.ts'

export interface Bodies {
  n: number
  x: Float64Array
  y: Float64Array
  vx: Float64Array
  vy: Float64Array
  ax: Float64Array
  ay: Float64Array
  m: Float64Array
}

export function createBodies(n: number): Bodies {
  return {
    n,
    x: new Float64Array(n),
    y: new Float64Array(n),
    vx: new Float64Array(n),
    vy: new Float64Array(n),
    ax: new Float64Array(n),
    ay: new Float64Array(n),
    m: new Float64Array(n),
  }
}

export function cloneBodies(b: Bodies): Bodies {
  return { n: b.n, x: b.x.slice(), y: b.y.slice(), vx: b.vx.slice(), vy: b.vy.slice(), ax: b.ax.slice(), ay: b.ay.slice(), m: b.m.slice() }
}

/** Direct pairwise summation using Newton's third law (N(N−1)/2 interactions). */
export function directAccel(b: Bodies, G: number, eps: number): void {
  const { n, x, y, m, ax, ay } = b
  ax.fill(0)
  ay.fill(0)
  const e2 = eps * eps
  for (let i = 0; i < n; i++) {
    const xi = x[i], yi = y[i], mi = m[i]
    let axi = 0, ayi = 0
    for (let j = i + 1; j < n; j++) {
      const dx = x[j] - xi
      const dy = y[j] - yi
      const r2 = dx * dx + dy * dy + e2
      const inv = 1 / (r2 * Math.sqrt(r2))
      const fi = m[j] * inv
      const fj = mi * inv
      axi += fi * dx
      ayi += fi * dy
      ax[j] -= fj * dx
      ay[j] -= fj * dy
    }
    ax[i] += axi
    ay[i] += ayi
  }
  for (let i = 0; i < n; i++) {
    ax[i] *= G
    ay[i] *= G
  }
}

const MAX_DEPTH = 48

/**
 * Barnes–Hut quadtree stored in flat typed arrays. Each internal node owns four
 * contiguous children; leaves hold a linked list of bodies (normally one; several only
 * at MAX_DEPTH when bodies coincide). Nodes accumulate mass and mass-weighted position
 * during insertion, so the centre of mass is available without a second pass.
 */
export class QuadTree {
  cap = 0
  count = 0
  cx = new Float64Array(0)
  cy = new Float64Array(0)
  half = new Float64Array(0)
  mass = new Float64Array(0)
  mx = new Float64Array(0)
  my = new Float64Array(0)
  /** Second moments Σm·x², Σm·xy, Σm·y² (about the origin) for the quadrupole term. */
  sxx = new Float64Array(0)
  sxy = new Float64Array(0)
  syy = new Float64Array(0)
  child = new Int32Array(0)
  head = new Int32Array(0)
  nBodies = new Int32Array(0)
  next = new Int32Array(0)
  private stack = new Int32Array(256)
  /** Interactions evaluated by the last `accel` call (body–node + body–body). */
  interactions = 0

  private ensure(cap: number): void {
    if (cap <= this.cap) return
    const c = Math.max(cap, this.cap * 2, 64)
    const grow = <T extends Float64Array | Int32Array>(a: T, make: (n: number) => T): T => {
      const r = make(c)
      r.set(a.subarray(0, this.count))
      return r
    }
    const f = (n: number) => new Float64Array(n)
    const i32 = (n: number) => new Int32Array(n)
    this.cx = grow(this.cx, f)
    this.cy = grow(this.cy, f)
    this.half = grow(this.half, f)
    this.mass = grow(this.mass, f)
    this.mx = grow(this.mx, f)
    this.my = grow(this.my, f)
    this.sxx = grow(this.sxx, f)
    this.sxy = grow(this.sxy, f)
    this.syy = grow(this.syy, f)
    this.child = grow(this.child, i32)
    this.head = grow(this.head, i32)
    this.nBodies = grow(this.nBodies, i32)
    this.cap = c
  }

  private alloc(cx: number, cy: number, half: number): number {
    if (this.count >= this.cap) this.ensure(this.count + 1)
    const k = this.count++
    this.cx[k] = cx
    this.cy[k] = cy
    this.half[k] = half
    this.mass[k] = 0
    this.mx[k] = 0
    this.my[k] = 0
    this.sxx[k] = 0
    this.sxy[k] = 0
    this.syy[k] = 0
    this.child[k] = -1
    this.head[k] = -1
    this.nBodies[k] = 0
    return k
  }

  private quadrant(node: number, px: number, py: number): number {
    return (px >= this.cx[node] ? 1 : 0) + (py >= this.cy[node] ? 2 : 0)
  }

  private subdivide(node: number): number {
    const h = this.half[node] / 2
    const cx = this.cx[node], cy = this.cy[node]
    const c = this.alloc(cx - h, cy - h, h)
    this.alloc(cx + h, cy - h, h)
    this.alloc(cx - h, cy + h, h)
    this.alloc(cx + h, cy + h, h)
    this.child[node] = c
    return c
  }

  build(b: Bodies): void {
    const { n, x, y, m } = b
    this.count = 0
    this.ensure(4 * n + 16)
    if (this.next.length < n) this.next = new Int32Array(n)
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity
    for (let i = 0; i < n; i++) {
      if (x[i] < minX) minX = x[i]
      if (x[i] > maxX) maxX = x[i]
      if (y[i] < minY) minY = y[i]
      if (y[i] > maxY) maxY = y[i]
    }
    if (n === 0) {
      minX = minY = -1
      maxX = maxY = 1
    }
    const half = Math.max(maxX - minX, maxY - minY, 1e-9) * 0.5 * 1.0001
    this.alloc((minX + maxX) / 2, (minY + maxY) / 2, half)
    for (let i = 0; i < n; i++) {
      const xi = x[i], yi = y[i], mi = m[i]
      let node = 0
      let depth = 0
      for (;;) {
        this.mass[node] += mi
        this.mx[node] += mi * xi
        this.my[node] += mi * yi
        this.sxx[node] += mi * xi * xi
        this.sxy[node] += mi * xi * yi
        this.syy[node] += mi * yi * yi
        if (this.child[node] >= 0) {
          node = this.child[node] + this.quadrant(node, xi, yi)
          depth++
          continue
        }
        if (this.nBodies[node] === 0) {
          this.head[node] = i
          this.next[i] = -1
          this.nBodies[node] = 1
          break
        }
        if (depth >= MAX_DEPTH) {
          this.next[i] = this.head[node]
          this.head[node] = i
          this.nBodies[node]++
          break
        }
        // Split the occupied leaf: push its single body down one level, then keep descending.
        const j = this.head[node]
        const c = this.subdivide(node)
        const cj = c + this.quadrant(node, x[j], y[j])
        this.head[cj] = j
        this.next[j] = -1
        this.nBodies[cj] = 1
        this.mass[cj] = m[j]
        this.mx[cj] = m[j] * x[j]
        this.my[cj] = m[j] * y[j]
        this.sxx[cj] = m[j] * x[j] * x[j]
        this.sxy[cj] = m[j] * x[j] * y[j]
        this.syy[cj] = m[j] * y[j] * y[j]
        this.head[node] = -1
        this.nBodies[node] = 0
        node = c + this.quadrant(node, xi, yi)
        depth++
      }
    }
  }

  /**
   * Accelerations with opening angle θ: a node of width s at distance d from the body is
   * replaced by its multipole expansion when s/d < θ and the body lies outside the node.
   * The expansion is the monopole plus (optionally) the traceless quadrupole
   * Q_ij = Σ m (3 x_i x_j − r² δ_ij) about the node's centre of mass, which cuts the
   * force error by roughly an order of magnitude at the same θ.
   */
  accel(b: Bodies, theta: number, G: number, eps: number, quadrupole = true): void {
    const { n, x, y, ax, ay, m } = b
    const e2 = eps * eps
    const th2 = theta * theta
    let stack = this.stack
    let inter = 0
    for (let i = 0; i < n; i++) {
      const xi = x[i], yi = y[i]
      let axi = 0, ayi = 0
      let sp = 0
      stack[sp++] = 0
      while (sp > 0) {
        const node = stack[--sp]
        const M = this.mass[node]
        if (M === 0) continue
        const c = this.child[node]
        if (c < 0) {
          for (let j = this.head[node]; j >= 0; j = this.next[j]) {
            if (j === i) continue
            const dx = x[j] - xi
            const dy = y[j] - yi
            const r2 = dx * dx + dy * dy + e2
            const f = m[j] / (r2 * Math.sqrt(r2))
            axi += f * dx
            ayi += f * dy
            inter++
          }
          continue
        }
        const comx = this.mx[node] / M
        const comy = this.my[node] / M
        const dx = comx - xi
        const dy = comy - yi
        const d2 = dx * dx + dy * dy
        const h = this.half[node]
        const s = 2 * h
        const inside = Math.abs(xi - this.cx[node]) <= h && Math.abs(yi - this.cy[node]) <= h
        if (!inside && s * s < th2 * d2) {
          const r2 = d2 + e2
          const inv3 = 1 / (r2 * Math.sqrt(r2))
          axi += M * inv3 * dx
          ayi += M * inv3 * dy
          if (quadrupole) {
            // Central second moments, then Q = [[2Ixx − Iyy, 3Ixy], [3Ixy, 2Iyy − Ixx]].
            const ixx = this.sxx[node] - M * comx * comx
            const ixy = this.sxy[node] - M * comx * comy
            const iyy = this.syy[node] - M * comy * comy
            const qxx = 2 * ixx - iyy
            const qyy = 2 * iyy - ixx
            const qxy = 3 * ixy
            // r points from the centre of mass to the body; a = Q·r/r⁵ − 5/2 (r·Q·r) r/r⁷.
            const rx = -dx
            const ry = -dy
            const qrx = qxx * rx + qxy * ry
            const qry = qxy * rx + qyy * ry
            const rqr = rx * qrx + ry * qry
            const inv5 = inv3 / r2
            const inv7 = inv5 / r2
            axi += qrx * inv5 - 2.5 * rqr * rx * inv7
            ayi += qry * inv5 - 2.5 * rqr * ry * inv7
          }
          inter++
        } else {
          if (sp + 4 > stack.length) {
            const bigger = new Int32Array(stack.length * 2)
            bigger.set(stack)
            stack = this.stack = bigger
          }
          stack[sp++] = c
          stack[sp++] = c + 1
          stack[sp++] = c + 2
          stack[sp++] = c + 3
        }
      }
      ax[i] = G * axi
      ay[i] = G * ayi
    }
    this.interactions = inter
  }
}

export type ForceMethod = 'direct' | 'barnes-hut'

export interface NBodyParams {
  G: number
  eps: number
  theta: number
  method: ForceMethod
  /** Include the quadrupole term in Barnes–Hut cell expansions. */
  quadrupole?: boolean
}

export class NBodySim {
  b: Bodies
  p: NBodyParams
  t = 0
  readonly tree = new QuadTree()
  lastForceMs = 0

  constructor(b: Bodies, p: NBodyParams) {
    this.b = b
    this.p = p
    this.computeAccel()
  }

  computeAccel(): void {
    const t0 = performance.now()
    if (this.p.method === 'direct') directAccel(this.b, this.p.G, this.p.eps)
    else {
      this.tree.build(this.b)
      this.tree.accel(this.b, this.p.theta, this.p.G, this.p.eps, this.p.quadrupole ?? true)
    }
    this.lastForceMs = performance.now() - t0
  }

  /** Kick–drift–kick leapfrog (symplectic, time-reversible, second order). */
  step(dt: number): void {
    const { n, x, y, vx, vy, ax, ay } = this.b
    const h = 0.5 * dt
    for (let i = 0; i < n; i++) {
      vx[i] += h * ax[i]
      vy[i] += h * ay[i]
      x[i] += dt * vx[i]
      y[i] += dt * vy[i]
    }
    this.computeAccel()
    for (let i = 0; i < n; i++) {
      vx[i] += h * ax[i]
      vy[i] += h * ay[i]
    }
    this.t += dt
  }
}

/** Kinetic + softened potential energy (O(N²); the potential matches the Plummer force). */
export function energy(b: Bodies, G: number, eps: number): { kinetic: number; potential: number; total: number } {
  const { n, x, y, vx, vy, m } = b
  let K = 0
  let U = 0
  const e2 = eps * eps
  for (let i = 0; i < n; i++) {
    K += 0.5 * m[i] * (vx[i] * vx[i] + vy[i] * vy[i])
    for (let j = i + 1; j < n; j++) {
      const dx = x[j] - x[i]
      const dy = y[j] - y[i]
      U -= (G * m[i] * m[j]) / Math.sqrt(dx * dx + dy * dy + e2)
    }
  }
  return { kinetic: K, potential: U, total: K + U }
}

export function momentum(b: Bodies): { px: number; py: number; L: number } {
  let px = 0, py = 0, L = 0
  for (let i = 0; i < b.n; i++) {
    px += b.m[i] * b.vx[i]
    py += b.m[i] * b.vy[i]
    L += b.m[i] * (b.x[i] * b.vy[i] - b.y[i] * b.vx[i])
  }
  return { px, py, L }
}

/** Shift to the centre-of-mass frame (zero total momentum, CoM at origin). */
export function toComFrame(b: Bodies): void {
  let M = 0, cx = 0, cy = 0, px = 0, py = 0
  for (let i = 0; i < b.n; i++) {
    M += b.m[i]
    cx += b.m[i] * b.x[i]
    cy += b.m[i] * b.y[i]
    px += b.m[i] * b.vx[i]
    py += b.m[i] * b.vy[i]
  }
  for (let i = 0; i < b.n; i++) {
    b.x[i] -= cx / M
    b.y[i] -= cy / M
    b.vx[i] -= px / M
    b.vy[i] -= py / M
  }
}

// ─── Presets ────────────────────────────────────────────────────────────────

export type PresetId = 'galaxies' | 'figure8' | 'solar' | 'collapse'

export interface Preset {
  id: PresetId
  name: string
  bodies: Bodies
  params: Pick<NBodyParams, 'G' | 'eps'>
  dt: number
  /** Half-width of the initial view. */
  view: number
  /** Indices to draw as large bodies (stars / planets). */
  heavy: number[]
}

export const PRESET_NAMES: Record<PresetId, string> = {
  galaxies: 'Galaxy collision',
  figure8: 'Figure-8 three-body choreography',
  solar: 'Solar-system-like + asteroid belt',
  collapse: 'Cold collapse of a uniform disk',
}

/** Chenciner & Montgomery (2000) figure-eight: three equal masses chasing on one curve. */
export function figure8(): Preset {
  const b = createBodies(3)
  const x1 = 0.97000436, y1 = -0.24308753
  const vx3 = -0.93240737, vy3 = -0.86473146
  b.m.fill(1)
  b.x.set([x1, -x1, 0])
  b.y.set([y1, -y1, 0])
  b.vx.set([-vx3 / 2, -vx3 / 2, vx3])
  b.vy.set([-vy3 / 2, -vy3 / 2, vy3])
  return { id: 'figure8', name: PRESET_NAMES.figure8, bodies: b, params: { G: 1, eps: 0 }, dt: 0.001, view: 1.4, heavy: [0, 1, 2] }
}

function addDisk(b: Bodies, start: number, count: number, cx: number, cy: number, vcx: number, vcy: number, Mc: number, mStar: number, rMin: number, rMax: number, sense: number, rng: Rng, G: number): void {
  b.x[start] = cx
  b.y[start] = cy
  b.vx[start] = vcx
  b.vy[start] = vcy
  b.m[start] = Mc
  const radii: number[] = []
  for (let k = 0; k < count; k++) radii.push(Math.sqrt(rMin * rMin + (rMax * rMax - rMin * rMin) * rng.next()))
  radii.sort((p, q) => p - q)
  for (let k = 0; k < count; k++) {
    const i = start + 1 + k
    const r = radii[k]
    const a = rng.uniform(0, 2 * Math.PI)
    const enclosed = Mc + k * mStar
    const v = Math.sqrt((G * enclosed) / r)
    b.x[i] = cx + r * Math.cos(a)
    b.y[i] = cy + r * Math.sin(a)
    b.vx[i] = vcx - sense * v * Math.sin(a)
    b.vy[i] = vcy + sense * v * Math.cos(a)
    b.m[i] = mStar
  }
}

export function galaxies(nPerGalaxy = 1000, seed = 7): Preset {
  const rng = new Rng(seed)
  const n = 2 * (nPerGalaxy + 1)
  const b = createBodies(n)
  const G = 1
  const mStar = 0.1 / nPerGalaxy
  addDisk(b, 0, nPerGalaxy, -2.0, -0.6, 0.26, 0.05, 1, mStar, 0.15, 1.0, 1, rng, G)
  addDisk(b, nPerGalaxy + 1, nPerGalaxy, 2.0, 0.6, -0.26, -0.05, 1, mStar, 0.15, 0.9, -1, rng, G)
  toComFrame(b)
  return { id: 'galaxies', name: PRESET_NAMES.galaxies, bodies: b, params: { G, eps: 0.04 }, dt: 0.004, view: 3.4, heavy: [0, nPerGalaxy + 1] }
}

export function solar(asteroids = 600, seed = 3): Preset {
  const rng = new Rng(seed)
  const planets: [number, number][] = [
    [0.39, 1.7e-7], [0.72, 2.4e-6], [1.0, 3.0e-6], [1.52, 3.2e-7], [5.2, 9.5e-4], [9.5, 2.9e-4],
  ]
  const n = 1 + planets.length + asteroids
  const b = createBodies(n)
  const G = 4 * Math.PI * Math.PI // AU, years, solar masses
  b.m[0] = 1
  planets.forEach(([r, m], k) => {
    const i = k + 1
    const a = rng.uniform(0, 2 * Math.PI)
    const v = Math.sqrt(G / r)
    b.x[i] = r * Math.cos(a)
    b.y[i] = r * Math.sin(a)
    b.vx[i] = -v * Math.sin(a)
    b.vy[i] = v * Math.cos(a)
    b.m[i] = m
  })
  for (let k = 0; k < asteroids; k++) {
    const i = 1 + planets.length + k
    const r = rng.uniform(2.1, 3.3)
    const a = rng.uniform(0, 2 * Math.PI)
    const v = Math.sqrt(G / r) * (1 + 0.02 * rng.normal())
    b.x[i] = r * Math.cos(a)
    b.y[i] = r * Math.sin(a)
    b.vx[i] = -v * Math.sin(a)
    b.vy[i] = v * Math.cos(a)
    b.m[i] = 1e-12
  }
  toComFrame(b)
  return { id: 'solar', name: PRESET_NAMES.solar, bodies: b, params: { G, eps: 1e-3 }, dt: 0.002, view: 6, heavy: [0, 1, 2, 3, 4, 5, 6] }
}

export function collapse(n = 1500, seed = 11): Preset {
  const rng = new Rng(seed)
  const b = createBodies(n)
  for (let i = 0; i < n; i++) {
    const r = Math.sqrt(rng.next())
    const a = rng.uniform(0, 2 * Math.PI)
    b.x[i] = r * Math.cos(a)
    b.y[i] = r * Math.sin(a)
    // A little solid-body rotation keeps the collapse from being perfectly radial.
    b.vx[i] = -0.15 * b.y[i]
    b.vy[i] = 0.15 * b.x[i]
    b.m[i] = 1 / n
  }
  toComFrame(b)
  return { id: 'collapse', name: PRESET_NAMES.collapse, bodies: b, params: { G: 1, eps: 0.03 }, dt: 0.002, view: 1.6, heavy: [] }
}

export function makePreset(id: PresetId, n?: number): Preset {
  switch (id) {
    case 'galaxies':
      return galaxies(n ? Math.max(10, Math.floor(n / 2) - 1) : undefined)
    case 'figure8':
      return figure8()
    case 'solar':
      return solar(n ? Math.max(0, n - 7) : undefined)
    case 'collapse':
      return collapse(n)
  }
}

/** Random uniform-disk cloud used by the benchmark and tests. */
export function randomCloud(n: number, seed = 1): Bodies {
  return collapse(n, seed).bodies
}
