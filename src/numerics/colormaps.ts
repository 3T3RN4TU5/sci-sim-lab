/**
 * Perceptual colormaps as 256-entry RGB lookup tables, built by linear interpolation
 * between published control points: viridis and inferno (van der Walt & Smith, matplotlib)
 * and Moreland's diverging cool–warm.
 */

export type ColormapId = 'viridis' | 'inferno' | 'coolwarm'

const STOPS: Record<ColormapId, string[]> = {
  viridis: ['440154', '482878', '3e4989', '31688e', '26828e', '1f9e89', '35b779', '6ece58', 'b5de2b', 'fde725'],
  inferno: ['000004', '1b0c41', '4a0c6b', '781c6d', 'a52c60', 'cf4446', 'ed6925', 'fb9b06', 'f7d13d', 'fcffa4'],
  coolwarm: ['3b4cc0', '5977e3', '7b9ff9', '9ebeff', 'c0d4f5', 'dddcdc', 'f2cbb7', 'f7ac8e', 'ee8468', 'd65244', 'b40426'],
}

const cache = new Map<ColormapId, Uint8Array>()

/** 256×3 RGB lookup table. */
export function colormapLUT(id: ColormapId): Uint8Array {
  let lut = cache.get(id)
  if (lut) return lut
  const stops = STOPS[id].map((h) => [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)])
  lut = new Uint8Array(256 * 3)
  const segs = stops.length - 1
  for (let i = 0; i < 256; i++) {
    const x = (i / 255) * segs
    const k = Math.min(segs - 1, Math.floor(x))
    const f = x - k
    for (let c = 0; c < 3; c++) lut[i * 3 + c] = Math.round(stops[k][c] * (1 - f) + stops[k + 1][c] * f)
  }
  cache.set(id, lut)
  return lut
}

/** CSS colour for a value in [0, 1]. */
export function colormapCss(id: ColormapId, t: number): string {
  const lut = colormapLUT(id)
  const i = Math.max(0, Math.min(255, Math.round(t * 255))) * 3
  return `rgb(${lut[i]} ${lut[i + 1]} ${lut[i + 2]})`
}

/**
 * Write a scalar field into RGBA pixels. Values are mapped linearly from [lo, hi]
 * onto the colormap (clamped).
 */
export function fieldToRgba(field: ArrayLike<number>, out: Uint8ClampedArray, id: ColormapId, lo: number, hi: number): void {
  const lut = colormapLUT(id)
  const scale = 255 / (hi - lo || 1)
  for (let k = 0, p = 0; k < field.length; k++, p += 4) {
    let i = ((field[k] - lo) * scale) | 0
    i = i < 0 ? 0 : i > 255 ? 255 : i
    out[p] = lut[i * 3]
    out[p + 1] = lut[i * 3 + 1]
    out[p + 2] = lut[i * 3 + 2]
    out[p + 3] = 255
  }
}
