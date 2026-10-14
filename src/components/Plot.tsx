import { useEffect, useRef, useState, type PointerEvent } from 'react'
import { fitCanvas, useElementWidth } from '../lib/hooks.ts'
import { fmt } from '../lib/format.ts'
import { C } from '../lib/palette.ts'

export interface Series {
  x: ArrayLike<number>
  y: ArrayLike<number>
  color: string
  label?: string
  width?: number
  dash?: number[]
  /** Draw markers (default false). */
  points?: boolean
  /** Draw the connecting line (default true). */
  line?: boolean
  radius?: number
}

export interface PlotProps {
  series: Series[]
  height?: number
  xLog?: boolean
  yLog?: boolean
  xLabel?: string
  yLabel?: string
  xDomain?: [number, number]
  yDomain?: [number, number]
  /** Equal data units per pixel on both axes (for orbits and phase portraits). */
  equal?: boolean
  vLines?: { x: number; color?: string; label?: string }[]
  hLines?: { y: number; color?: string; label?: string }[]
  legend?: boolean
  ariaLabel: string
}

const M = { l: 58, r: 14, t: 10, b: 38 }

type Scale = { toPx: (v: number) => number; fromPx: (p: number) => number; lo: number; hi: number; log: boolean }

function makeScale(lo: number, hi: number, p0: number, p1: number, log: boolean): Scale {
  const f = log ? Math.log10 : (v: number) => v
  const a = f(lo)
  const b = f(hi)
  const k = (p1 - p0) / (b - a || 1)
  return {
    lo,
    hi,
    log,
    toPx: (v) => p0 + (f(v) - a) * k,
    fromPx: (p) => {
      const u = a + (p - p0) / k
      return log ? 10 ** u : u
    },
  }
}

function niceTicks(lo: number, hi: number, count: number): number[] {
  const span = hi - lo
  if (!(span > 0)) return [lo]
  const raw = span / count
  const mag = 10 ** Math.floor(Math.log10(raw))
  const norm = raw / mag
  const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag
  const ticks: number[] = []
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) ticks.push(Math.abs(v) < step * 1e-9 ? 0 : v)
  return ticks
}

function logTicks(lo: number, hi: number): number[] {
  const a = Math.floor(Math.log10(lo))
  const b = Math.ceil(Math.log10(hi))
  const every = Math.max(1, Math.ceil((b - a) / 7))
  const ticks: number[] = []
  for (let e = a; e <= b; e += every) {
    const v = 10 ** e
    if (v >= lo * 0.999 && v <= hi * 1.001) ticks.push(v)
  }
  return ticks
}

function extent(series: Series[], axis: 'x' | 'y', log: boolean): [number, number] {
  let lo = Infinity
  let hi = -Infinity
  for (const s of series) {
    const arr = s[axis]
    const other = axis === 'x' ? s.y : s.x
    for (let i = 0; i < arr.length; i++) {
      const v = arr[i]
      if (!Number.isFinite(v) || (log && v <= 0) || !Number.isFinite(other[i])) continue
      if (v < lo) lo = v
      if (v > hi) hi = v
    }
  }
  if (!Number.isFinite(lo)) return log ? [1, 10] : [0, 1]
  if (lo === hi) return log ? [lo / 2, hi * 2] : [lo - 1, hi + 1]
  if (log) return [lo / 1.3, hi * 1.3]
  const pad = (hi - lo) * 0.05
  return [lo - pad, hi + pad]
}

function tickLabel(v: number, log: boolean): string {
  if (log) {
    const e = Math.round(Math.log10(v))
    return e === 0 ? '1' : e === 1 ? '10' : `1e${e}`.replace('-', '−')
  }
  return fmt(v, 3)
}

/** Canvas line/scatter chart with linear or log axes, legend and a hover read-out. */
export function Plot({ series, height = 240, xLog = false, yLog = false, xLabel, yLabel, xDomain, yDomain, equal, vLines, hLines, legend, ariaLabel }: PlotProps) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const width = useElementWidth(wrapRef)
  const scales = useRef<{ x: Scale; y: Scale } | null>(null)
  const [hover, setHover] = useState<{ px: number; py: number; label: string; color: string; x: number; y: number } | null>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || width < 50) return
    const ctx = fitCanvas(canvas, width, height)
    if (!ctx) return
    let [x0, x1] = xDomain ?? extent(series, 'x', xLog)
    let [y0, y1] = yDomain ?? extent(series, 'y', yLog)
    const pw = width - M.l - M.r
    const ph = height - M.t - M.b
    if (equal && !xLog && !yLog) {
      const ux = (x1 - x0) / pw
      const uy = (y1 - y0) / ph
      if (ux > uy) {
        const c = (y0 + y1) / 2
        y0 = c - (ux * ph) / 2
        y1 = c + (ux * ph) / 2
      } else {
        const c = (x0 + x1) / 2
        x0 = c - (uy * pw) / 2
        x1 = c + (uy * pw) / 2
      }
    }
    const sx = makeScale(x0, x1, M.l, M.l + pw, xLog)
    const sy = makeScale(y0, y1, M.t + ph, M.t, yLog)
    scales.current = { x: sx, y: sy }

    ctx.clearRect(0, 0, width, height)
    ctx.font = '11px Inter, system-ui, sans-serif'
    // Grid and tick labels.
    const xt = xLog ? logTicks(x0, x1) : niceTicks(x0, x1, Math.max(3, Math.floor(pw / 90)))
    const yt = yLog ? logTicks(y0, y1) : niceTicks(y0, y1, Math.max(3, Math.floor(ph / 45)))
    ctx.strokeStyle = C.grid
    ctx.lineWidth = 1
    ctx.fillStyle = C.muted
    ctx.textAlign = 'center'
    ctx.textBaseline = 'top'
    for (const v of xt) {
      const p = Math.round(sx.toPx(v)) + 0.5
      ctx.beginPath()
      ctx.moveTo(p, M.t)
      ctx.lineTo(p, M.t + ph)
      ctx.stroke()
      ctx.fillText(tickLabel(v, xLog), p, M.t + ph + 5)
    }
    ctx.textAlign = 'right'
    ctx.textBaseline = 'middle'
    for (const v of yt) {
      const p = Math.round(sy.toPx(v)) + 0.5
      ctx.beginPath()
      ctx.moveTo(M.l, p)
      ctx.lineTo(M.l + pw, p)
      ctx.stroke()
      ctx.fillText(tickLabel(v, yLog), M.l - 6, p)
    }
    ctx.strokeStyle = 'rgba(148,163,184,0.35)'
    ctx.strokeRect(M.l + 0.5, M.t + 0.5, pw, ph)
    // Axis titles.
    ctx.fillStyle = '#cbd5e1'
    ctx.textAlign = 'center'
    ctx.textBaseline = 'bottom'
    if (xLabel) ctx.fillText(xLabel, M.l + pw / 2, height - 2)
    if (yLabel) {
      ctx.save()
      ctx.translate(12, M.t + ph / 2)
      ctx.rotate(-Math.PI / 2)
      ctx.textBaseline = 'middle'
      ctx.fillText(yLabel, 0, 0)
      ctx.restore()
    }

    ctx.save()
    ctx.beginPath()
    ctx.rect(M.l, M.t, pw, ph)
    ctx.clip()
    const refLine = (p0x: number, p0y: number, p1x: number, p1y: number, color: string, label?: string) => {
      ctx.strokeStyle = color
      ctx.setLineDash([4, 4])
      ctx.lineWidth = 1
      ctx.beginPath()
      ctx.moveTo(p0x, p0y)
      ctx.lineTo(p1x, p1y)
      ctx.stroke()
      ctx.setLineDash([])
      if (label) {
        ctx.fillStyle = '#cbd5e1'
        ctx.textAlign = 'left'
        ctx.textBaseline = 'top'
        ctx.fillText(label, p0x + 4, Math.min(p0y, p1y) + 3)
      }
    }
    for (const v of vLines ?? []) {
      const p = sx.toPx(v.x)
      refLine(p, M.t + ph, p, M.t, v.color ?? C.muted, v.label)
    }
    for (const h of hLines ?? []) {
      const p = sy.toPx(h.y)
      refLine(M.l, p, M.l + pw, p, h.color ?? C.muted, h.label)
    }
    for (const s of series) {
      const n = Math.min(s.x.length, s.y.length)
      const ok = (i: number) => Number.isFinite(s.x[i]) && Number.isFinite(s.y[i]) && (!xLog || s.x[i] > 0) && (!yLog || s.y[i] > 0)
      if (s.line !== false) {
        ctx.strokeStyle = s.color
        ctx.lineWidth = s.width ?? 2
        ctx.lineJoin = 'round'
        ctx.setLineDash(s.dash ?? [])
        ctx.beginPath()
        let pen = false
        for (let i = 0; i < n; i++) {
          if (!ok(i)) {
            pen = false
            continue
          }
          // Clamp to a generous range so blow-ups still draw sanely.
          const px = Math.max(-1e4, Math.min(1e4, sx.toPx(s.x[i])))
          const py = Math.max(-1e4, Math.min(1e4, sy.toPx(s.y[i])))
          if (pen) ctx.lineTo(px, py)
          else ctx.moveTo(px, py)
          pen = true
        }
        ctx.stroke()
        ctx.setLineDash([])
      }
      if (s.points) {
        ctx.fillStyle = s.color
        ctx.strokeStyle = '#0f172a'
        ctx.lineWidth = 1.5
        const r = s.radius ?? 3.5
        for (let i = 0; i < n; i++) {
          if (!ok(i)) continue
          ctx.beginPath()
          ctx.arc(sx.toPx(s.x[i]), sy.toPx(s.y[i]), r, 0, 2 * Math.PI)
          ctx.fill()
          if (r >= 3) ctx.stroke()
        }
      }
    }
    ctx.restore()
  }, [series, width, height, xLog, yLog, xLabel, yLabel, xDomain, yDomain, equal, vLines, hLines])

  const onMove = (e: PointerEvent<HTMLCanvasElement>) => {
    const sc = scales.current
    if (!sc) return
    const rect = e.currentTarget.getBoundingClientRect()
    const mx = e.clientX - rect.left
    const my = e.clientY - rect.top
    let best = Infinity
    let hit: typeof hover = null
    for (const s of series) {
      const n = Math.min(s.x.length, s.y.length)
      const stride = Math.max(1, Math.floor(n / 4000))
      for (let i = 0; i < n; i += stride) {
        const x = s.x[i]
        const y = s.y[i]
        if (!Number.isFinite(x) || !Number.isFinite(y) || (sc.x.log && x <= 0) || (sc.y.log && y <= 0)) continue
        const px = sc.x.toPx(x)
        const py = sc.y.toPx(y)
        const d = (px - mx) ** 2 + (py - my) ** 2
        if (d < best) {
          best = d
          hit = { px, py, label: s.label ?? '', color: s.color, x, y }
        }
      }
    }
    setHover(best < 40 * 40 ? hit : null)
  }

  const showLegend = legend ?? series.filter((s) => s.label).length >= 2
  return (
    <figure className="m-0" aria-label={ariaLabel}>
      {showLegend && (
        <figcaption className="mb-1 flex flex-wrap gap-x-4 gap-y-1 text-xs text-slate-300">
          {series
            .filter((s) => s.label)
            .map((s) => (
              <span key={s.label} className="inline-flex items-center gap-1.5">
                <span className="inline-block h-0.5 w-4 rounded" style={{ background: s.color, borderTop: s.dash ? `2px dashed ${s.color}` : undefined, height: s.dash ? 0 : undefined }} aria-hidden />
                {s.label}
              </span>
            ))}
        </figcaption>
      )}
      <div ref={wrapRef} className="relative w-full" style={{ height }}>
        <canvas ref={canvasRef} role="img" aria-label={ariaLabel} className="block" onPointerMove={onMove} onPointerLeave={() => setHover(null)} />
        {hover && (
          <>
            <span className="pointer-events-none absolute size-3 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-slate-950" style={{ left: hover.px, top: hover.py, background: hover.color }} />
            <div
              className="pointer-events-none absolute z-10 rounded-lg border border-white/10 bg-slate-950/95 px-2 py-1 font-mono text-[11px] whitespace-nowrap text-slate-200 shadow-lg"
              style={{ left: Math.min(hover.px + 10, width - 150), top: Math.max(0, hover.py - 34) }}
            >
              {hover.label && <div className="font-sans text-slate-400">{hover.label}</div>}
              {xLabel ?? 'x'} = {fmt(hover.x, 4)}, {yLabel ?? 'y'} = {fmt(hover.y, 4)}
            </div>
          </>
        )}
      </div>
    </figure>
  )
}
