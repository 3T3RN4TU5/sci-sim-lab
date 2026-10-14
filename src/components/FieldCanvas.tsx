import { forwardRef, useImperativeHandle, useRef, type PointerEvent } from 'react'
import { fieldToRgba, type ColormapId } from '../numerics/colormaps.ts'

export interface FieldCanvasHandle {
  /** Paint an n×n scalar field through a colormap; `mask` cells (≠ 0) are drawn in slate. */
  render: (field: ArrayLike<number>, cmap: ColormapId, lo: number, hi: number, mask?: Uint8Array) => void
  /** Overlay drawing on the display canvas in grid coordinates (called after render). */
  overlay: (fn: (ctx: CanvasRenderingContext2D, cell: number) => void) => void
}

/**
 * Square canvas that displays an n×n grid via an offscreen n×n ImageData scaled up with
 * smoothing. Pointer events report fractional grid coordinates.
 */
export const FieldCanvas = forwardRef<
  FieldCanvasHandle,
  { n: number; size: number; label: string; onPointer?: (gi: number, gj: number, e: PointerEvent<HTMLCanvasElement>, phase: 'down' | 'move' | 'up') => void }
>(function FieldCanvas({ n, size, label, onPointer }, ref) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const off = useRef<{ canvas: HTMLCanvasElement; img: ImageData } | null>(null)
  const down = useRef(false)

  useImperativeHandle(ref, () => ({
    render(field, cmap, lo, hi, mask) {
      const canvas = canvasRef.current
      if (!canvas) return
      if (!off.current || off.current.img.width !== n) {
        const c = document.createElement('canvas')
        c.width = n
        c.height = n
        off.current = { canvas: c, img: new ImageData(n, n) }
      }
      const { img, canvas: oc } = off.current
      fieldToRgba(field, img.data, cmap, lo, hi)
      if (mask) {
        for (let k = 0; k < mask.length; k++) {
          if (mask[k]) {
            img.data[k * 4] = 100
            img.data[k * 4 + 1] = 116
            img.data[k * 4 + 2] = 139
          }
        }
      }
      oc.getContext('2d')!.putImageData(img, 0, 0)
      const dpr = Math.min(2, window.devicePixelRatio || 1)
      const px = Math.round(size * dpr)
      if (canvas.width !== px) {
        canvas.width = px
        canvas.height = px
      }
      canvas.style.width = `${size}px`
      canvas.style.height = `${size}px`
      const ctx = canvas.getContext('2d')!
      ctx.setTransform(1, 0, 0, 1, 0, 0)
      ctx.imageSmoothingEnabled = true
      ctx.imageSmoothingQuality = 'high'
      ctx.drawImage(oc, 0, 0, px, px)
    },
    overlay(fn) {
      const canvas = canvasRef.current
      if (!canvas) return
      const ctx = canvas.getContext('2d')!
      const dpr = canvas.width / size
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      fn(ctx, size / n)
    },
  }))

  const coords = (e: PointerEvent<HTMLCanvasElement>): [number, number] => {
    const r = e.currentTarget.getBoundingClientRect()
    return [((e.clientX - r.left) / r.width) * n, ((e.clientY - r.top) / r.height) * n]
  }

  return (
    <canvas
      ref={canvasRef}
      width={size}
      height={size}
      style={{ width: size, height: size }}
      className="sim-canvas block cursor-crosshair rounded-xl"
      aria-label={label}
      onPointerDown={(e) => {
        e.currentTarget.setPointerCapture(e.pointerId)
        down.current = true
        onPointer?.(...coords(e), e, 'down')
      }}
      onPointerMove={(e) => down.current && onPointer?.(...coords(e), e, 'move')}
      onPointerUp={(e) => {
        down.current = false
        onPointer?.(...coords(e), e, 'up')
      }}
    />
  )
})
