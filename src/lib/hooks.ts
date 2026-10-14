import { useEffect, useRef, useState, useSyncExternalStore, type RefObject } from 'react'

function subscribeHash(onChange: () => void) {
  window.addEventListener('hashchange', onChange)
  return () => window.removeEventListener('hashchange', onChange)
}

/** The current `location.hash` route (without `#/`). */
export function useHashRoute(): string {
  return useSyncExternalStore(
    subscribeHash,
    () => window.location.hash.replace(/^#\/?/, ''),
    () => '',
  )
}

/**
 * Run `tick(dt)` on every animation frame while `running` is true. The callback is
 * held in a ref so it always sees fresh state without restarting the loop.
 */
export function useAnimationLoop(tick: (dtMs: number) => void, running: boolean): void {
  const ref = useRef(tick)
  useEffect(() => {
    ref.current = tick
  })
  useEffect(() => {
    if (!running) return
    let raf = 0
    let last = performance.now()
    const loop = (now: number) => {
      const dt = Math.min(100, now - last)
      last = now
      ref.current(dt)
      raf = requestAnimationFrame(loop)
    }
    raf = requestAnimationFrame(loop)
    return () => cancelAnimationFrame(raf)
  }, [running])
}

/** Width of an element, tracked with a ResizeObserver. */
export function useElementWidth(ref: RefObject<HTMLElement | null>): number {
  const [w, setW] = useState(0)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const ro = new ResizeObserver((entries) => setW(Math.floor(entries[0].contentRect.width)))
    ro.observe(el)
    setW(Math.floor(el.getBoundingClientRect().width))
    return () => ro.disconnect()
  }, [ref])
  return w
}

/** Re-render at most every `ms` milliseconds while `active` (for live readouts). */
export function useThrottledTick(ms: number, active: boolean): number {
  const [n, setN] = useState(0)
  useEffect(() => {
    if (!active) return
    const id = window.setInterval(() => setN((v) => v + 1), ms)
    return () => window.clearInterval(id)
  }, [ms, active])
  return n
}

/** Size a canvas's backing store to its CSS size × devicePixelRatio. Returns the context. */
export function fitCanvas(canvas: HTMLCanvasElement, cssW: number, cssH: number): CanvasRenderingContext2D | null {
  const dpr = Math.min(2, window.devicePixelRatio || 1)
  const w = Math.max(1, Math.round(cssW * dpr))
  const h = Math.max(1, Math.round(cssH * dpr))
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w
    canvas.height = h
  }
  canvas.style.width = `${cssW}px`
  canvas.style.height = `${cssH}px`
  const ctx = canvas.getContext('2d')
  ctx?.setTransform(dpr, 0, 0, dpr, 0, 0)
  return ctx
}
