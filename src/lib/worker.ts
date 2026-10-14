import { useCallback, useEffect, useRef, useState } from 'react'

/**
 * Run a request/response computation in a Web Worker. Each `run` terminates any
 * in-flight job, so the latest request always wins.
 */
export function useWorkerJob<Req, Res>(factory: () => Worker): { run: (req: Req) => void; result: Res | null; busy: boolean; error: string | null } {
  const workerRef = useRef<Worker | null>(null)
  const factoryRef = useRef(factory)
  const [result, setResult] = useState<Res | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => () => workerRef.current?.terminate(), [])

  const run = (req: Req) => {
    workerRef.current?.terminate()
    const w = factoryRef.current()
    workerRef.current = w
    setBusy(true)
    setError(null)
    w.onmessage = (e: MessageEvent<Res>) => {
      setResult(e.data)
      setBusy(false)
      w.terminate()
      if (workerRef.current === w) workerRef.current = null
    }
    w.onerror = (e) => {
      setError(e.message || 'worker failed')
      setBusy(false)
    }
    w.postMessage(req)
  }
  return { run, result, busy, error }
}

/** A long-lived worker that streams progress messages. */
export function useStreamingWorker<Req, Msg>(factory: () => Worker, onMessage: (m: Msg) => void, initialRequest?: Req): { start: (req: Req) => void; stop: () => void } {
  const workerRef = useRef<Worker | null>(null)
  const factoryRef = useRef(factory)
  const handler = useRef(onMessage)
  const initial = useRef(initialRequest)
  useEffect(() => {
    handler.current = onMessage
  })
  const start = useCallback((req: Req) => {
    workerRef.current?.terminate()
    const w = factoryRef.current()
    workerRef.current = w
    w.onmessage = (e: MessageEvent<Msg>) => handler.current(e.data)
    w.postMessage(req)
  }, [])
  const stop = useCallback(() => {
    workerRef.current?.terminate()
    workerRef.current = null
  }, [])
  useEffect(() => {
    if (initial.current !== undefined) start(initial.current)
    return stop
  }, [start, stop])
  return { start, stop }
}
