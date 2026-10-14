/** Minimal typing for a dedicated worker's global scope (the app's tsconfig uses the DOM lib). */
export interface WorkerScope<In> {
  onmessage: ((e: MessageEvent<In>) => void) | null
  postMessage: (message: unknown) => void
}

export function workerScope<In>(): WorkerScope<In> {
  return self as unknown as WorkerScope<In>
}
