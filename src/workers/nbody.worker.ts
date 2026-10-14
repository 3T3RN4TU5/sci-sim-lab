import { benchmarkPoint } from '../numerics/benchmark.ts'
import { workerScope } from './scope.ts'

const scope = workerScope<{ ns: number[]; theta: number }>()
scope.onmessage = (e) => {
  for (const n of e.data.ns) scope.postMessage({ type: 'point', point: benchmarkPoint(n, e.data.theta) })
  scope.postMessage({ type: 'done' })
}
