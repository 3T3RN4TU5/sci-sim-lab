import { runStiffDemo, type StiffRequest } from '../numerics/stiff.ts'
import { workerScope } from './scope.ts'

const scope = workerScope<StiffRequest>()
scope.onmessage = (e) => scope.postMessage(runStiffDemo(e.data))
