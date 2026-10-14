import { Ising, measure, type IsingAlgorithm } from '../numerics/ising.ts'
import { piErrorScaling } from '../numerics/montecarlo.ts'
import { workerScope } from './scope.ts'

export type StatRequest = { kind: 'sweep'; L: number; algo: IsingAlgorithm; temps: number[]; equil: number; samples: number } | { kind: 'pi' }

const scope = workerScope<StatRequest>()
scope.onmessage = (e) => {
  const req = e.data
  if (req.kind === 'pi') {
    scope.postMessage({ kind: 'pi', result: piErrorScaling(6, 16, 48, 5) })
    return
  }
  // Sweep from high to low temperature, reusing the configuration (annealing).
  const temps = [...req.temps].sort((a, b) => b - a)
  const model = new Ising(req.L, temps[0], 17, 'hot')
  for (const T of temps) {
    model.setTemperature(T)
    scope.postMessage({ kind: 'point', point: measure(model, req.algo, req.equil, req.samples) })
  }
  scope.postMessage({ kind: 'done' })
}
