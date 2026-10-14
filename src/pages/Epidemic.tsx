import { useMemo, useState } from 'react'
import { Plot, type Series } from '../components/Plot.tsx'
import { Tex } from '../components/Tex.tsx'
import { MethodSection, PageHeader, Panel, Readout, Segmented, Slider } from '../components/ui.tsx'
import { fmt, fmtFixed } from '../lib/format.ts'
import { useAnimationLoop } from '../lib/hooks.ts'
import { C } from '../lib/palette.ts'
import { basicReproductionNumber, finalSize, fitEpidemic, simulateEpidemic, syntheticData, type EpiModel, type FitResult } from '../numerics/epidemic.ts'

const POP = 1_000_000
const KNOWN = { N: POP, I0: 10, sigma: 1 / 5.2 }

export default function Epidemic() {
  const [model, setModel] = useState<EpiModel>('SEIR')
  const [beta, setBeta] = useState(0.4)
  const [gamma, setGamma] = useState(0.125)
  const [latent, setLatent] = useState(5.2)
  const [I0, setI0] = useState(10)
  const [days, setDays] = useState(220)

  const params = { beta, gamma, sigma: 1 / latent, N: POP, I0 }
  const series = useMemo(() => simulateEpidemic(model, { beta, gamma, sigma: 1 / latent, N: POP, I0 }, days), [model, beta, gamma, latent, I0, days])
  const R0 = basicReproductionNumber(params)
  let peakDay = 0
  for (let d = 0; d < series.I.length; d++) if (series.I[d] > series.I[peakDay]) peakDay = d
  const t = Array.from(series.t)
  const pct = (a: Float64Array) => Array.from(a, (v) => (100 * v) / POP)
  const curves: Series[] = [
    { x: t, y: pct(series.S), color: C.blue, label: 'S susceptible' },
    ...(model === 'SEIR' ? [{ x: t, y: pct(series.E), color: C.yellow, label: 'E exposed' }] : []),
    { x: t, y: pct(series.I), color: C.red, label: 'I infectious' },
    { x: t, y: pct(series.R), color: C.aqua, label: 'R recovered' },
  ]

  return (
    <div>
      <PageHeader eyebrow="Module 7 · Mathematical epidemiology" title="SIR / SEIR models and parameter fitting">
        Compartmental models integrated with RK4, the classic threshold quantities, and an inverse problem: recover transmission and recovery rates from noisy prevalence
        data with a from-scratch Nelder–Mead optimiser.
      </PageHeader>
      <div className="grid gap-4 lg:grid-cols-[280px_minmax(0,1fr)]">
        <Panel title="Model">
          <div className="space-y-4">
            <Segmented
              label="Model"
              value={model}
              onChange={setModel}
              options={[
                { value: 'SIR', label: 'SIR' },
                { value: 'SEIR', label: 'SEIR' },
              ]}
            />
            <Slider label="Transmission rate β (1/day)" value={beta} min={0.05} max={1.2} step={0.005} onChange={setBeta} format={(v) => v.toFixed(3)} />
            <Slider label="Infectious period 1/γ (days)" value={1 / gamma} min={1} max={21} step={0.5} onChange={(v) => setGamma(1 / v)} format={(v) => v.toFixed(1)} />
            {model === 'SEIR' && <Slider label="Latent period 1/σ (days)" value={latent} min={0.5} max={14} step={0.1} onChange={setLatent} format={(v) => v.toFixed(1)} />}
            <Slider label="Initial infectious I₀" value={I0} min={1} max={1000} step={1} onChange={setI0} />
            <Slider label="Days" value={days} min={30} max={720} step={10} onChange={setDays} />
            <p className="text-xs text-slate-400">Population N = 1,000,000. Curves show percent of N.</p>
          </div>
        </Panel>
        <div className="min-w-0 space-y-4">
          <Panel>
            <div className="mb-3 grid grid-cols-2 gap-2 sm:grid-cols-5">
              <Readout label="R₀ = β/γ" value={fmtFixed(R0, 3)} tone={R0 > 1 ? 'bad' : 'good'} />
              <Readout label="Peak infectious" value={`${fmtFixed((100 * series.I[peakDay]) / POP, 2)}%`} />
              <Readout label="Peak day" value={String(peakDay)} />
              <Readout label="Final size (theory)" value={`${fmtFixed(100 * finalSize(R0), 2)}%`} />
              <Readout label="Herd immunity 1 − 1/R₀" value={R0 > 1 ? `${fmtFixed(100 * (1 - 1 / R0), 1)}%` : '—'} />
            </div>
            <Plot ariaLabel="Epidemic curves" series={curves} height={300} xLabel="day" yLabel="% of population" yDomain={[0, 100]} />
          </Panel>
        </div>
      </div>
      <FitPanel model={model} />
      <MethodSection
        references={[
          { text: 'W. O. Kermack, A. G. McKendrick — A contribution to the mathematical theory of epidemics, Proc. R. Soc. A 115 (1927) 700–721.' },
          { text: 'J. A. Nelder, R. Mead — A simplex method for function minimization, Comput. J. 7 (1965) 308–313.' },
          { text: 'J. C. Lagarias, J. A. Reeds, M. H. Wright, P. E. Wright — Convergence properties of the Nelder–Mead simplex method in low dimensions, SIAM J. Optim. 9 (1998) 112–147.' },
          { text: 'O. Diekmann, H. Heesterbeek, T. Britton — Mathematical Tools for Understanding Infectious Disease Dynamics, Princeton (2013).' },
        ]}
      >
        <p>The SEIR model (SIR is the limit σ → ∞):</p>
        <Tex block>{String.raw`\dot S = -\beta\frac{SI}{N},\quad \dot E = \beta\frac{SI}{N} - \sigma E,\quad \dot I = \sigma E - \gamma I,\quad \dot R = \gamma I`}</Tex>
        <p>
          Linearising around the disease-free state gives <Tex>{String.raw`R_0 = \beta/\gamma`}</Tex>; the epidemic grows iff <Tex>{String.raw`R_0 > 1`}</Tex>, the herd-immunity
          threshold is <Tex>{String.raw`1 - 1/R_0`}</Tex>, and the final attack rate <Tex>z</Tex> solves <Tex>{String.raw`z = 1 - e^{-R_0 z}`}</Tex> (solved with Newton; the test suite
          checks the simulated SIR final size against it).
        </p>
        <p>
          <strong>Fitting.</strong> Synthetic observations <Tex>{String.raw`I^{\text{obs}}_d = I(d)\,e^{\varepsilon_d}`}</Tex>, <Tex>{String.raw`\varepsilon_d \sim \mathcal N(0, s^2)`}</Tex>{' '}
          (multiplicative noise). We minimise the log-scale least-squares objective over <Tex>{String.raw`\theta = (\ln\beta, \ln\gamma)`}</Tex> (positivity for free):
        </p>
        <Tex block>{String.raw`J(\theta) = \sum_d \big[\ln(1 + I^{\text{obs}}_d) - \ln(1 + I_d(\theta))\big]^2`}</Tex>
        <p>
          Nelder–Mead keeps a simplex of <Tex>n + 1</Tex> points and replaces the worst by reflection (α = 1), expansion (γ = 2), or contraction (ρ = ½), shrinking (σ = ½) when all
          fail. It needs no gradients — each evaluation is a full ODE solve. The tests recover β and γ to within 3% from data with 10% noise and minimise Rosenbrock’s function
          to 10⁻⁶.
        </p>
      </MethodSection>
    </div>
  )
}

function FitPanel({ model }: { model: EpiModel }) {
  const [trueBeta, setTrueBeta] = useState(0.45)
  const [trueGamma, setTrueGamma] = useState(0.15)
  const [noise, setNoise] = useState(0.15)
  const [seed, setSeed] = useState(7)
  const days = 120
  const data = useMemo(() => syntheticData(model, { ...KNOWN, beta: trueBeta, gamma: trueGamma }, days, noise, seed), [model, trueBeta, trueGamma, noise, seed])
  const [fit, setFit] = useState<FitResult | null>(null)
  const [frame, setFrame] = useState(0)
  const [animating, setAnimating] = useState(false)

  const runFit = () => {
    const f = fitEpidemic(model, data, KNOWN, { beta: 0.2, gamma: 0.3 })
    setFit(f)
    setFrame(0)
    setAnimating(true)
  }
  useAnimationLoop(() => {
    if (!fit) return
    setFrame((f) => {
      if (f + 1 >= fit.history.length) {
        setAnimating(false)
        return fit.history.length - 1
      }
      return f + 1
    })
  }, animating)

  const cur = fit ? fit.history[Math.min(frame, fit.history.length - 1)] : null
  const curBeta = cur ? Math.exp(cur.x[0]) : NaN
  const curGamma = cur ? Math.exp(cur.x[1]) : NaN
  const curve = useMemo(() => (cur ? simulateEpidemic(model, { ...KNOWN, beta: Math.exp(cur.x[0]), gamma: Math.exp(cur.x[1]) }, days) : null), [cur, model])
  const truth = useMemo(() => simulateEpidemic(model, { ...KNOWN, beta: trueBeta, gamma: trueGamma }, days), [model, trueBeta, trueGamma])
  const stale = fit && (fit.history.length === 0 || !cur)

  const dataSeries: Series[] = [
    { x: data.t, y: data.I, color: C.muted, points: true, line: false, radius: 2.5, label: 'noisy observations' },
    { x: Array.from(truth.t), y: Array.from(truth.I), color: C.aqua, dash: [5, 4], label: 'true model', width: 1.5 },
    ...(curve ? [{ x: Array.from(curve.t), y: Array.from(curve.I), color: C.orange, label: `fit (iteration ${frame})`, width: 2.5 }] : []),
  ]
  const path: Series[] = fit
    ? [
        { x: fit.history.slice(0, frame + 1).map((h) => Math.exp(h.x[0])), y: fit.history.slice(0, frame + 1).map((h) => Math.exp(h.x[1])), color: C.orange, label: 'simplex best point', points: true, radius: 2.5 },
        { x: [trueBeta], y: [trueGamma], color: C.aqua, points: true, line: false, radius: 6, label: 'true (β, γ)' },
      ]
    : []

  return (
    <Panel
      title={`Inverse problem: fit ${model} to noisy data with Nelder–Mead`}
      className="mt-4"
      actions={
        <button type="button" className="btn btn-primary" onClick={runFit}>
          {fit ? 'Re-fit' : 'Fit with Nelder–Mead'}
        </button>
      }
    >
      <div className="grid gap-4 lg:grid-cols-[260px_minmax(0,1fr)]">
        <div className="space-y-4">
          <Slider label="True β" value={trueBeta} min={0.15} max={0.9} step={0.01} onChange={setTrueBeta} format={(v) => v.toFixed(2)} />
          <Slider label="True γ" value={trueGamma} min={0.05} max={0.4} step={0.005} onChange={setTrueGamma} format={(v) => v.toFixed(3)} />
          <Slider label="Noise level s (log scale)" value={noise} min={0} max={0.6} step={0.01} onChange={setNoise} format={(v) => v.toFixed(2)} />
          <button type="button" className="btn w-full" onClick={() => setSeed((s) => s + 1)}>
            New noise realisation
          </button>
          <div className="grid grid-cols-2 gap-2">
            <Readout label="β fitted" value={fit && !stale ? fmtFixed(curBeta, 4) : '—'} />
            <Readout label="γ fitted" value={fit && !stale ? fmtFixed(curGamma, 4) : '—'} />
            <Readout label="R₀ fitted" value={fit && !stale ? fmtFixed(curBeta / curGamma, 3) : '—'} tone={fit && !animating && Math.abs(curBeta / curGamma - trueBeta / trueGamma) / (trueBeta / trueGamma) < 0.05 ? 'good' : undefined} />
            <Readout label="R₀ true" value={fmtFixed(trueBeta / trueGamma, 3)} />
            <Readout label="Objective J" value={cur ? fmt(cur.fx) : '—'} />
            <Readout label="ODE solves" value={fit ? String(fit.evaluations) : '—'} />
          </div>
        </div>
        <div className="grid min-w-0 gap-4 xl:grid-cols-2">
          <Plot ariaLabel="Data and fitted prevalence" series={dataSeries} yLog height={280} xLabel="day" yLabel="infectious I(t)" yDomain={[1, POP / 2]} />
          {fit ? (
            <Plot ariaLabel="Nelder–Mead path in parameter space" series={path} height={280} xLabel="β" yLabel="γ" xDomain={[0.1, Math.max(1, trueBeta * 1.3)]} yDomain={[0.03, 0.45]} />
          ) : (
            <div className="flex items-center justify-center rounded-xl border border-dashed border-white/10 p-6 text-center text-sm text-slate-400">
              Press “Fit” to start the simplex at (β, γ) = (0.2, 0.3) and watch it walk to the minimum.
            </div>
          )}
          {fit && (
            <div className="xl:col-span-2">
              <Plot
                ariaLabel="Objective vs iteration"
                height={180}
                yLog
                xLabel="iteration"
                yLabel="J"
                series={[{ x: fit.history.slice(0, frame + 1).map((_, i) => i), y: fit.history.slice(0, frame + 1).map((h) => h.fx), color: C.violet, label: 'objective' }]}
                xDomain={[0, fit.history.length]}
              />
            </div>
          )}
        </div>
      </div>
    </Panel>
  )
}
