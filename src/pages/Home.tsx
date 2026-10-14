import { ROUTES } from '../config.ts'

const VERIFY: [string, string, string][] = [
  ['Explicit Euler / Midpoint / RK4 — convergence order', '1 / 2 / 4', '1.049 / 2.007 / 4.008'],
  ['Velocity Verlet / Yoshida-4 / Backward Euler', '2 / 4 / 1', '2.000 / 4.000 / 0.954'],
  ['Dormand–Prince RK45 vs SciPy DOP853 (Lorenz, VdP, Kepler)', '< 1e-8', '✓ (75 steps vs SciPy RK45 75)'],
  ['Verlet energy error over 10⁵ steps (pendulum)', 'bounded', '1.6e-3, no growth'],
  ['FFT vs naive DFT, N ≤ 1024 · Parseval', '< 1e-12', '6.7e-15 · 2.6e-15'],
  ['Heat eq. FTCS / ADI vs analytic mode decay · ADI order', '< 1% · 2', '3.4e-4 / 1.5e-4 · 2.00'],
  ['Barnes–Hut θ = 0.5 RMS force error (mono / quad)', '< 2% / < 0.5%', '1.4% / 0.24%'],
  ['Ising T = 2.0 vs Onsager: e, |m| (Wolff, L = 32)', '−1.7456, 0.9113', '−1.7471, 0.9119'],
  ['Lorenz λ₁ (Benettin, t = 1000)', '0.9056', '0.9025'],
  ['Monte Carlo / quasi-MC error slope', '−0.5 / ≈ −1', '−0.508 / −1.006'],
  ['Nelder–Mead SIR fit: β, γ (true 0.45, 0.15)', '< 3%', '0.4488, 0.1503'],
]

export default function Home() {
  return (
    <div>
      <section className="py-6 sm:py-10">
        <div className="eyebrow text-sky-300/80">Scientific computing portfolio · by Jeevan</div>
        <h1 className="mt-2 max-w-4xl text-3xl font-extrabold tracking-tight text-white sm:text-5xl">
          Numerical methods and physical simulations, <span className="bg-linear-to-r from-sky-400 to-violet-400 bg-clip-text text-transparent">verified</span>.
        </h1>
        <p className="mt-4 max-w-3xl text-base leading-relaxed text-slate-400">
          Every solver here — ODE integrators, Barnes–Hut trees, finite-difference PDE schemes, Monte Carlo samplers, the FFT and the optimiser — is written from scratch in
          strict TypeScript on typed arrays, with no numerical libraries. Each one is backed by a test that measures what theory predicts: convergence orders, conservation
          laws, analytic solutions and cross-checks against SciPy.
        </p>
        <div className="mt-6 flex flex-wrap gap-3">
          <a href="#/integrators" className="btn btn-primary px-4 py-2">
            Open the integrator lab
          </a>
          <a href="https://github.com/Jeevan2410/sci-sim-lab" target="_blank" rel="noreferrer" className="btn px-4 py-2">
            Read the code & tests
          </a>
        </div>
      </section>

      <section aria-label="Modules" className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {ROUTES.map((r, i) => (
          <a key={r.id} href={`#/${r.id}`} className="card group block p-5 transition hover:border-sky-400/30 hover:bg-slate-900">
            <div className="eyebrow">Module {i + 1}</div>
            <h2 className="mt-1 text-lg font-semibold text-white group-hover:text-sky-200">{r.label}</h2>
            <p className="mt-2 text-sm leading-relaxed text-slate-400">{r.blurb}</p>
          </a>
        ))}
      </section>

      <section className="card mt-6 p-5" aria-labelledby="verify-heading">
        <h2 id="verify-heading" className="text-lg font-semibold text-white">
          Verification at a glance
        </h2>
        <p className="mt-1 text-sm text-slate-400">
          From <code className="text-slate-300">npm test</code> (70 tests, Node’s built-in runner). Expected values come from theory or independent references.
        </p>
        <div className="mt-3 overflow-x-auto">
          <table className="w-full min-w-[560px] text-sm">
            <thead>
              <tr className="text-left text-xs text-slate-400">
                <th className="py-1.5 font-medium">Check</th>
                <th className="py-1.5 font-medium">Expected</th>
                <th className="py-1.5 font-medium">Measured</th>
              </tr>
            </thead>
            <tbody>
              {VERIFY.map(([c, e, m]) => (
                <tr key={c} className="border-t border-white/5">
                  <td className="py-1.5 text-slate-200">{c}</td>
                  <td className="py-1.5 font-mono text-slate-400">{e}</td>
                  <td className="py-1.5 font-mono text-emerald-300">{m}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  )
}
