import { lazy, Suspense, useEffect, type ComponentType, type LazyExoticComponent } from 'react'
import { cn } from './lib/utils.ts'
import { REPO_URL, ROUTES } from './config.ts'
import { useHashRoute } from './lib/hooks.ts'
import Home from './pages/Home.tsx'

const PAGES: Record<string, LazyExoticComponent<ComponentType>> = {
  integrators: lazy(() => import('./pages/IntegratorLab.tsx')),
  chaos: lazy(() => import('./pages/Chaos.tsx')),
  nbody: lazy(() => import('./pages/NBody.tsx')),
  pde: lazy(() => import('./pages/Pde.tsx')),
  statphys: lazy(() => import('./pages/StatPhys.tsx')),
  signals: lazy(() => import('./pages/Signals.tsx')),
  epidemic: lazy(() => import('./pages/Epidemic.tsx')),
}

export default function App() {
  const route = useHashRoute()
  const Page = PAGES[route]

  useEffect(() => {
    const info = ROUTES.find((r) => r.id === route)
    document.title = info ? `${info.label} · SciSim Lab` : 'SciSim Lab — Numerical Methods, Verified'
    window.scrollTo(0, 0)
  }, [route])

  return (
    <div className="flex min-h-dvh flex-col">
      <header className="sticky top-0 z-30 border-b border-white/[0.06] bg-slate-950/85 backdrop-blur-md">
        <div className="mx-auto flex max-w-7xl items-center gap-4 px-4 py-2.5 sm:px-6">
          <a href="#/" className="flex shrink-0 items-center gap-2.5 rounded-lg" aria-label="SciSim Lab home">
            <img src="/favicon.svg" alt="" className="size-8" />
            <span className="leading-tight">
              <span className="block text-sm font-bold tracking-tight text-white sm:text-base">SciSim Lab</span>
              <span className="hidden text-[11px] text-slate-400 sm:block">Numerical methods, verified</span>
            </span>
          </a>
          <nav aria-label="Modules" className="-mx-1 flex min-w-0 flex-1 gap-1 overflow-x-auto px-1 py-1">
            {ROUTES.map((r) => (
              <a
                key={r.id}
                href={`#/${r.id}`}
                aria-current={route === r.id ? 'page' : undefined}
                className={cn(
                  'rounded-lg px-3 py-1.5 text-sm font-medium whitespace-nowrap transition',
                  route === r.id ? 'bg-sky-500/15 text-white ring-1 ring-sky-400/30' : 'text-slate-400 hover:bg-white/5 hover:text-white',
                )}
              >
                {r.short}
              </a>
            ))}
          </nav>
          <a href={REPO_URL} target="_blank" rel="noreferrer" className="hidden shrink-0 rounded-lg px-2 py-1.5 text-sm text-slate-400 hover:text-white md:block">
            GitHub
          </a>
        </div>
      </header>
      <main className="mx-auto w-full max-w-7xl flex-1 px-4 py-6 sm:px-6 sm:py-8">
        {Page ? (
          <Suspense fallback={<div className="py-20 text-center text-slate-400">Loading module…</div>}>
            <Page key={route} />
          </Suspense>
        ) : (
          <Home />
        )}
      </main>
      <footer className="border-t border-white/[0.06] py-5 text-center text-xs text-slate-500">
        Built by <a className="text-slate-300 hover:text-white" href="https://github.com/3T3RN4TU5" target="_blank" rel="noreferrer">Jeevan</a> · All numerics written from scratch in TypeScript ·{' '}
        <a className="text-slate-300 hover:text-white" href={REPO_URL} target="_blank" rel="noreferrer">
          Source &amp; verification suite
        </a>
      </footer>
    </div>
  )
}
