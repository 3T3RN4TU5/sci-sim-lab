import { useId, type ReactNode } from 'react'

export function cn(...parts: (string | false | null | undefined)[]): string {
  return parts.filter(Boolean).join(' ')
}

export function Slider({
  label,
  value,
  min,
  max,
  step,
  onChange,
  format,
  disabled,
  hint,
}: {
  label: ReactNode
  value: number
  min: number
  max: number
  step: number
  onChange: (v: number) => void
  format?: (v: number) => string
  disabled?: boolean
  hint?: ReactNode
}) {
  const id = useId()
  const fill = ((value - min) / (max - min)) * 100
  return (
    <div className="space-y-1">
      <div className="flex items-baseline justify-between gap-2 text-sm">
        <label htmlFor={id} className="text-slate-300">
          {label}
        </label>
        <span className="font-mono text-xs text-slate-100 tabular-nums">{format ? format(value) : value}</span>
      </div>
      <input
        id={id}
        type="range"
        className="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        style={{ ['--fill' as string]: `${Math.max(0, Math.min(100, fill))}%` }}
        onChange={(e) => onChange(Number(e.target.value))}
      />
      {hint && <p className="text-xs text-slate-500">{hint}</p>}
    </div>
  )
}

export function Select<T extends string>({ label, value, options, onChange }: { label: ReactNode; value: T; options: readonly { value: T; label: string }[]; onChange: (v: T) => void }) {
  const id = useId()
  return (
    <div className="space-y-1">
      <label htmlFor={id} className="text-sm text-slate-300">
        {label}
      </label>
      <select id={id} className="select" value={value} onChange={(e) => onChange(e.target.value as T)}>
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    </div>
  )
}

export function Check({ label, checked, onChange, color }: { label: ReactNode; checked: boolean; onChange: (v: boolean) => void; color?: string }) {
  return (
    <label className="flex cursor-pointer items-center gap-2 text-sm text-slate-300 select-none">
      <input type="checkbox" className="size-4 accent-sky-400" checked={checked} onChange={(e) => onChange(e.target.checked)} />
      {color && <span className="inline-block h-0.5 w-4 rounded" style={{ background: color }} aria-hidden />}
      {label}
    </label>
  )
}

export function Segmented<T extends string>({ label, value, options, onChange }: { label: string; value: T; options: readonly { value: T; label: string }[]; onChange: (v: T) => void }) {
  return (
    <div role="radiogroup" aria-label={label} className="flex flex-wrap gap-1 rounded-xl border border-white/10 bg-slate-900/80 p-1">
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          role="radio"
          aria-checked={o.value === value}
          onClick={() => onChange(o.value)}
          className={cn(
            'flex-1 rounded-lg px-3 py-1.5 text-sm font-medium whitespace-nowrap transition',
            o.value === value ? 'bg-sky-500/20 text-white ring-1 ring-sky-400/40' : 'text-slate-400 hover:text-white',
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  )
}

export function Readout({ label, value, tone }: { label: ReactNode; value: ReactNode; tone?: 'good' | 'bad' | 'warn' }) {
  return (
    <div className="rounded-xl border border-white/[0.06] bg-slate-950/50 px-3 py-2">
      <div className="text-[11px] tracking-wide text-slate-400 uppercase">{label}</div>
      <div className={cn('font-mono text-sm tabular-nums', tone === 'good' ? 'text-emerald-300' : tone === 'bad' ? 'text-rose-300' : tone === 'warn' ? 'text-amber-300' : 'text-slate-100')}>
        {value}
      </div>
    </div>
  )
}

export function Panel({ title, children, className, actions }: { title?: ReactNode; children: ReactNode; className?: string; actions?: ReactNode }) {
  return (
    <section className={cn('card p-4', className)}>
      {(title || actions) && (
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
          {title && <h3 className="text-sm font-semibold text-white">{title}</h3>}
          {actions}
        </div>
      )}
      {children}
    </section>
  )
}

export function PageHeader({ eyebrow, title, children }: { eyebrow: string; title: string; children: ReactNode }) {
  return (
    <header className="mb-6">
      <div className="eyebrow text-sky-300/80">{eyebrow}</div>
      <h1 className="mt-1 text-2xl font-bold tracking-tight text-white sm:text-3xl">{title}</h1>
      <p className="mt-2 max-w-3xl text-sm leading-relaxed text-slate-400 sm:text-base">{children}</p>
    </header>
  )
}

export function MethodSection({ children, references }: { children: ReactNode; references: { text: string; href?: string }[] }) {
  return (
    <section className="card mt-6 p-5" aria-labelledby="method-heading">
      <h2 id="method-heading" className="text-lg font-semibold text-white">
        Method
      </h2>
      <div className="prose-method mt-2 text-sm">{children}</div>
      <h3 className="mt-4 text-sm font-semibold text-slate-200">References</h3>
      <ol className="mt-1 list-decimal space-y-1 pl-5 text-xs text-slate-400">
        {references.map((r) => (
          <li key={r.text}>
            {r.href ? (
              <a className="underline decoration-slate-600 underline-offset-2 hover:text-slate-200" href={r.href} target="_blank" rel="noreferrer">
                {r.text}
              </a>
            ) : (
              r.text
            )}
          </li>
        ))}
      </ol>
    </section>
  )
}
