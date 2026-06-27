export default function Card({ title, subtitle, right, children }) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="mb-3 flex items-start justify-between gap-2">
        <div>
          <h3 className="font-semibold text-slate-800">{title}</h3>
          {subtitle && <p className="text-xs text-slate-400">{subtitle}</p>}
        </div>
        {right}
      </div>
      {children}
    </div>
  )
}

export function Skeleton({ height = 260 }) {
  return <div className="animate-pulse rounded bg-slate-100" style={{ height }} />
}
