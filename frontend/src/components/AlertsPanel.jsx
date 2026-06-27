import { useAlerts } from '../hooks/useSurveillanceData.js'
import Card, { Skeleton } from './Card.jsx'

const BADGE = {
  HIGH: 'bg-red-100 text-red-700',
  MEDIUM: 'bg-amber-100 text-amber-700',
  NORMAL: 'bg-emerald-100 text-emerald-700',
}

export default function AlertsPanel({ disease, state }) {
  const { data = [], isLoading } = useAlerts(disease, state)
  return (
    <Card title="Recent alerts" subtitle="Most recent flagged anomalies">
      {isLoading ? (
        <Skeleton />
      ) : data.length === 0 ? (
        <p className="py-8 text-center text-sm text-slate-400">No alerts for this selection.</p>
      ) : (
        <ul className="max-h-[260px] divide-y divide-slate-100 overflow-auto">
          {data.map((a, i) => (
            <li key={i} className="flex items-center justify-between gap-3 py-2">
              <div className="min-w-0">
                <div className="truncate text-sm font-medium text-slate-700">{a.state}</div>
                <div className="text-xs text-slate-400">{a.date}</div>
              </div>
              <span className={`shrink-0 rounded px-2 py-0.5 text-xs font-medium ${BADGE[a.severity] || ''}`}>
                {a.severity}
              </span>
            </li>
          ))}
        </ul>
      )}
    </Card>
  )
}
