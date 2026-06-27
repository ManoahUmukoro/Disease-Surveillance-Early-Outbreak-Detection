import { useAnomalies } from '../hooks/useSurveillanceData.js'
import Card, { Skeleton } from './Card.jsx'

const STATUS = {
  HIGH: { dot: 'bg-red-500', label: 'High risk', tag: 'Red' },
  MEDIUM: { dot: 'bg-amber-500', label: 'Elevated', tag: 'Amber' },
  NORMAL: { dot: 'bg-emerald-500', label: 'Normal', tag: 'Green' },
}

export default function AnomalyPanel({ disease, state }) {
  const { data = [], isLoading } = useAnomalies(disease, state)
  const latest = data.length ? data[data.length - 1] : null
  const recent = data.slice(-26)
  const s = STATUS[latest?.severity] || STATUS.NORMAL
  return (
    <Card title="Outbreak status" subtitle={state === 'all' ? 'National · latest week' : `${state} · latest week`}>
      {isLoading || !latest ? (
        <Skeleton />
      ) : (
        <div>
          <div className="flex items-center gap-3">
            <span className={`inline-block h-4 w-4 rounded-full ${s.dot}`} />
            <span className="text-lg font-semibold text-slate-800">{s.label}</span>
            <span className="text-xs text-slate-400">({s.tag})</span>
          </div>
          <div className="mt-5 flex flex-wrap gap-1">
            {recent.map((p) => (
              <span
                key={p.date}
                title={`${p.date} · ${p.severity} · ${p.cases} cases`}
                className={`h-5 w-5 rounded ${(STATUS[p.severity] || STATUS.NORMAL).dot} opacity-80`}
              />
            ))}
          </div>
          <p className="mt-4 text-xs text-slate-400">
            Last {recent.length} weeks · {latest.cases.toLocaleString()} confirmed in the latest week
          </p>
        </div>
      )}
    </Card>
  )
}
