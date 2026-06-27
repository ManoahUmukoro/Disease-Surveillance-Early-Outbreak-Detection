import { useSummary } from '../hooks/useSurveillanceData.js'

function Stat({ label, value, accent }) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <div className="text-xs uppercase tracking-wide text-slate-400">{label}</div>
      <div className={`mt-1 text-2xl font-bold ${accent || 'text-slate-800'}`}>{value}</div>
    </div>
  )
}

export default function SummaryBar({ disease, state }) {
  const { data, isLoading } = useSummary(disease, state)
  if (isLoading || !data) {
    return <div className="h-[88px] animate-pulse rounded-xl bg-slate-100" />
  }
  return (
    <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
      <Stat label="Total confirmed" value={data.total_cases.toLocaleString()} />
      <Stat label="Active alerts (2024)" value={data.active_alerts} accent="text-amber-600" />
      <Stat label="States affected" value={data.states_affected} />
      <Stat label="Case fatality rate" value={`${(data.cfr * 100).toFixed(1)}%`} accent="text-red-600" />
    </div>
  )
}
