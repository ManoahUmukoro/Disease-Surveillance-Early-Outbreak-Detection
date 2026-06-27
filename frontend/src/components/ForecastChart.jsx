import { ComposedChart, Area, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts'
import { useForecast } from '../hooks/useSurveillanceData.js'
import Card, { Skeleton } from './Card.jsx'

export default function ForecastChart({ disease }) {
  const { data, isLoading } = useForecast(disease)
  const fc = data?.forecast || []
  const mape = data?.metrics?.mape
  // Stacked areas render the confidence band: an invisible base (lower_ci) plus
  // the visible band height (upper_ci − lower_ci).
  const chart = fc.map((p) => ({
    week: `wk ${p.week}`,
    predicted: p.predicted,
    base: p.lower_ci,
    band: Math.max(p.upper_ci - p.lower_ci, 0),
  }))
  return (
    <Card
      title="4-week forecast (national)"
      subtitle="LSTM · shaded 95% confidence band"
      right={mape != null && (
        <span className="rounded bg-slate-100 px-2 py-1 text-xs text-slate-500">MAPE {mape}%</span>
      )}
    >
      {isLoading ? (
        <Skeleton />
      ) : (
        <ResponsiveContainer width="100%" height={260}>
          <ComposedChart data={chart} margin={{ top: 5, right: 10, bottom: 0, left: -10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#eef2f7" />
            <XAxis dataKey="week" tick={{ fontSize: 11 }} />
            <YAxis tick={{ fontSize: 10 }} />
            <Tooltip />
            <Area dataKey="base" stackId="ci" stroke="none" fill="transparent" isAnimationActive={false} />
            <Area dataKey="band" stackId="ci" stroke="none" fill="#93c5fd" fillOpacity={0.45} isAnimationActive={false} />
            <Line dataKey="predicted" stroke="#1d4ed8" strokeWidth={2} dot={{ r: 3 }} />
          </ComposedChart>
        </ResponsiveContainer>
      )}
    </Card>
  )
}
