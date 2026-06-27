import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts'
import { useCases } from '../hooks/useSurveillanceData.js'
import Card, { Skeleton } from './Card.jsx'

export default function TrendChart({ disease, state }) {
  const { data = [], isLoading } = useCases(disease, state)
  return (
    <Card title="Weekly confirmed cases" subtitle={state === 'all' ? 'National · 2015–2024' : `${state} · 2015–2024`}>
      {isLoading ? (
        <Skeleton />
      ) : (
        <ResponsiveContainer width="100%" height={260}>
          <LineChart data={data} margin={{ top: 5, right: 10, bottom: 0, left: -10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#eef2f7" />
            <XAxis dataKey="date" tick={{ fontSize: 10 }} minTickGap={50} />
            <YAxis tick={{ fontSize: 10 }} />
            <Tooltip />
            <Line type="monotone" dataKey="confirmed" stroke="#0f172a" dot={false} strokeWidth={1.5} />
          </LineChart>
        </ResponsiveContainer>
      )}
    </Card>
  )
}
