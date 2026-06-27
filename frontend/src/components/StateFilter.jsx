import { useStates } from '../hooks/useSurveillanceData.js'

export default function StateFilter({ state, onChange }) {
  const { data: states = [] } = useStates()
  return (
    <select
      value={state}
      onChange={(e) => onChange(e.target.value)}
      className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm shadow-sm focus:outline-none focus:ring-2 focus:ring-slate-300"
    >
      <option value="all">All states (national)</option>
      {states.map((s) => (
        <option key={s} value={s}>{s}</option>
      ))}
    </select>
  )
}
