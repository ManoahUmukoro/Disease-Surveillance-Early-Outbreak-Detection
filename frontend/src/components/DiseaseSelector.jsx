const DISEASES = [
  { key: 'lassa', label: 'Lassa Fever' },
  { key: 'cholera', label: 'Cholera' },
  { key: 'meningitis', label: 'Meningitis (CSM)' },
  { key: 'mpox', label: 'Mpox' },
]

export default function DiseaseSelector({ disease, onChange }) {
  return (
    <div className="inline-flex flex-wrap rounded-lg border border-slate-200 bg-white p-1 shadow-sm">
      {DISEASES.map((d) => (
        <button
          key={d.key}
          onClick={() => onChange(d.key)}
          className={`rounded-md px-3 py-1.5 text-sm font-medium transition ${
            disease === d.key ? 'bg-slate-900 text-white' : 'text-slate-600 hover:bg-slate-100'
          }`}
        >
          {d.label}
        </button>
      ))}
    </div>
  )
}
