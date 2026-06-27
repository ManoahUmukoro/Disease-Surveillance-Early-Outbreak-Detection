import { useState } from 'react'
import DiseaseSelector from '../components/DiseaseSelector.jsx'
import StateFilter from '../components/StateFilter.jsx'
import SummaryBar from '../components/SummaryBar.jsx'
import TrendChart from '../components/TrendChart.jsx'
import ForecastChart from '../components/ForecastChart.jsx'
import AnomalyPanel from '../components/AnomalyPanel.jsx'
import AlertsPanel from '../components/AlertsPanel.jsx'

export default function Dashboard() {
  const [disease, setDisease] = useState('lassa')
  const [state, setState] = useState('all')

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800">
      <header className="bg-slate-900 text-white">
        <div className="mx-auto max-w-7xl px-6 py-5">
          <h1 className="text-xl font-bold">Disease Surveillance &amp; Early Outbreak Detection</h1>
          <p className="text-sm text-slate-300">
            Nigeria · NCDC 2015–2024 · ML-driven early warning (LSTM forecasting + anomaly detection)
          </p>
        </div>
      </header>

      <main className="mx-auto max-w-7xl space-y-6 px-6 py-6">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <DiseaseSelector disease={disease} onChange={setDisease} />
          <StateFilter state={state} onChange={setState} />
        </div>

        <SummaryBar disease={disease} state={state} />

        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          <TrendChart disease={disease} state={state} />
          <ForecastChart disease={disease} />
          <AnomalyPanel disease={disease} state={state} />
          <AlertsPanel disease={disease} state={state} />
        </div>

        <footer className="pt-2 text-center text-xs text-slate-400">
          Prototype · forecasts and alerts precomputed and served from MongoDB via FastAPI
        </footer>
      </main>
    </div>
  )
}
