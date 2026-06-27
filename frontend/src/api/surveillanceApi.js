import axios from 'axios'

// Empty baseURL in dev → relative /api/* → proxied to the backend by Vite.
// In production set VITE_API_BASE_URL to the deployed backend URL.
const client = axios.create({ baseURL: import.meta.env.VITE_API_BASE_URL || '' })

export const getDiseases = () => client.get('/api/diseases').then((r) => r.data)
export const getStates = () => client.get('/api/states').then((r) => r.data)
export const getCases = (disease, state = 'all') =>
  client.get('/api/cases', { params: { disease, state } }).then((r) => r.data)
export const getForecast = (disease) =>
  client.get('/api/forecast', { params: { disease } }).then((r) => r.data)
export const getAnomalies = (disease, state = 'all') =>
  client.get('/api/anomalies', { params: { disease, state } }).then((r) => r.data)
export const getAlerts = (disease, state = 'all', limit = 15) =>
  client.get('/api/alerts', { params: { disease, state, limit } }).then((r) => r.data)
export const getSummary = (disease, state = 'all') =>
  client.get('/api/summary', { params: { disease, state } }).then((r) => r.data)
