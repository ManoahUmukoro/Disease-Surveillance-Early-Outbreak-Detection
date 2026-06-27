import { useQuery } from '@tanstack/react-query'
import * as api from '../api/surveillanceApi.js'

export const useDiseases = () =>
  useQuery({ queryKey: ['diseases'], queryFn: api.getDiseases })

export const useStates = () =>
  useQuery({ queryKey: ['states'], queryFn: api.getStates })

export const useCases = (disease, state) =>
  useQuery({ queryKey: ['cases', disease, state], queryFn: () => api.getCases(disease, state) })

export const useForecast = (disease) =>
  useQuery({ queryKey: ['forecast', disease], queryFn: () => api.getForecast(disease) })

export const useAnomalies = (disease, state) =>
  useQuery({ queryKey: ['anomalies', disease, state], queryFn: () => api.getAnomalies(disease, state) })

export const useAlerts = (disease, state) =>
  useQuery({ queryKey: ['alerts', disease, state], queryFn: () => api.getAlerts(disease, state) })

export const useSummary = (disease, state) =>
  useQuery({ queryKey: ['summary', disease, state], queryFn: () => api.getSummary(disease, state) })
