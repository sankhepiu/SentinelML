import { useQuery } from '@tanstack/react-query'
import { fetchModelInfo, fetchTrainingSummary } from '../api/model'

export function useModelInfo() {
  return useQuery({
    queryKey: ['model'],
    queryFn: fetchModelInfo,
    retry: false,
  })
}

export function useTrainingSummary() {
  return useQuery({
    queryKey: ['model', 'training-summary'],
    queryFn: fetchTrainingSummary,
    retry: false,
  })
}
