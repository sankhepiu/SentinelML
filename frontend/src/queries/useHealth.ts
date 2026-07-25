import { useQuery } from '@tanstack/react-query'
import { fetchHealth, fetchReadiness } from '../api/health'

export function useHealth() {
  return useQuery({
    queryKey: ['health'],
    queryFn: fetchHealth,
    retry: false,
    refetchInterval: 15_000,
  })
}

export function useReadiness() {
  return useQuery({
    queryKey: ['ready'],
    queryFn: fetchReadiness,
    retry: false,
    refetchInterval: 15_000,
  })
}
