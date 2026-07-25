import { useMutation } from '@tanstack/react-query'
import { predictBatch, predictOne } from '../api/prediction'

export function usePredict() {
  return useMutation({
    mutationFn: predictOne,
  })
}

export function usePredictBatch() {
  return useMutation({
    mutationFn: predictBatch,
  })
}
