import { apiPost } from './client'

export interface PredictionResult {
  predicted_class: string
  confidence: number
  class_probabilities: Record<string, number> | null
  model_version: string
}

export function predictOne(features: Record<string, number>): Promise<PredictionResult> {
  return apiPost<PredictionResult>('/predict', { features })
}

export interface BatchPredictionResult {
  predictions: PredictionResult[]
  count: number
}

export function predictBatch(instances: Record<string, number>[]): Promise<BatchPredictionResult> {
  return apiPost<BatchPredictionResult>('/predict/batch', { instances })
}
