import type { PredictionResult } from '../../api/prediction'

export function classBreakdown(predictions: PredictionResult[]): Record<string, number> {
  const counts: Record<string, number> = {}
  for (const prediction of predictions) {
    counts[prediction.predicted_class] = (counts[prediction.predicted_class] ?? 0) + 1
  }
  return counts
}
