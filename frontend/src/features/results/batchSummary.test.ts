import { describe, expect, it } from 'vitest'
import { classBreakdown } from './batchSummary'
import type { PredictionResult } from '../../api/prediction'

function result(predicted_class: string): PredictionResult {
  return { predicted_class, confidence: 0.9, class_probabilities: null, model_version: 'v1' }
}

describe('classBreakdown', () => {
  it('counts predictions per class', () => {
    const predictions = [result('BENIGN'), result('BENIGN'), result('DoS Hulk')]

    expect(classBreakdown(predictions)).toEqual({ BENIGN: 2, 'DoS Hulk': 1 })
  })

  it('returns an empty object for no predictions', () => {
    expect(classBreakdown([])).toEqual({})
  })
})
