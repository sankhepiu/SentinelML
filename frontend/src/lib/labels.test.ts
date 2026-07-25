import { describe, expect, it } from 'vitest'
import { orderedLabelNames } from './labels'

describe('orderedLabelNames', () => {
  it('orders class names by their numeric label code, not insertion or string order', () => {
    const mapping = { '2': 'DoS Hulk', '0': 'BENIGN', '10': 'Heartbleed', '1': 'DoS GoldenEye' }

    expect(orderedLabelNames(mapping)).toEqual(['BENIGN', 'DoS GoldenEye', 'DoS Hulk', 'Heartbleed'])
  })

  it('returns an empty array for an empty mapping', () => {
    expect(orderedLabelNames({})).toEqual([])
  })
})
