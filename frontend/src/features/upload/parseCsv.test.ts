import { describe, expect, it } from 'vitest'
import { parseFeatureCsv } from './parseCsv'

const FEATURES = ['Destination Port', 'Flow Duration']

describe('parseFeatureCsv', () => {
  it('parses a valid CSV into numeric rows', () => {
    const csv = 'Destination Port,Flow Duration\n443,1000\n80,2000\n'

    const result = parseFeatureCsv(csv, FEATURES)

    expect(result.errors).toEqual([])
    expect(result.rows).toEqual([
      { 'Destination Port': 443, 'Flow Duration': 1000 },
      { 'Destination Port': 80, 'Flow Duration': 2000 },
    ])
  })

  it('reports missing columns and produces no rows', () => {
    const csv = 'Destination Port\n443\n'

    const result = parseFeatureCsv(csv, FEATURES)

    expect(result.errors.some((e) => e.includes('Missing columns'))).toBe(true)
    expect(result.errors.some((e) => e.includes('Flow Duration'))).toBe(true)
    expect(result.rows).toEqual([])
  })

  it('reports unexpected extra columns', () => {
    const csv = 'Destination Port,Flow Duration,Extra Column\n443,1000,9\n'

    const result = parseFeatureCsv(csv, FEATURES)

    expect(result.errors.some((e) => e.includes('Unexpected columns'))).toBe(true)
    expect(result.errors.some((e) => e.includes('Extra Column'))).toBe(true)
  })

  it('reports a specific row/column for a non-numeric value', () => {
    const csv = 'Destination Port,Flow Duration\n443,not-a-number\n'

    const result = parseFeatureCsv(csv, FEATURES)

    expect(result.errors.some((e) => e.includes('Row 2') && e.includes('Flow Duration'))).toBe(
      true,
    )
  })

  it('handles an empty file with just a header', () => {
    const csv = 'Destination Port,Flow Duration\n'

    const result = parseFeatureCsv(csv, FEATURES)

    expect(result.errors).toEqual([])
    expect(result.rows).toEqual([])
  })
})
