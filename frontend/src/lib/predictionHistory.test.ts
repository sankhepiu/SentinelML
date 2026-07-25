import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { appendHistoryEntry, clearHistory } from './predictionHistory'

const STORAGE_KEY = 'sentinelml.prediction_history'

function readRaw(): unknown[] {
  return JSON.parse(localStorage.getItem(STORAGE_KEY) ?? '[]')
}

describe('predictionHistory', () => {
  beforeEach(() => {
    localStorage.clear()
  })

  afterEach(() => {
    localStorage.clear()
  })

  it('starts empty', () => {
    expect(readRaw()).toEqual([])
  })

  it('appends an entry with a generated id and timestamp', () => {
    appendHistoryEntry({
      kind: 'single',
      model_version: 'v1',
      predicted_class: 'BENIGN',
      confidence: 0.98,
    })

    const stored = readRaw() as Record<string, unknown>[]
    expect(stored).toHaveLength(1)
    expect(stored[0].id).toBeTruthy()
    expect(stored[0].timestamp).toBeTruthy()
    expect(stored[0].predicted_class).toBe('BENIGN')
  })

  it('prepends new entries so the most recent is first', () => {
    appendHistoryEntry({ kind: 'single', model_version: 'v1', predicted_class: 'BENIGN' })
    appendHistoryEntry({ kind: 'single', model_version: 'v1', predicted_class: 'DoS Hulk' })

    const stored = readRaw() as Record<string, unknown>[]
    expect(stored[0].predicted_class).toBe('DoS Hulk')
    expect(stored[1].predicted_class).toBe('BENIGN')
  })

  it('caps history at 200 entries', () => {
    for (let i = 0; i < 205; i++) {
      appendHistoryEntry({ kind: 'single', model_version: 'v1', predicted_class: `class-${i}` })
    }

    expect(readRaw()).toHaveLength(200)
  })

  it('clearHistory empties the store', () => {
    appendHistoryEntry({ kind: 'single', model_version: 'v1', predicted_class: 'BENIGN' })

    clearHistory()

    expect(readRaw()).toEqual([])
  })
})
