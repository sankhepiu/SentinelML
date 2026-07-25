import { afterEach, describe, expect, it, vi } from 'vitest'
import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { renderWithProviders } from '../test/testUtils'
import { SinglePredictionPage } from './SinglePredictionPage'

function mockApi() {
  vi.stubGlobal(
    'fetch',
    vi.fn((url: string, init?: RequestInit) => {
      if (typeof url === 'string' && url.includes('/model') && init?.method === undefined) {
        return Promise.resolve({
          ok: true,
          status: 200,
          json: () =>
            Promise.resolve({
              model_type: 'lightgbm',
              model_version: 'v1',
              feature_names: ['Destination Port', 'Flow Duration'],
              label_classes: ['BENIGN', 'DoS Hulk'],
              metrics: { accuracy: 0.99 },
            }),
        })
      }
      if (typeof url === 'string' && url.includes('/predict') && init?.method === 'POST') {
        return Promise.resolve({
          ok: true,
          status: 200,
          json: () =>
            Promise.resolve({
              predicted_class: 'BENIGN',
              confidence: 0.97,
              class_probabilities: { BENIGN: 0.97, 'DoS Hulk': 0.03 },
              model_version: 'v1',
            }),
        })
      }
      return Promise.reject(new Error(`Unhandled fetch in test: ${url}`))
    }),
  )
}

afterEach(() => {
  vi.unstubAllGlobals()
  localStorage.clear()
})

describe('SinglePredictionPage', () => {
  it('loads a form field per feature, submits, and shows the prediction result', async () => {
    mockApi()
    renderWithProviders(<SinglePredictionPage />)

    expect(await screen.findByText('Destination Port')).toBeInTheDocument()
    expect(screen.getByText('Flow Duration')).toBeInTheDocument()

    await userEvent.click(screen.getByRole('button', { name: /^predict$/i }))

    await waitFor(() => expect(screen.getAllByText('97.0%').length).toBeGreaterThan(0))
    expect(screen.getAllByText('BENIGN').length).toBeGreaterThan(0)
  })

  it('records a successful prediction to history', async () => {
    mockApi()
    renderWithProviders(<SinglePredictionPage />)

    await screen.findByText('Destination Port')
    await userEvent.click(screen.getByRole('button', { name: /^predict$/i }))
    await waitFor(() => expect(screen.getAllByText('97.0%').length).toBeGreaterThan(0))

    const stored = JSON.parse(localStorage.getItem('sentinelml.prediction_history') ?? '[]')
    expect(stored).toHaveLength(1)
    expect(stored[0].predicted_class).toBe('BENIGN')
    expect(stored[0].kind).toBe('single')
  })
})
