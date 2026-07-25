import { afterEach, describe, expect, it, vi } from 'vitest'
import { apiGet, apiGetTolerant, apiPost, ApiError, formatApiErrorDetail } from './client'

describe('API_BASE_URL', () => {
  afterEach(() => {
    vi.unstubAllEnvs()
    vi.resetModules()
  })

  it('falls back to /api/v1 when VITE_API_BASE_URL is unset', async () => {
    vi.stubEnv('VITE_API_BASE_URL', undefined as unknown as string)
    vi.resetModules()

    const { API_BASE_URL } = await import('./client')

    expect(API_BASE_URL).toBe('/api/v1')
  })

  it('falls back to /api/v1 when VITE_API_BASE_URL is an empty string (e.g. an unset Docker build ARG)', async () => {
    vi.stubEnv('VITE_API_BASE_URL', '')
    vi.resetModules()

    const { API_BASE_URL } = await import('./client')

    expect(API_BASE_URL).toBe('/api/v1')
  })

  it('uses an explicitly set VITE_API_BASE_URL', async () => {
    vi.stubEnv('VITE_API_BASE_URL', 'https://api.example.com/api/v1')
    vi.resetModules()

    const { API_BASE_URL } = await import('./client')

    expect(API_BASE_URL).toBe('https://api.example.com/api/v1')
  })
})

function mockFetchOnce(status: number, body: unknown) {
  vi.stubGlobal(
    'fetch',
    vi.fn().mockResolvedValue({
      ok: status >= 200 && status < 300,
      status,
      json: () => Promise.resolve(body),
    }),
  )
}

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('apiGet', () => {
  it('resolves with the parsed body on a 2xx response', async () => {
    mockFetchOnce(200, { ready: true })

    await expect(apiGet('/ready')).resolves.toEqual({ ready: true })
  })

  it('throws an ApiError carrying the detail field on a non-2xx response', async () => {
    mockFetchOnce(503, { detail: 'Model is not loaded' })

    await expect(apiGet('/model')).rejects.toMatchObject({
      status: 503,
      detail: 'Model is not loaded',
    })
  })

  it('throws an ApiError with the raw body when there is no detail field', async () => {
    mockFetchOnce(500, { error: 'internal_server_error' })

    const error = await apiGet('/model').catch((e: unknown) => e)

    expect(error).toBeInstanceOf(ApiError)
    expect((error as ApiError).detail).toEqual({ error: 'internal_server_error' })
  })
})

describe('apiPost', () => {
  it('sends a JSON body and resolves with the parsed response', async () => {
    mockFetchOnce(200, { predicted_class: 'BENIGN' })

    const result = await apiPost('/predict', { features: { a: 1 } })

    expect(result).toEqual({ predicted_class: 'BENIGN' })
    expect(fetch).toHaveBeenCalledWith(
      expect.stringContaining('/predict'),
      expect.objectContaining({ method: 'POST' }),
    )
  })

  it('surfaces structured validation detail on a 422', async () => {
    mockFetchOnce(422, { detail: { missing_features: ['Flow Duration'] } })

    await expect(apiPost('/predict', { features: {} })).rejects.toMatchObject({
      status: 422,
      detail: { missing_features: ['Flow Duration'] },
    })
  })
})

describe('apiGetTolerant', () => {
  it('returns the body even on a non-2xx response', async () => {
    mockFetchOnce(503, { ready: false, detail: 'not loaded' })

    await expect(apiGetTolerant('/ready')).resolves.toEqual({ ready: false, detail: 'not loaded' })
  })
})

describe('formatApiErrorDetail', () => {
  it('returns a string detail as-is', () => {
    expect(formatApiErrorDetail('Model is not loaded')).toBe('Model is not loaded')
  })

  it('formats a structured detail object into one readable line', () => {
    const formatted = formatApiErrorDetail({
      missing_features: ['Flow Duration', 'Destination Port'],
    })

    expect(formatted).toContain('missing features')
    expect(formatted).toContain('Flow Duration')
    expect(formatted).toContain('Destination Port')
  })

  it('falls back to a generic message for an unrecognized shape', () => {
    expect(formatApiErrorDetail(undefined)).toBe('An unexpected error occurred.')
  })
})
