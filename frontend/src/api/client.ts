export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? '/api/v1'

/**
 * Thrown for any non-2xx response. `detail` carries FastAPI's `detail` field
 * as-is -- a plain string for most errors, or a structured object for
 * prediction payload validation (`{missing_features, unexpected_features}`).
 */
export class ApiError extends Error {
  status: number
  detail: unknown

  constructor(status: number, detail: unknown) {
    super(typeof detail === 'string' ? detail : `Request failed with status ${status}`)
    this.name = 'ApiError'
    this.status = status
    this.detail = detail
  }
}

/** Render an ApiError's `detail` (string or structured object) as one readable line. */
export function formatApiErrorDetail(detail: unknown): string {
  if (typeof detail === 'string') return detail
  if (detail && typeof detail === 'object') {
    return Object.entries(detail as Record<string, unknown>)
      .map(([key, value]) => `${key.replaceAll('_', ' ')}: ${Array.isArray(value) ? value.join(', ') : String(value)}`)
      .join(' — ')
  }
  return 'An unexpected error occurred.'
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let detail: unknown
    try {
      const body = await response.json()
      detail = body?.detail ?? body
    } catch {
      detail = undefined
    }
    throw new ApiError(response.status, detail)
  }
  return response.json() as Promise<T>
}

export async function apiGet<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`)
  return handleResponse<T>(response)
}

export async function apiPost<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  return handleResponse<T>(response)
}

/** Fetches a path and always returns the parsed JSON body, regardless of status code. */
export async function apiGetTolerant<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`)
  return response.json() as Promise<T>
}
