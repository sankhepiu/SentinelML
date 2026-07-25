import { apiGet, apiGetTolerant } from './client'

export interface HealthResponse {
  status: string
  app_name: string
  environment: string
}

export interface ReadinessResponse {
  ready: boolean
  detail: string | null
}

export function fetchHealth(): Promise<HealthResponse> {
  return apiGet<HealthResponse>('/health')
}

/**
 * /ready responds 503 (with a valid body) when the model isn't loaded --
 * that's expected data for a status indicator, not a request failure, so
 * this never throws on a non-2xx response.
 */
export function fetchReadiness(): Promise<ReadinessResponse> {
  return apiGetTolerant<ReadinessResponse>('/ready')
}
