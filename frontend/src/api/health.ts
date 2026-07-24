import { apiGet } from './client'

export interface HealthResponse {
  status: string
  app_name: string
  environment: string
}

export function fetchHealth(): Promise<HealthResponse> {
  return apiGet<HealthResponse>('/health')
}
