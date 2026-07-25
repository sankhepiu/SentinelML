import { ApiError, formatApiErrorDetail } from '../api/client'

export function describeError(error: unknown): string {
  if (error instanceof ApiError) return formatApiErrorDetail(error.detail)
  if (error instanceof Error) return error.message
  return 'An unexpected error occurred.'
}
