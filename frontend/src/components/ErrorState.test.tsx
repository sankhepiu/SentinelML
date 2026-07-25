import { describe, expect, it, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { ErrorState } from './ErrorState'
import { ApiError } from '../api/client'

describe('ErrorState', () => {
  it('renders a plain Error message', () => {
    render(<ErrorState error={new Error('boom')} />)

    expect(screen.getByText('boom')).toBeInTheDocument()
  })

  it('renders an ApiError string detail', () => {
    render(<ErrorState error={new ApiError(503, 'Model is not loaded')} />)

    expect(screen.getByText('Model is not loaded')).toBeInTheDocument()
  })

  it('renders a structured ApiError detail readably', () => {
    render(<ErrorState error={new ApiError(422, { missing_features: ['Flow Duration'] })} />)

    expect(screen.getByText(/missing features/)).toBeInTheDocument()
    expect(screen.getByText(/Flow Duration/)).toBeInTheDocument()
  })

  it('calls onRetry when the retry button is clicked', async () => {
    const onRetry = vi.fn()
    render(<ErrorState error={new Error('boom')} onRetry={onRetry} />)

    await userEvent.click(screen.getByRole('button', { name: /retry/i }))

    expect(onRetry).toHaveBeenCalledOnce()
  })

  it('omits the retry button when onRetry is not provided', () => {
    render(<ErrorState error={new Error('boom')} />)

    expect(screen.queryByRole('button', { name: /retry/i })).not.toBeInTheDocument()
  })
})
