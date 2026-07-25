import { describe, expect, it } from 'vitest'
import { render, screen } from '@testing-library/react'
import { StatusBadge } from './StatusBadge'

describe('StatusBadge', () => {
  it('renders the label text', () => {
    render(<StatusBadge tone="good" label="API ready" />)

    expect(screen.getByText('API ready')).toBeInTheDocument()
  })

  it.each(['good', 'warning', 'critical', 'neutral'] as const)('renders for tone=%s', (tone) => {
    render(<StatusBadge tone={tone} label="status" />)

    expect(screen.getByText('status')).toBeInTheDocument()
  })
})
