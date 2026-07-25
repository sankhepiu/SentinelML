import type { ReactNode } from 'react'

interface CardProps {
  title?: string
  description?: string
  children: ReactNode
  className?: string
}

export function Card({ title, description, children, className = '' }: CardProps) {
  return (
    <section
      className={`rounded-xl border border-black/10 bg-surface-1 p-5 shadow-sm dark:border-white/10 ${className}`}
    >
      {title && <h2 className="text-sm font-semibold text-text-primary">{title}</h2>}
      {description && <p className="mt-1 text-sm text-text-secondary">{description}</p>}
      <div className={title || description ? 'mt-4' : ''}>{children}</div>
    </section>
  )
}
