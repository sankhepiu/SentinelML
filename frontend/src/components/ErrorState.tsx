import { describeError } from '../lib/errors'

interface ErrorStateProps {
  error: unknown
  onRetry?: () => void
  title?: string
}

export function ErrorState({ error, onRetry, title = 'Something went wrong' }: ErrorStateProps) {
  return (
    <div className="rounded-xl border border-[var(--status-critical)]/30 bg-[var(--status-critical)]/5 p-6">
      <div className="flex items-start gap-3">
        <span
          className="mt-0.5 h-2 w-2 shrink-0 rounded-full bg-[var(--status-critical)]"
          aria-hidden="true"
        />
        <div className="flex-1">
          <p className="text-sm font-medium text-[var(--status-critical)]">{title}</p>
          <p className="mt-1 text-sm text-text-secondary">{describeError(error)}</p>
          {onRetry && (
            <button
              type="button"
              onClick={onRetry}
              className="mt-3 rounded-md border border-black/10 bg-surface-1 px-3 py-1.5 text-sm font-medium text-text-primary hover:bg-black/5 dark:border-white/10 dark:hover:bg-white/5"
            >
              Retry
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
