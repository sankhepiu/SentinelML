interface LoadingStateProps {
  label?: string
}

export function LoadingState({ label = 'Loading…' }: LoadingStateProps) {
  return (
    <div className="flex items-center gap-3 rounded-xl border border-black/10 bg-surface-1 p-6 text-sm text-text-secondary dark:border-white/10">
      <span
        className="h-4 w-4 animate-spin rounded-full border-2 border-[var(--series-1)] border-t-transparent"
        aria-hidden="true"
      />
      <span>{label}</span>
    </div>
  )
}
