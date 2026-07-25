type StatusTone = 'good' | 'warning' | 'critical' | 'neutral'

const TONE_STYLES: Record<StatusTone, { dot: string; text: string }> = {
  good: { dot: 'bg-[var(--status-good)]', text: 'text-[var(--status-good)]' },
  warning: { dot: 'bg-[var(--status-warning)]', text: 'text-[var(--status-warning)]' },
  critical: { dot: 'bg-[var(--status-critical)]', text: 'text-[var(--status-critical)]' },
  neutral: { dot: 'bg-text-muted', text: 'text-text-muted' },
}

interface StatusBadgeProps {
  tone: StatusTone
  label: string
}

/** Status is always icon/dot + label -- never color alone. */
export function StatusBadge({ tone, label }: StatusBadgeProps) {
  const styles = TONE_STYLES[tone]
  return (
    <span className="inline-flex items-center gap-2 rounded-full border border-black/10 bg-surface-1 px-3 py-1 text-sm dark:border-white/10">
      <span className={`h-2 w-2 rounded-full ${styles.dot}`} aria-hidden="true" />
      <span className={`font-medium ${styles.text}`}>{label}</span>
    </span>
  )
}
