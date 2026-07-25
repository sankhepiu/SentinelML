interface MetricCardProps {
  label: string
  value: string
  hint?: string
}

/** Stat tile: label (sentence case) + semibold proportional-figure value + optional hint. */
export function MetricCard({ label, value, hint }: MetricCardProps) {
  return (
    <div className="rounded-xl border border-black/10 bg-surface-1 p-4 dark:border-white/10">
      <p className="text-sm text-text-secondary">{label}</p>
      <p className="mt-1 text-2xl font-semibold text-text-primary [font-variant-numeric:normal]">
        {value}
      </p>
      {hint && <p className="mt-1 text-xs text-text-muted">{hint}</p>}
    </div>
  )
}
