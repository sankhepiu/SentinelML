interface TooltipPayloadEntry {
  value: number
  name?: string
  color?: string
}

interface ChartTooltipProps {
  active?: boolean
  label?: string
  payload?: TooltipPayloadEntry[]
  valueLabel?: string
  formatValue?: (value: number) => string
}

/** Consistent tooltip styling across every chart -- surface + ink tokens, never a series color for text. */
export function ChartTooltip({ active, label, payload, valueLabel = 'Value', formatValue }: ChartTooltipProps) {
  if (!active || !payload || payload.length === 0) return null

  return (
    <div className="rounded-lg border border-black/10 bg-surface-1 px-3 py-2 text-xs shadow-md dark:border-white/10">
      {label && <p className="font-medium text-text-primary">{label}</p>}
      {payload.map((entry, index) => (
        <p key={index} className="mt-0.5 flex items-center gap-1.5 text-text-secondary">
          {entry.color && (
            <span
              className="inline-block h-2 w-2 rounded-full"
              style={{ backgroundColor: entry.color }}
              aria-hidden="true"
            />
          )}
          <span>{entry.name ?? valueLabel}:</span>
          <span className="font-medium text-text-primary">
            {formatValue ? formatValue(entry.value) : entry.value}
          </span>
        </p>
      ))}
    </div>
  )
}
