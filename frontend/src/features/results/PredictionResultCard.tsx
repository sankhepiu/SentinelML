interface PredictionResultCardProps {
  predictedClass: string
  confidence: number
  classProbabilities: Record<string, number> | null
  modelVersion: string
}

/**
 * Emphasis pattern: the predicted class is the point, every other class is
 * context -> accent color on the winner, muted gray on the rest, direct
 * percentage labels throughout (never color alone).
 */
export function PredictionResultCard({
  predictedClass,
  confidence,
  classProbabilities,
  modelVersion,
}: PredictionResultCardProps) {
  const sortedProbabilities = classProbabilities
    ? Object.entries(classProbabilities).sort((a, b) => b[1] - a[1])
    : []

  return (
    <div className="rounded-xl border border-[var(--status-good)]/30 bg-[var(--status-good)]/5 p-5">
      <div className="flex items-center justify-between gap-4">
        <div>
          <p className="text-xs text-text-secondary">Predicted class</p>
          <p className="text-xl font-semibold text-text-primary">{predictedClass}</p>
        </div>
        <div className="text-right">
          <p className="text-xs text-text-secondary">Confidence</p>
          <p className="text-xl font-semibold text-[var(--status-good)]">
            {(confidence * 100).toFixed(1)}%
          </p>
        </div>
      </div>

      {sortedProbabilities.length > 0 && (
        <div className="mt-4 space-y-2">
          <p className="text-xs font-medium text-text-secondary">Class probabilities</p>
          {sortedProbabilities.map(([label, probability]) => {
            const isPredicted = label === predictedClass
            return (
              <div key={label} className="flex items-center gap-2">
                <span className="w-32 shrink-0 truncate text-xs text-text-secondary" title={label}>
                  {label}
                </span>
                <div className="h-2 flex-1 overflow-hidden rounded-full bg-black/5 dark:bg-white/10">
                  <div
                    className="h-full rounded-full"
                    style={{
                      width: `${Math.max(probability * 100, 1)}%`,
                      backgroundColor: isPredicted ? 'var(--status-good)' : 'var(--text-muted)',
                    }}
                  />
                </div>
                <span className="w-14 shrink-0 text-right text-xs tabular-nums text-text-secondary">
                  {(probability * 100).toFixed(1)}%
                </span>
              </div>
            )
          })}
        </div>
      )}

      <p className="mt-3 text-xs text-text-muted">Model version {modelVersion}</p>
    </div>
  )
}
