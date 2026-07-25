import { PageHeader } from '../components/PageHeader'
import { Card } from '../components/Card'
import { clearHistory, usePredictionHistory } from '../lib/predictionHistory'

export function PredictionHistoryPage() {
  const history = usePredictionHistory()

  return (
    <div>
      <PageHeader
        title="Prediction History"
        description="Predictions made from this browser, most recent first. Stored locally -- clearing your browser data clears this too."
      />

      <Card>
        <div className="mb-4 flex items-center justify-between">
          <p className="text-sm text-text-secondary">
            {history.length} entr{history.length === 1 ? 'y' : 'ies'}
          </p>
          {history.length > 0 && (
            <button
              type="button"
              onClick={clearHistory}
              className="rounded-md border border-black/10 px-3 py-1.5 text-sm font-medium text-text-primary hover:bg-black/5 dark:border-white/10 dark:hover:bg-white/5"
            >
              Clear history
            </button>
          )}
        </div>

        {history.length === 0 ? (
          <p className="text-sm text-text-muted">
            No predictions yet. Try the Single or Batch Prediction pages.
          </p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full min-w-max text-left text-sm">
              <thead>
                <tr className="border-b border-black/10 text-text-muted dark:border-white/10">
                  <th className="py-2 pr-4 font-medium">Time</th>
                  <th className="py-2 pr-4 font-medium">Type</th>
                  <th className="py-2 pr-4 font-medium">Result</th>
                  <th className="py-2 pr-4 font-medium">Model version</th>
                </tr>
              </thead>
              <tbody>
                {history.map((entry) => (
                  <tr key={entry.id} className="border-b border-black/5 dark:border-white/5">
                    <td className="py-2 pr-4 whitespace-nowrap text-text-secondary">
                      {new Date(entry.timestamp).toLocaleString()}
                    </td>
                    <td className="py-2 pr-4">
                      <span className="rounded-full border border-black/10 px-2 py-0.5 text-xs capitalize text-text-secondary dark:border-white/10">
                        {entry.kind}
                      </span>
                    </td>
                    <td className="py-2 pr-4 text-text-primary">
                      {entry.kind === 'single' ? (
                        <span>
                          {entry.predicted_class} ·{' '}
                          {entry.confidence !== undefined
                            ? `${(entry.confidence * 100).toFixed(1)}%`
                            : '—'}
                        </span>
                      ) : (
                        <span>
                          {entry.count} rows
                          {entry.class_breakdown && (
                            <>
                              {' — '}
                              {Object.entries(entry.class_breakdown)
                                .sort((a, b) => b[1] - a[1])
                                .slice(0, 3)
                                .map(([label, count]) => `${label} (${count})`)
                                .join(', ')}
                            </>
                          )}
                        </span>
                      )}
                    </td>
                    <td className="py-2 pr-4 text-text-secondary tabular-nums">
                      {entry.model_version}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  )
}
