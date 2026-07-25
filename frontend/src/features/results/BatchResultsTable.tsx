import type { PredictionResult } from '../../api/prediction'

interface BatchResultsTableProps {
  predictions: PredictionResult[]
}

export function BatchResultsTable({ predictions }: BatchResultsTableProps) {
  return (
    <div className="max-h-96 overflow-auto">
      <table className="w-full min-w-max text-left text-sm">
        <thead className="sticky top-0 bg-surface-1">
          <tr className="border-b border-black/10 text-text-muted dark:border-white/10">
            <th className="py-2 pr-4 font-medium">#</th>
            <th className="py-2 pr-4 font-medium">Predicted class</th>
            <th className="py-2 pr-4 font-medium">Confidence</th>
          </tr>
        </thead>
        <tbody>
          {predictions.map((prediction, index) => (
            <tr key={index} className="border-b border-black/5 dark:border-white/5">
              <td className="py-1.5 pr-4 tabular-nums text-text-muted">{index + 1}</td>
              <td className="py-1.5 pr-4 text-text-primary">{prediction.predicted_class}</td>
              <td className="py-1.5 pr-4 tabular-nums text-text-secondary">
                {(prediction.confidence * 100).toFixed(1)}%
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
