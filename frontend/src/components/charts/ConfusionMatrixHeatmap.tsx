import { sequentialBlueAt } from '../../lib/chartTokens'

interface ConfusionMatrixHeatmapProps {
  matrix: number[][]
  labels: string[]
}

/**
 * Grid of counts, sequential blue per cell (row-normalized so every true
 * class's own scale is visible even under CICIDS2017's extreme imbalance).
 * The count is always the direct label -- required here, not optional:
 * this is a grid of exact values, not a shape meant to be read by color
 * alone.
 */
export function ConfusionMatrixHeatmap({ matrix, labels }: ConfusionMatrixHeatmapProps) {
  const rowMax = matrix.map((row) => Math.max(...row, 1))

  return (
    <div className="overflow-x-auto">
      <p className="mb-2 text-xs text-text-muted">Rows = true class, columns = predicted class</p>
      <table className="border-separate" style={{ borderSpacing: 2 }}>
        <thead>
          <tr>
            <th className="p-1" />
            {labels.map((label) => (
              <th
                key={label}
                className="max-w-16 truncate p-1 text-center text-xs font-medium text-text-secondary"
                title={label}
              >
                {label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={labels[i]}>
              <th
                className="max-w-24 truncate pr-2 text-right text-xs font-medium text-text-secondary"
                title={labels[i]}
              >
                {labels[i]}
              </th>
              {row.map((count, j) => {
                const intensity = count / rowMax[i]
                const background = count === 0 ? 'var(--gridline)' : sequentialBlueAt(0.15 + 0.75 * intensity)
                const useWhiteText = count > 0 && intensity > 0.6
                return (
                  <td key={labels[j]} className="p-0">
                    <div
                      className="flex h-11 w-14 items-center justify-center rounded text-xs font-medium tabular-nums"
                      style={{
                        backgroundColor: background,
                        color: useWhiteText ? '#ffffff' : 'var(--text-primary)',
                      }}
                      title={`True ${labels[i]}, predicted ${labels[j]}: ${count.toLocaleString()}`}
                    >
                      {count.toLocaleString()}
                    </div>
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
