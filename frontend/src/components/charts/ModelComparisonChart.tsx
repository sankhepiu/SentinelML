import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { CATEGORICAL_SERIES, CHART_INK } from '../../lib/chartTokens'
import { ChartTooltip } from './ChartTooltip'

interface ModelMetricsSubset {
  accuracy: number
  precision_macro: number
  recall_macro: number
  f1_macro: number
}

interface ModelComparisonChartProps {
  valMetrics: Record<string, ModelMetricsSubset>
  modelOrder: string[]
}

const METRIC_FIELDS: { key: keyof ModelMetricsSubset; label: string }[] = [
  { key: 'accuracy', label: 'Accuracy' },
  { key: 'precision_macro', label: 'Precision' },
  { key: 'recall_macro', label: 'Recall' },
  { key: 'f1_macro', label: 'F1 (macro)' },
]

/** Distinct named models being compared -> categorical color, fixed order, legend mandatory. */
export function ModelComparisonChart({ valMetrics, modelOrder }: ModelComparisonChartProps) {
  const data = METRIC_FIELDS.map(({ key, label }) => {
    const row: Record<string, string | number> = { metric: label }
    for (const model of modelOrder) {
      row[model] = valMetrics[model]?.[key] ?? 0
    }
    return row
  })

  return (
    <ResponsiveContainer width="100%" height={280}>
      <BarChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
        <CartesianGrid stroke={CHART_INK.gridline} vertical={false} />
        <XAxis
          dataKey="metric"
          tick={{ fill: CHART_INK.secondary, fontSize: 12 }}
          axisLine={{ stroke: CHART_INK.axis }}
          tickLine={false}
        />
        <YAxis
          domain={[0, 1]}
          tick={{ fill: CHART_INK.muted, fontSize: 12 }}
          axisLine={{ stroke: CHART_INK.axis }}
          tickLine={false}
          tickFormatter={(v: number) => v.toFixed(2)}
        />
        <Tooltip
          content={<ChartTooltip formatValue={(v) => v.toFixed(4)} />}
          cursor={{ fill: 'var(--gridline)', opacity: 0.4 }}
        />
        <Legend
          formatter={(value: string) => value.replaceAll('_', ' ')}
          wrapperStyle={{ fontSize: 12, color: 'var(--text-secondary)' }}
        />
        {modelOrder.map((model, index) => (
          <Bar
            key={model}
            dataKey={model}
            name={model.replaceAll('_', ' ')}
            fill={CATEGORICAL_SERIES[index % CATEGORICAL_SERIES.length]}
            radius={[4, 4, 0, 0]}
            maxBarSize={24}
            isAnimationActive={false}
          />
        ))}
      </BarChart>
    </ResponsiveContainer>
  )
}
