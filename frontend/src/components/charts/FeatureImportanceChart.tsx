import { Bar, BarChart, CartesianGrid, Cell, LabelList, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { CHART_INK, sequentialBlueAt } from '../../lib/chartTokens'
import { ChartTooltip } from './ChartTooltip'

interface FeatureImportanceChartProps {
  importances: Record<string, number>
  topN?: number
}

/** Magnitude comparison across features -> sequential blue, one hue. */
export function FeatureImportanceChart({ importances, topN = 10 }: FeatureImportanceChartProps) {
  const sorted = Object.entries(importances)
    .sort((a, b) => b[1] - a[1])
    .slice(0, topN)
    .map(([name, value]) => ({ name, value }))
    .reverse()

  const maxValue = Math.max(...sorted.map((d) => d.value), 1e-9)

  return (
    <ResponsiveContainer width="100%" height={Math.max(220, sorted.length * 32)}>
      <BarChart data={sorted} layout="vertical" margin={{ top: 8, right: 40, bottom: 8, left: 8 }}>
        <CartesianGrid stroke={CHART_INK.gridline} horizontal={false} />
        <XAxis
          type="number"
          tick={{ fill: CHART_INK.muted, fontSize: 12 }}
          axisLine={{ stroke: CHART_INK.axis }}
          tickLine={false}
        />
        <YAxis
          type="category"
          dataKey="name"
          width={170}
          tick={{ fill: CHART_INK.secondary, fontSize: 12 }}
          axisLine={{ stroke: CHART_INK.axis }}
          tickLine={false}
        />
        <Tooltip
          content={<ChartTooltip valueLabel="Importance" formatValue={(v) => v.toFixed(3)} />}
          cursor={{ fill: 'var(--gridline)', opacity: 0.4 }}
        />
        <Bar dataKey="value" radius={[0, 4, 4, 0]} maxBarSize={20} isAnimationActive={false}>
          {sorted.map((entry) => (
            <Cell key={entry.name} fill={sequentialBlueAt(0.35 + 0.55 * (entry.value / maxValue))} />
          ))}
          <LabelList
            dataKey="value"
            position="right"
            formatter={(value: unknown) => (typeof value === 'number' ? value.toFixed(3) : '')}
            style={{ fill: 'var(--text-secondary)', fontSize: 11 }}
          />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}
