import { Bar, BarChart, CartesianGrid, Cell, LabelList, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { CHART_INK, sequentialBlueAt } from '../../lib/chartTokens'
import { ChartTooltip } from './ChartTooltip'

interface ClassDistributionChartProps {
  distribution: Record<string, number>
}

/**
 * Row counts per class -> sequential blue (magnitude, not identity).
 * CICIDS2017 classes span orders of magnitude (BENIGN in the tens of
 * thousands, Heartbleed in single digits), so a linear axis would render
 * everything but the top one or two classes as invisible slivers -- log
 * scale is what makes the rest of the distribution actually readable.
 */
export function ClassDistributionChart({ distribution }: ClassDistributionChartProps) {
  const sorted = Object.entries(distribution)
    .sort((a, b) => b[1] - a[1])
    .map(([name, value]) => ({ name, value }))
    .reverse()

  const maxValue = Math.max(...sorted.map((d) => d.value), 1)

  return (
    <ResponsiveContainer width="100%" height={Math.max(220, sorted.length * 34)}>
      <BarChart data={sorted} layout="vertical" margin={{ top: 8, right: 56, bottom: 8, left: 8 }}>
        <CartesianGrid stroke={CHART_INK.gridline} horizontal={false} />
        <XAxis
          type="number"
          scale="log"
          domain={['auto', 'auto']}
          allowDataOverflow
          tick={{ fill: CHART_INK.muted, fontSize: 12 }}
          axisLine={{ stroke: CHART_INK.axis }}
          tickLine={false}
          tickFormatter={(v: number) => v.toLocaleString()}
        />
        <YAxis
          type="category"
          dataKey="name"
          width={150}
          tick={{ fill: CHART_INK.secondary, fontSize: 12 }}
          axisLine={{ stroke: CHART_INK.axis }}
          tickLine={false}
        />
        <Tooltip
          content={<ChartTooltip valueLabel="Rows" formatValue={(v) => v.toLocaleString()} />}
          cursor={{ fill: 'var(--gridline)', opacity: 0.4 }}
        />
        <Bar dataKey="value" radius={[0, 4, 4, 0]} maxBarSize={20} isAnimationActive={false}>
          {sorted.map((entry) => (
            <Cell key={entry.name} fill={sequentialBlueAt(0.35 + 0.55 * (entry.value / maxValue))} />
          ))}
          <LabelList
            dataKey="value"
            position="right"
            formatter={(value: unknown) => (typeof value === 'number' ? value.toLocaleString() : '')}
            style={{ fill: 'var(--text-secondary)', fontSize: 11 }}
          />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}
