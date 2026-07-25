/**
 * Chart color tokens, mirroring the CSS custom properties in index.css.
 * Recharts renders to SVG in the browser, so `var(--series-1)` etc. resolve
 * correctly at paint time -- these strings pick up light/dark automatically,
 * no JS-side theme detection needed.
 */

// Fixed categorical order -- never reassign per-filter, never cycle past 8.
export const CATEGORICAL_SERIES = [
  'var(--series-1)',
  'var(--series-2)',
  'var(--series-3)',
  'var(--series-4)',
  'var(--series-5)',
  'var(--series-6)',
  'var(--series-7)',
  'var(--series-8)',
] as const

export const SEQUENTIAL_BLUE = [
  'var(--seq-100)',
  'var(--seq-150)',
  'var(--seq-200)',
  'var(--seq-250)',
  'var(--seq-300)',
  'var(--seq-350)',
  'var(--seq-400)',
  'var(--seq-450)',
  'var(--seq-500)',
  'var(--seq-550)',
  'var(--seq-600)',
  'var(--seq-650)',
  'var(--seq-700)',
] as const

export const STATUS_COLORS = {
  good: 'var(--status-good)',
  warning: 'var(--status-warning)',
  serious: 'var(--status-serious)',
  critical: 'var(--status-critical)',
} as const

export const CHART_INK = {
  primary: 'var(--text-primary)',
  secondary: 'var(--text-secondary)',
  muted: 'var(--text-muted)',
  gridline: 'var(--gridline)',
  axis: 'var(--axis)',
  surface: 'var(--surface-1)',
} as const

/** Interpolate the sequential blue ramp at `t` in [0, 1] -- 0 = lightest, 1 = darkest. */
export function sequentialBlueAt(t: number): string {
  const clamped = Math.min(1, Math.max(0, t))
  const index = Math.round(clamped * (SEQUENTIAL_BLUE.length - 1))
  return SEQUENTIAL_BLUE[index]
}
