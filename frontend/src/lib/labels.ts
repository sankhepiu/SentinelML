/** Class names in label-encoded order (0, 1, 2, ...) -- matches confusion_matrix row/column order. */
export function orderedLabelNames(labelMapping: Record<string, string>): string[] {
  return Object.keys(labelMapping)
    .sort((a, b) => Number(a) - Number(b))
    .map((key) => labelMapping[key])
}
