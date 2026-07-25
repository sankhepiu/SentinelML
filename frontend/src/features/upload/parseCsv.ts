import Papa from 'papaparse'

export interface CsvParseResult {
  rows: Record<string, number>[]
  columns: string[]
  errors: string[]
}

/** Parses a feature CSV and validates its header against `expectedFeatures` exactly. */
export function parseFeatureCsv(text: string, expectedFeatures: string[]): CsvParseResult {
  const parsed = Papa.parse<Record<string, string>>(text, {
    header: true,
    skipEmptyLines: true,
  })

  const errors: string[] = parsed.errors.map(
    (error) => `Row ${error.row ?? '?'}: ${error.message}`,
  )
  const columns = parsed.meta.fields ?? []

  const expectedSet = new Set(expectedFeatures)
  const actualSet = new Set(columns)
  const missing = expectedFeatures.filter((feature) => !actualSet.has(feature))
  const unexpected = columns.filter((column) => !expectedSet.has(column))

  if (missing.length > 0) errors.push(`Missing columns: ${missing.join(', ')}`)
  if (unexpected.length > 0) errors.push(`Unexpected columns: ${unexpected.join(', ')}`)

  const rows: Record<string, number>[] = []
  if (missing.length === 0) {
    parsed.data.forEach((rawRow, index) => {
      const row: Record<string, number> = {}
      for (const feature of expectedFeatures) {
        const value = Number(rawRow[feature])
        if (!Number.isFinite(value)) {
          errors.push(`Row ${index + 2}: "${feature}" is not a valid number (${rawRow[feature]})`)
        }
        row[feature] = Number.isFinite(value) ? value : 0
      }
      rows.push(row)
    })
  }

  return { rows, columns, errors }
}
