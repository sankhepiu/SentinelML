import { useState } from 'react'
import { Link } from 'react-router-dom'
import { PageHeader } from '../components/PageHeader'
import { Card } from '../components/Card'
import { LoadingState } from '../components/LoadingState'
import { ErrorState } from '../components/ErrorState'
import { useModelInfo } from '../queries/useModel'
import { usePredictBatch } from '../queries/usePrediction'
import { parseFeatureCsv } from '../features/upload/parseCsv'
import { BatchResultsTable } from '../features/results/BatchResultsTable'
import { classBreakdown } from '../features/results/batchSummary'
import { ClassDistributionChart } from '../components/charts/ClassDistributionChart'
import { appendHistoryEntry } from '../lib/predictionHistory'

export function BatchPredictionPage() {
  const modelInfo = useModelInfo()
  const predictBatch = usePredictBatch()
  const [fileName, setFileName] = useState<string | null>(null)
  const [parseErrors, setParseErrors] = useState<string[]>([])
  const [rows, setRows] = useState<Record<string, number>[]>([])

  if (modelInfo.isPending) return <LoadingState label="Loading feature set…" />
  if (modelInfo.isError)
    return <ErrorState error={modelInfo.error} onRetry={() => modelInfo.refetch()} />
  if (!modelInfo.data) return null

  const { feature_names: featureNames } = modelInfo.data

  const handleFile = async (file: File) => {
    setFileName(file.name)
    predictBatch.reset()
    const text = await file.text()
    const result = parseFeatureCsv(text, featureNames)
    setParseErrors(result.errors)
    setRows(result.rows)
  }

  const handleSubmit = () => {
    predictBatch.mutate(rows, {
      onSuccess: (result) => {
        appendHistoryEntry({
          kind: 'batch',
          model_version: result.predictions[0]?.model_version ?? 'unknown',
          count: result.count,
          class_breakdown: classBreakdown(result.predictions),
        })
      },
    })
  }

  return (
    <div>
      <PageHeader
        title="Batch Prediction"
        description="Upload a CSV of network flows to predict classes for every row."
      />

      <Card title="Upload CSV">
        <p className="mb-3 text-xs text-text-muted">
          Expected columns: exactly the {featureNames.length} feature names from{' '}
          <Link to="/model" className="underline">
            Model Information
          </Link>
          . Header row required.
        </p>
        <input
          type="file"
          accept=".csv,text/csv"
          onChange={(event) => {
            const file = event.target.files?.[0]
            if (file) void handleFile(file)
          }}
          className="block w-full text-sm text-text-secondary file:mr-3 file:rounded-md file:border-0 file:bg-[var(--series-1)] file:px-3 file:py-1.5 file:text-sm file:font-medium file:text-white"
        />

        {fileName && (
          <p className="mt-3 text-sm text-text-secondary">
            {fileName} — {rows.length} row{rows.length === 1 ? '' : 's'} parsed
          </p>
        )}

        {parseErrors.length > 0 && (
          <div className="mt-3 rounded-md border border-[var(--status-critical)]/30 bg-[var(--status-critical)]/5 p-3 text-xs text-[var(--status-critical)]">
            <ul className="list-disc pl-4">
              {parseErrors.slice(0, 10).map((error, index) => (
                <li key={index}>{error}</li>
              ))}
              {parseErrors.length > 10 && <li>…and {parseErrors.length - 10} more</li>}
            </ul>
          </div>
        )}

        <button
          type="button"
          onClick={handleSubmit}
          disabled={rows.length === 0 || parseErrors.length > 0 || predictBatch.isPending}
          className="mt-4 rounded-md bg-[var(--series-1)] px-4 py-2 text-sm font-medium text-white hover:opacity-90 disabled:opacity-50"
        >
          {predictBatch.isPending
            ? 'Predicting…'
            : `Predict ${rows.length} row${rows.length === 1 ? '' : 's'}`}
        </button>
      </Card>

      {predictBatch.isError && (
        <div className="mt-6">
          <ErrorState error={predictBatch.error} title="Batch prediction failed" />
        </div>
      )}

      {predictBatch.isSuccess && (
        <div className="mt-6 grid grid-cols-1 gap-6 lg:grid-cols-2">
          <Card title="Predicted class breakdown">
            <ClassDistributionChart distribution={classBreakdown(predictBatch.data.predictions)} />
          </Card>
          <Card title={`Results (${predictBatch.data.count})`}>
            <BatchResultsTable predictions={predictBatch.data.predictions} />
          </Card>
        </div>
      )}
    </div>
  )
}
