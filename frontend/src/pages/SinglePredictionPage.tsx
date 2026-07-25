import { useEffect, useState, type FormEvent } from 'react'
import { PageHeader } from '../components/PageHeader'
import { Card } from '../components/Card'
import { LoadingState } from '../components/LoadingState'
import { ErrorState } from '../components/ErrorState'
import { useModelInfo } from '../queries/useModel'
import { usePredict } from '../queries/usePrediction'
import { PredictionResultCard } from '../features/results/PredictionResultCard'
import { appendHistoryEntry } from '../lib/predictionHistory'

export function SinglePredictionPage() {
  const modelInfo = useModelInfo()
  const predict = usePredict()
  const [values, setValues] = useState<Record<string, string>>({})

  useEffect(() => {
    if (modelInfo.data && Object.keys(values).length === 0) {
      setValues(Object.fromEntries(modelInfo.data.feature_names.map((name) => [name, '0'])))
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [modelInfo.data])

  if (modelInfo.isPending) return <LoadingState label="Loading feature set…" />
  if (modelInfo.isError)
    return <ErrorState error={modelInfo.error} onRetry={() => modelInfo.refetch()} />
  if (!modelInfo.data) return null

  const featureNames = modelInfo.data.feature_names

  const handleChange = (name: string, value: string) => {
    setValues((prev) => ({ ...prev, [name]: value }))
  }

  const handleRandomize = () => {
    setValues(Object.fromEntries(featureNames.map((name) => [name, (Math.random() * 1000).toFixed(2)])))
  }

  const handleReset = () => {
    setValues(Object.fromEntries(featureNames.map((name) => [name, '0'])))
  }

  const handleSubmit = (event: FormEvent) => {
    event.preventDefault()
    const features: Record<string, number> = {}
    for (const name of featureNames) {
      const parsed = Number(values[name])
      features[name] = Number.isFinite(parsed) ? parsed : 0
    }
    predict.mutate(features, {
      onSuccess: (result) => {
        appendHistoryEntry({
          kind: 'single',
          model_version: result.model_version,
          predicted_class: result.predicted_class,
          confidence: result.confidence,
        })
      },
    })
  }

  return (
    <div>
      <PageHeader
        title="Single Prediction"
        description="Enter a value for every feature the model expects, then predict a single flow's class."
      />

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <Card title="Feature values">
          <form onSubmit={handleSubmit}>
            <div className="mb-4 flex gap-2">
              <button
                type="button"
                onClick={handleRandomize}
                className="rounded-md border border-black/10 px-3 py-1.5 text-sm font-medium text-text-primary hover:bg-black/5 dark:border-white/10 dark:hover:bg-white/5"
              >
                Randomize values
              </button>
              <button
                type="button"
                onClick={handleReset}
                className="rounded-md border border-black/10 px-3 py-1.5 text-sm font-medium text-text-primary hover:bg-black/5 dark:border-white/10 dark:hover:bg-white/5"
              >
                Reset to zero
              </button>
            </div>

            <div className="grid max-h-[28rem] grid-cols-1 gap-3 overflow-y-auto pr-1 sm:grid-cols-2">
              {featureNames.map((name) => (
                <label key={name} className="text-xs text-text-secondary">
                  <span className="block truncate" title={name}>
                    {name}
                  </span>
                  <input
                    type="number"
                    step="any"
                    value={values[name] ?? ''}
                    onChange={(event) => handleChange(name, event.target.value)}
                    className="mt-1 w-full rounded-md border border-black/10 bg-transparent px-2 py-1.5 text-sm text-text-primary focus:border-[var(--series-1)] focus:outline-none dark:border-white/10"
                  />
                </label>
              ))}
            </div>

            <button
              type="submit"
              disabled={predict.isPending}
              className="mt-4 w-full rounded-md bg-[var(--series-1)] px-4 py-2 text-sm font-medium text-white hover:opacity-90 disabled:opacity-50"
            >
              {predict.isPending ? 'Predicting…' : 'Predict'}
            </button>
          </form>
        </Card>

        <Card title="Result">
          {predict.status === 'idle' && (
            <p className="text-sm text-text-muted">Submit the form to see a prediction.</p>
          )}
          {predict.isPending && <LoadingState label="Running prediction…" />}
          {predict.isError && <ErrorState error={predict.error} title="Prediction failed" />}
          {predict.isSuccess && (
            <PredictionResultCard
              predictedClass={predict.data.predicted_class}
              confidence={predict.data.confidence}
              classProbabilities={predict.data.class_probabilities}
              modelVersion={predict.data.model_version}
            />
          )}
        </Card>
      </div>
    </div>
  )
}
