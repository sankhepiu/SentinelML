import { PageHeader } from '../components/PageHeader'
import { Card } from '../components/Card'
import { MetricCard } from '../components/MetricCard'
import { LoadingState } from '../components/LoadingState'
import { ErrorState } from '../components/ErrorState'
import { ModelComparisonChart } from '../components/charts/ModelComparisonChart'
import { FeatureImportanceChart } from '../components/charts/FeatureImportanceChart'
import { ConfusionMatrixHeatmap } from '../components/charts/ConfusionMatrixHeatmap'
import { useModelInfo, useTrainingSummary } from '../queries/useModel'
import { formatMetricValue, formatPercent } from '../lib/format'
import { orderedLabelNames } from '../lib/labels'

export function ModelInfoPage() {
  const modelInfo = useModelInfo()
  const trainingSummary = useTrainingSummary()

  return (
    <div>
      <PageHeader
        title="Model Information"
        description="Metadata, evaluation metrics, and training details for the currently deployed model."
      />

      {modelInfo.isPending && <LoadingState label="Loading model info…" />}
      {modelInfo.isError && (
        <ErrorState error={modelInfo.error} onRetry={() => modelInfo.refetch()} />
      )}

      {modelInfo.data && (
        <>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <MetricCard
              label="Deployed model"
              value={modelInfo.data.model_type.replaceAll('_', ' ')}
              hint={`version ${modelInfo.data.model_version}`}
            />
            <MetricCard label="Accuracy (test)" value={formatPercent(modelInfo.data.metrics.accuracy)} />
            <MetricCard label="F1 (macro, test)" value={formatMetricValue(modelInfo.data.metrics.f1_macro)} />
            <MetricCard
              label="ROC-AUC (OvR macro, test)"
              value={formatMetricValue(modelInfo.data.metrics.roc_auc_ovr_macro)}
            />
          </div>

          <Card title="Feature set" className="mt-6">
            <p className="text-sm text-text-secondary">
              {modelInfo.data.feature_names.length} features · {modelInfo.data.label_classes.length}{' '}
              classes
            </p>
            <div className="mt-2 flex flex-wrap gap-1.5">
              {modelInfo.data.label_classes.map((label) => (
                <span
                  key={label}
                  className="rounded-full border border-black/10 px-2.5 py-1 text-xs text-text-secondary dark:border-white/10"
                >
                  {label}
                </span>
              ))}
            </div>
          </Card>
        </>
      )}

      {trainingSummary.isPending && (
        <div className="mt-6">
          <LoadingState label="Loading training summary…" />
        </div>
      )}
      {trainingSummary.isError && (
        <div className="mt-6">
          <ErrorState
            error={trainingSummary.error}
            onRetry={() => trainingSummary.refetch()}
            title="Couldn't load training summary"
          />
        </div>
      )}

      {trainingSummary.data && (
        <>
          <Card
            title="Candidate model comparison"
            description={`Validation-split metrics for every trained candidate. "${trainingSummary.data.best_model_type.replaceAll('_', ' ')}" was selected by ${trainingSummary.data.selection_metric.replaceAll('_', ' ')}.`}
            className="mt-6"
          >
            <ModelComparisonChart
              valMetrics={trainingSummary.data.val_metrics}
              modelOrder={trainingSummary.data.trained_models}
            />
            <div className="mt-4 overflow-x-auto">
              <table className="w-full min-w-max text-left text-sm">
                <thead>
                  <tr className="border-b border-black/10 text-text-muted dark:border-white/10">
                    <th className="py-2 pr-4 font-medium">Model</th>
                    <th className="py-2 pr-4 font-medium">Accuracy</th>
                    <th className="py-2 pr-4 font-medium">Precision</th>
                    <th className="py-2 pr-4 font-medium">Recall</th>
                    <th className="py-2 pr-4 font-medium">F1 (macro)</th>
                  </tr>
                </thead>
                <tbody>
                  {trainingSummary.data.trained_models.map((model) => {
                    const metrics = trainingSummary.data!.val_metrics[model]
                    const isBest = model === trainingSummary.data!.best_model_type
                    return (
                      <tr key={model} className="border-b border-black/5 dark:border-white/5">
                        <td className="py-2 pr-4 font-medium text-text-primary">
                          {model.replaceAll('_', ' ')}
                          {isBest && (
                            <span className="ml-2 rounded-full bg-[var(--status-good)]/10 px-2 py-0.5 text-xs font-medium text-[var(--status-good)]">
                              Selected
                            </span>
                          )}
                        </td>
                        <td className="py-2 pr-4 tabular-nums text-text-secondary">
                          {formatMetricValue(metrics.accuracy)}
                        </td>
                        <td className="py-2 pr-4 tabular-nums text-text-secondary">
                          {formatMetricValue(metrics.precision_macro)}
                        </td>
                        <td className="py-2 pr-4 tabular-nums text-text-secondary">
                          {formatMetricValue(metrics.recall_macro)}
                        </td>
                        <td className="py-2 pr-4 tabular-nums text-text-secondary">
                          {formatMetricValue(metrics.f1_macro)}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
            {Object.keys(trainingSummary.data.skipped_models).length > 0 && (
              <p className="mt-3 text-xs text-text-muted">
                Not trained in this environment:{' '}
                {Object.entries(trainingSummary.data.skipped_models)
                  .map(([model, reason]) => `${model.replaceAll('_', ' ')} (${reason})`)
                  .join(', ')}
              </p>
            )}
          </Card>

          <Card
            title="Feature importance"
            description={`Top features for "${trainingSummary.data.best_model_type.replaceAll('_', ' ')}".`}
            className="mt-6"
          >
            <FeatureImportanceChart
              importances={trainingSummary.data.feature_importances[trainingSummary.data.best_model_type]}
            />
          </Card>

          <Card
            title="Confusion matrix"
            description="Selected model's predictions on the held-out test split."
            className="mt-6"
          >
            <ConfusionMatrixHeatmap
              matrix={trainingSummary.data.test_metrics.confusion_matrix}
              labels={orderedLabelNames(trainingSummary.data.label_mapping)}
            />
          </Card>
        </>
      )}
    </div>
  )
}
