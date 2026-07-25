import { PageHeader } from '../components/PageHeader'
import { Card } from '../components/Card'
import { MetricCard } from '../components/MetricCard'
import { LoadingState } from '../components/LoadingState'
import { ErrorState } from '../components/ErrorState'
import { StatusBadge } from '../components/StatusBadge'
import { ClassDistributionChart } from '../components/charts/ClassDistributionChart'
import { useHealth, useReadiness } from '../queries/useHealth'
import { useModelInfo, useTrainingSummary } from '../queries/useModel'
import { formatMetricValue, formatPercent } from '../lib/format'

export function OverviewPage() {
  const health = useHealth()
  const readiness = useReadiness()
  const modelInfo = useModelInfo()
  const trainingSummary = useTrainingSummary()

  return (
    <div>
      <PageHeader
        title="Overview"
        description="Live status of the SentinelML inference API and its currently deployed model."
      />

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Card title="API health">
          {health.isPending ? (
            <LoadingState label="Checking…" />
          ) : health.isError ? (
            <ErrorState error={health.error} onRetry={() => health.refetch()} />
          ) : (
            <StatusBadge tone="good" label={`${health.data.status} · ${health.data.environment}`} />
          )}
        </Card>

        <Card title="Model readiness">
          {readiness.isPending ? (
            <LoadingState label="Checking…" />
          ) : (
            <StatusBadge
              tone={readiness.data?.ready ? 'good' : 'warning'}
              label={readiness.data?.ready ? 'Ready' : (readiness.data?.detail ?? 'Not ready')}
            />
          )}
        </Card>

        {modelInfo.data && (
          <>
            <MetricCard
              label="Deployed model"
              value={modelInfo.data.model_type.replaceAll('_', ' ')}
              hint={`version ${modelInfo.data.model_version}`}
            />
            <MetricCard
              label="F1 (macro, test)"
              value={formatMetricValue(modelInfo.data.metrics.f1_macro)}
            />
          </>
        )}
      </div>

      {modelInfo.isPending && (
        <div className="mt-4">
          <LoadingState label="Loading model summary…" />
        </div>
      )}
      {modelInfo.isError && (
        <div className="mt-4">
          <ErrorState
            error={modelInfo.error}
            onRetry={() => modelInfo.refetch()}
            title="Couldn't load model info"
          />
        </div>
      )}

      {modelInfo.data && (
        <div className="mt-6 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <MetricCard label="Accuracy" value={formatPercent(modelInfo.data.metrics.accuracy)} />
          <MetricCard
            label="Precision (macro)"
            value={formatMetricValue(modelInfo.data.metrics.precision_macro)}
          />
          <MetricCard
            label="Recall (macro)"
            value={formatMetricValue(modelInfo.data.metrics.recall_macro)}
          />
          <MetricCard
            label="ROC-AUC (OvR macro)"
            value={formatMetricValue(modelInfo.data.metrics.roc_auc_ovr_macro)}
          />
        </div>
      )}

      <div className="mt-6">
        <Card title="Class distribution" description="Row counts per class in the training split.">
          {trainingSummary.isPending && <LoadingState label="Loading training summary…" />}
          {trainingSummary.isError && (
            <ErrorState error={trainingSummary.error} onRetry={() => trainingSummary.refetch()} />
          )}
          {trainingSummary.data && (
            <ClassDistributionChart distribution={trainingSummary.data.class_distribution.train} />
          )}
        </Card>
      </div>
    </div>
  )
}
