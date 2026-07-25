import { apiGet } from './client'

export interface ModelInfo {
  model_type: string
  model_version: string
  feature_names: string[]
  label_classes: string[]
  metrics: Record<string, number>
}

export interface EvaluationMetrics {
  accuracy: number
  precision_macro: number
  precision_weighted: number
  recall_macro: number
  recall_weighted: number
  f1_macro: number
  f1_weighted: number
  roc_auc_ovr_macro: number | null
  confusion_matrix: number[][]
  classification_report: Record<string, unknown>
}

export interface TrainingSummary {
  version: string
  best_model_type: string
  selection_metric: string
  trained_models: string[]
  skipped_models: Record<string, string>
  val_metrics: Record<string, EvaluationMetrics>
  test_metrics: EvaluationMetrics
  feature_importances: Record<string, Record<string, number>>
  class_distribution: Record<string, Record<string, number>>
  n_train_rows: number
  n_val_rows: number
  n_test_rows: number
  feature_columns: string[]
  label_mapping: Record<string, string>
  random_state: number
}

export function fetchModelInfo(): Promise<ModelInfo> {
  return apiGet<ModelInfo>('/model')
}

export function fetchTrainingSummary(): Promise<TrainingSummary> {
  return apiGet<TrainingSummary>('/model/training-summary')
}
