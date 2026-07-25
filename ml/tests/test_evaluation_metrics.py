import numpy as np

from ml.evaluation.metrics import evaluate_predictions


def test_evaluate_predictions_perfect_predictions():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = y_true.copy()

    result = evaluate_predictions(
        y_true, y_pred, None, labels=[0, 1, 2], target_names=["a", "b", "c"]
    )

    assert result.accuracy == 1.0
    assert result.precision_macro == 1.0
    assert result.recall_macro == 1.0
    assert result.f1_macro == 1.0
    assert result.f1_weighted == 1.0


def test_evaluate_predictions_computes_known_accuracy():
    y_true = np.array([0, 0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1, 1])  # 4/5 correct

    result = evaluate_predictions(y_true, y_pred, None, labels=[0, 1], target_names=["a", "b"])

    assert result.accuracy == 0.8


def test_evaluate_predictions_confusion_matrix_shape_and_values():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])

    result = evaluate_predictions(y_true, y_pred, None, labels=[0, 1], target_names=["a", "b"])

    assert result.confusion_matrix == [[1, 1], [0, 2]]


def test_evaluate_predictions_classification_report_uses_target_names():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])

    result = evaluate_predictions(
        y_true, y_pred, None, labels=[0, 1], target_names=["benign", "attack"]
    )

    assert "benign" in result.classification_report
    assert "attack" in result.classification_report


def test_evaluate_predictions_roc_auc_none_without_probabilities():
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 1, 2])

    result = evaluate_predictions(
        y_true, y_pred, None, labels=[0, 1, 2], target_names=["a", "b", "c"]
    )

    assert result.roc_auc_ovr_macro is None


def test_evaluate_predictions_roc_auc_computed_with_probabilities():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = y_true.copy()
    y_proba = np.array(
        [
            [0.9, 0.05, 0.05],
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.05, 0.9, 0.05],
            [0.05, 0.05, 0.9],
            [0.1, 0.1, 0.8],
        ]
    )

    result = evaluate_predictions(
        y_true, y_pred, y_proba, labels=[0, 1, 2], target_names=["a", "b", "c"]
    )

    assert result.roc_auc_ovr_macro is not None
    assert 0.0 <= result.roc_auc_ovr_macro <= 1.0


def test_evaluate_predictions_to_dict_is_json_serializable():
    import json

    y_true = np.array([0, 1])
    y_pred = np.array([0, 1])

    result = evaluate_predictions(y_true, y_pred, None, labels=[0, 1], target_names=["a", "b"])

    json.dumps(result.to_dict())  # must not raise
