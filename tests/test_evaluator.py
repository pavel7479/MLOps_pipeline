import numpy as np
import pytest

from src.training.evaluator import evaluate_predictions, select_best_model


def test_metrics_match_known_multiclass_result() -> None:
    actual = np.array(["SELL", "HOLD", "BUY", "BUY"])
    predicted = np.array(["SELL", "HOLD", "HOLD", "BUY"])
    result = evaluate_predictions("known", actual, predicted)
    assert result["accuracy"] == pytest.approx(0.75)
    assert result["macro_precision"] == pytest.approx(5 / 6)
    assert result["macro_recall"] == pytest.approx(5 / 6)
    assert result["macro_f1"] == pytest.approx(7 / 9)
    assert result["class_metrics"]["BUY"]["recall"] == pytest.approx(0.5)
    assert result["confusion_matrix"]["labels"] == ["SELL", "HOLD", "BUY"]
    assert result["confusion_matrix"]["matrix"] == [[1, 0, 0], [0, 1, 0], [0, 1, 1]]


def test_winner_selection_uses_requested_metric() -> None:
    results = [
        {"model": "first", "macro_f1": 0.42},
        {"model": "second", "macro_f1": 0.51},
        {"model": "third", "macro_f1": 0.48},
    ]
    assert select_best_model(results)["model"] == "second"
