import pandas as pd

from src.benchmarks.evaluator import _aggregate_ragas_metrics


def test_aggregate_ragas_metrics_uses_dataframe_means():
    frame = pd.DataFrame(
        [
            {
                "faithfulness": 1.0,
                "answer_relevancy": 0.5,
                "context_precision": 0.25,
                "context_recall": 0.75,
            },
            {
                "faithfulness": 0.0,
                "answer_relevancy": 1.0,
                "context_precision": 0.75,
                "context_recall": 0.25,
            },
        ]
    )

    metrics = _aggregate_ragas_metrics(frame)

    assert metrics["faithfulness"] == 0.5
    assert metrics["answer_relevancy"] == 0.75
    assert metrics["context_precision"] == 0.5
    assert metrics["context_recall"] == 0.5
