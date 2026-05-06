try:  # pragma: no cover - optional visualization dependency
    from .plots import (
        plot_tsne_embeddings,
        plot_retrieval_neighborhood,
        plot_evaluation_trends,
        plot_latency_breakdown,
        plot_similarity_matrix,
        create_dashboard_charts,
    )
except ImportError:  # pragma: no cover - plotly is optional in this environment
    plot_tsne_embeddings = None
    plot_retrieval_neighborhood = None
    plot_evaluation_trends = None
    plot_latency_breakdown = None
    plot_similarity_matrix = None
    create_dashboard_charts = None

from .benchmark_charts import create_benchmark_analytics_charts

__all__ = [
    "plot_tsne_embeddings",
    "plot_retrieval_neighborhood",
    "plot_evaluation_trends",
    "plot_latency_breakdown",
    "plot_similarity_matrix",
    "create_dashboard_charts",
    "create_benchmark_analytics_charts",
]
