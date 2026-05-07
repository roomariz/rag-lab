import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

from ..benchmarks.timing import TIMING_FIELDS


def plot_tsne_embeddings(
    embeddings: np.ndarray,
    labels: Optional[List[str]] = None,
    title: str = "t-SNE Embedding Clusters",
) -> go.Figure:
    if len(embeddings) < 2:
        return go.Figure()

    n_components = 2 if len(embeddings) >= 2 else 1

    tsne = TSNE(
        n_components=n_components,
        random_state=42,
        perplexity=min(5, len(embeddings) - 1),
    )

    coords = tsne.fit_transform(embeddings)

    fig = px.scatter(
        x=coords[:, 0],
        y=coords[:, 1],
        labels=labels,
        title=title,
    )

    fig.update_traces(marker=dict(size=10, opacity=0.7))
    fig.update_layout(
        xaxis_title="t-SNE 1",
        yaxis_title="t-SNE 2",
    )

    return fig


def plot_retrieval_neighborhood(
    query_embedding: np.ndarray,
    retrieved_embeddings: np.ndarray,
    all_embeddings: np.ndarray,
    retrieved_indices: List[int],
) -> go.Figure:
    all_coords = np.vstack([query_embedding.reshape(1, -1), all_embeddings])

    if len(all_coords) >= 2:
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(all_coords) - 1))
        coords = tsne.fit_transform(all_coords)

    query_coord = coords[0]
    doc_coords = coords[1:]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=[query_coord[0]],
        y=[query_coord[1]],
        mode="markers",
        marker=dict(size=20, color="red", symbol="star"),
        name="Query",
    ))

    retrieved_coords = doc_coords[retrieved_indices]
    fig.add_trace(go.Scatter(
        x=retrieved_coords[:, 0],
        y=retrieved_coords[:, 1],
        mode="markers",
        marker=dict(size=12, color="blue", symbol="circle"),
        name="Retrieved",
    ))

    other_indices = [i for i in range(len(doc_coords)) if i not in retrieved_indices]
    other_coords = doc_coords[other_indices]
    if len(other_coords) > 0:
        fig.add_trace(go.Scatter(
            x=other_coords[:, 0],
            y=other_coords[:, 1],
            mode="markers",
            marker=dict(size=8, color="gray", symbol="x"),
            name="Not Retrieved",
        ))

    fig.update_layout(
        title="Retrieval Neighborhood",
        xaxis_title="t-SNE 1",
        yaxis_title="t-SNE 2",
    )

    return fig


def plot_evaluation_trends(
    results_df: pd.DataFrame,
    metrics: List[str],
    x_col: str = "timestamp",
) -> go.Figure:
    fig = go.Figure()

    for metric in metrics:
        if metric in results_df.columns:
            fig.add_trace(go.Scatter(
                y=results_df[metric],
                mode="lines+markers",
                name=metric,
            ))

    fig.update_layout(
        title="Evaluation Metrics Over Time",
        yaxis_title="Score",
        xaxis_title="Run",
    )

    return fig


def plot_latency_breakdown(
    retrieval_latency: float,
    generation_latency: float,
    embedding_latency: float = 0,
) -> go.Figure:
    labels = ["Retrieval", "Embedding", "Generation"]
    values = [retrieval_latency, embedding_latency, generation_latency]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
    )])

    fig.update_layout(title="Latency Breakdown")

    return fig


def plot_similarity_matrix(
    embeddings: np.ndarray,
    labels: Optional[List[str]] = None,
) -> go.Figure:
    if len(embeddings) < 2:
        return go.Figure()

    sim_matrix = cosine_similarity(embeddings)

    fig = go.Figure(data=go.Heatmap(
        z=sim_matrix,
        x=labels,
        y=labels,
        colorscale="Viridis",
    ))

    fig.update_layout(title="Similarity Matrix")

    return fig


def create_dashboard_charts(
    results: pd.DataFrame,
) -> dict:
    charts = {}

    if "faithfulness" in results.columns:
        charts["faithfulness"] = px.bar(
            results,
            x="user_input",
            y="faithfulness",
            title="Faithfulness by Query",
        )

    if "retrieval_latency" in results.columns:
        charts["latency"] = px.line(
            results,
            y="retrieval_latency",
            title="Retrieval Latency Trend",
        )

    if "score" in results.columns:
        charts["scores"] = px.histogram(
            results,
            x="score",
            title="Retrieval Score Distribution",
        )

    return charts


def create_benchmark_analytics_charts(results: pd.DataFrame) -> dict:
    charts = {}

    if results.empty:
        return charts

    metric_columns = [
        column
        for column in ["hit_rate", "retrieval_accuracy", "recall_at_k", "precision_at_k", "mrr", "retrieval_latency"]
        if column in results.columns
    ]
    if "top_k" in results.columns and metric_columns:
        grouped = results.groupby("top_k")[metric_columns].mean().reset_index()
        fig = go.Figure()
        for metric in metric_columns:
            fig.add_trace(go.Scatter(
                x=grouped["top_k"],
                y=grouped[metric],
                mode="lines+markers",
                name=metric,
            ))
        fig.update_layout(
            title="Benchmark Metrics by Top-K",
            xaxis_title="Top-K",
            yaxis_title="Score",
        )
        charts["top_k_metrics"] = fig

    if "chunk_overlap" in results.columns and "retrieval_latency" in results.columns:
        grouped = results.groupby("chunk_overlap")["retrieval_latency"].mean().reset_index()
        charts["chunk_overlap_latency"] = px.line(
            grouped,
            x="chunk_overlap",
            y="retrieval_latency",
            markers=True,
            title="Retrieval Latency by Chunk Overlap",
        )

    if "model" in results.columns and "mean_latency" in results.columns:
        charts["model_latency"] = px.bar(
            results,
            x="model",
            y="mean_latency",
            title="Embedding Latency by Model",
        )

    model_metric_columns = [
        column
        for column in [
            "document_embedding_latency",
            "query_embedding_latency",
            "mean_latency",
            "mean_recall_at_k",
            "mean_faithfulness",
            "mean_retrieval_latency",
            "mean_hit_rate",
            "mean_precision_at_k",
            "mean_mrr",
        ]
        if column in results.columns
    ]
    if "model" in results.columns and len(model_metric_columns) > 1:
        charts["model_quality"] = px.bar(
            results,
            x="model",
            y=model_metric_columns,
            title="Embedding Model Comparison",
            barmode="group",
        )

    timing_columns = [column for column in TIMING_FIELDS if column in results.columns]
    if timing_columns:
        timing_frame = results.copy()
        if "model" in timing_frame.columns:
            x_col = "model"
        elif "top_k" in timing_frame.columns:
            x_col = "top_k"
        elif "chunking_strategy" in timing_frame.columns:
            x_col = "chunking_strategy"
        else:
            timing_frame = timing_frame.reset_index().rename(columns={"index": "sample"})
            x_col = "sample"

        charts["timing_breakdown"] = px.bar(
            timing_frame,
            x=x_col,
            y=timing_columns,
            title="Timing Breakdown",
            barmode="group",
        )

    if "faithfulness" in results.columns:
        frame = results.copy()
        if "user_input" not in frame.columns:
            frame = frame.reset_index().rename(columns={"index": "sample"})
            x_col = "sample"
        else:
            x_col = "user_input"
        charts["faithfulness"] = px.bar(
            frame,
            x=x_col,
            y="faithfulness",
            title="Faithfulness by Sample",
        )

    if "score" in results.columns:
        charts["score_distribution"] = px.histogram(
            results,
            x="score",
            title="Score Distribution",
        )

    return charts
