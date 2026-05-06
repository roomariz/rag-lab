# ============================================================
# RAGAs + Ollama Evaluation Lab
# Compatible with:
#
# ragas==0.4.3
# openai==2.35.0
# python==3.12
# ============================================================

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time
import pandas as pd

from openai import OpenAI

from ragas import evaluate
from ragas.dataset_schema import SingleTurnSample
from ragas import EvaluationDataset

from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._answer_relevance import ResponseRelevancy
from ragas.metrics._context_precision import ContextPrecision
from ragas.metrics._context_recall import ContextRecall

from ragas.llms import llm_factory
from ragas.embeddings.base import BaseRagasEmbeddings

# ============================================================
# OLLAMA CONFIG
# ============================================================

OLLAMA_BASE_URL = "http://localhost:11434/v1"

LLM_MODEL = "mistral:latest"
EMBED_MODEL = "nomic-embed-text:latest"

# ============================================================
# RAW OPENAI CLIENT (OPTIONAL TESTING)
# ============================================================

client = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key="ollama",
)

# ============================================================
# CONNECTION TEST
# ============================================================

print("\n================================================")
print("OLLAMA CONNECTION TEST")
print("================================================\n")

response = client.chat.completions.create(
    model=LLM_MODEL,
    messages=[
        {
            "role": "user",
            "content": "Say connection OK in exactly two words."
        }
    ],
)

print("Response:")
print(response.choices[0].message.content)

# ============================================================
# LANGCHAIN LLM
# ============================================================

llm = llm_factory(
    LLM_MODEL,
    client=client,
)

# ============================================================
# LANGCHAIN EMBEDDINGS
# ============================================================

embed_client = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key="ollama",
)

class OllamaEmbeddings(BaseRagasEmbeddings):

    # ========================================================
    # SYNC METHODS
    # ========================================================

    def embed_query(self, text: str):

        response = embed_client.embeddings.create(
            model=EMBED_MODEL,
            input=[text],
        )

        return response.data[0].embedding

    def embed_documents(self, texts):

        response = embed_client.embeddings.create(
            model=EMBED_MODEL,
            input=texts,
        )

        return [
            item.embedding
            for item in response.data
        ]

    # ========================================================
    # ASYNC METHODS
    # ========================================================

    async def aembed_query(self, text: str):

        return self.embed_query(text)

    async def aembed_documents(self, texts):

        return self.embed_documents(texts)


embeddings = OllamaEmbeddings()

# ============================================================
# DATASET
# ============================================================

samples = [

    # ========================================================
    # SAMPLE 1
    # ========================================================

    SingleTurnSample(
        user_input="What is Docker used for?",

        retrieved_contexts=[
            (
                "Docker is an open platform for developing, "
                "shipping, and running applications using "
                "lightweight portable containers."
            )
        ],

        response=(
            "Docker is used to develop and run applications "
            "inside portable containers."
        ),

        reference=(
            "Docker runs applications inside containers."
        ),
    ),

    # ========================================================
    # SAMPLE 2
    # ========================================================

    SingleTurnSample(
        user_input="What is Kubernetes?",

        retrieved_contexts=[
            (
                "Kubernetes is an open-source container "
                "orchestration platform for automating "
                "deployment and scaling."
            )
        ],

        response=(
            "Kubernetes automates deployment and scaling "
            "of containerized applications."
        ),

        reference=(
            "Kubernetes is used for container orchestration."
        ),
    ),

    # ========================================================
    # SAMPLE 3 (HALLUCINATION)
    # ========================================================

    SingleTurnSample(
        user_input="What is SQL injection?",

        retrieved_contexts=[
            (
                "SQL injection is a vulnerability where "
                "attackers manipulate SQL queries using "
                "malicious input."
            )
        ],

        response=(
            "SQL injection manipulates SQL queries using "
            "malicious input and can permanently destroy "
            "computer hardware."
        ),

        reference=(
            "SQL injection manipulates database queries."
        ),
    ),
]

dataset = EvaluationDataset(samples=samples)

print("\n================================================")
print("DATASET")
print("================================================\n")

print(f"Samples loaded: {len(samples)}")

# ============================================================
# METRICS
# ============================================================

metrics = [

    # --------------------------------------------------------
    # Faithfulness
    #
    # Detects hallucinations where the response includes
    # claims not grounded in retrieved context.
    # --------------------------------------------------------

    Faithfulness(
        llm=llm,
    ),

    # --------------------------------------------------------
    # Answer Relevancy
    #
    # Checks whether the response actually answers
    # the user's question.
    # --------------------------------------------------------

    ResponseRelevancy(
        llm=llm,
        embeddings=embeddings,
        strictness=1,
    ),

    # --------------------------------------------------------
    # Context Precision
    #
    # Measures whether retrieved chunks were relevant
    # to the question.
    # --------------------------------------------------------

    ContextPrecision(
        llm=llm,
    ),

    # --------------------------------------------------------
    # Context Recall
    #
    # Checks whether retrieval captured all information
    # needed to answer correctly.
    # --------------------------------------------------------

    ContextRecall(
        llm=llm,
    ),
]

# ============================================================
# RUN EVALUATION
# ============================================================

print("\n================================================")
print("RUNNING EVALUATION")
print("================================================\n")

print(f"Metrics enabled: {len(metrics)}")
print(f"Samples: {len(samples)}")
print(
    f"Total evaluation tasks: "
    f"{len(metrics) * len(samples)}"
)
print()

start_time = time.perf_counter()

result = evaluate(
    dataset=dataset,
    metrics=metrics,
    llm=llm,
    embeddings=embeddings,
    show_progress=True,
)

end_time = time.perf_counter()
elapsed = end_time - start_time

print("\n================================================")
print("EXECUTION TIME")
print("================================================\n")
print(f"Total evaluation time: {elapsed:.2f} seconds")

# ============================================================
# RESULTS
# ============================================================

print("\n================================================")
print("AGGREGATE SCORES")
print("================================================\n")

print(result)

# ============================================================
# DATAFRAME
# ============================================================

df = result.to_pandas()

print("\n================================================")
print("PER SAMPLE RESULTS")
print("================================================\n")

cols = [
    "user_input",
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]

existing_cols = [
    c for c in cols
    if c in df.columns
]

print(
    df[existing_cols].to_string(index=False)
)

# ============================================================
# SAVE CSV
# ============================================================

output_file = "ragas_results.csv"

df.to_csv(output_file, index=False)

print("\n================================================")
print("CSV SAVED")
print("================================================\n")

print(f"Saved to: {output_file}")

# ============================================================
# HALLUCINATION CHECK
# ============================================================

print("\n================================================")
print("LOW FAITHFULNESS DETECTION")
print("================================================\n")

for idx, row in df.iterrows():

    faithfulness = row.get("faithfulness", 1)

    if faithfulness < 0.8:

        print(
            f"Potential hallucination detected:\n"
            f"Question: {row['user_input']}\n"
            f"Faithfulness: {faithfulness:.2f}\n"
        )

print("Evaluation complete.")