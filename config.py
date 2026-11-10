import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")

MODEL = os.getenv(
    "OPENAI_MODEL",
    "gpt-5.4-mini"
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "text-embedding-3-small"
)

if not API_KEY:
    raise ValueError("OPENAI_API_KEY is missing from .env")

client = OpenAI(api_key=API_KEY)


# Store metrics for the current analysis
metrics = {
    "input_tokens": 0,
    "output_tokens": 0,
    "embedding_tokens": 0,
    "gpt_api_latency": 0.0,
    "embedding_api_latency": 0.0,
    "gpt_calls": 0,
    "embedding_calls": 0
}


def reset_metrics():

    for key in metrics:
        metrics[key] = 0


def get_metrics():

    return metrics.copy()


# GPT text generation
def ask_gpt(prompt):

    import time

    start = time.perf_counter()

    response = client.responses.create(
        model=MODEL,
        instructions=(
            "You are an academic research analyst. "
            "Use only the supplied evidence."
        ),
        input=prompt,
        max_output_tokens=3000
    )

    elapsed = time.perf_counter() - start

    metrics["gpt_api_latency"] += elapsed
    metrics["gpt_calls"] += 1

    if response.usage:

        metrics["input_tokens"] += (
            response.usage.input_tokens
        )

        metrics["output_tokens"] += (
            response.usage.output_tokens
        )

    return response.output_text


# Generate embeddings
def generate_embeddings(texts):

    import time

    start = time.perf_counter()

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )

    elapsed = time.perf_counter() - start

    metrics["embedding_api_latency"] += elapsed
    metrics["embedding_calls"] += 1

    if response.usage:

        metrics["embedding_tokens"] += (
            response.usage.prompt_tokens
        )

    return [
        item.embedding
        for item in response.data
    ]