
import os
import time
from openai import OpenAI
from dotenv import load_dotenv

from prompts import paper_extraction_prompt, gap_prompt

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")


def call_gpt(prompt):
    """Call GPT and return response text and token usage."""

    start = time.perf_counter()

    response = client.responses.create(
        model=MODEL,
        input=prompt
    )

    latency = time.perf_counter() - start

    return {
        "text": response.output_text,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "latency": latency
    }


def split_text(text, chunk_size=12000, overlap=1000):
    """Split long paper text into overlapping chunks."""

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])

        start += chunk_size - overlap

    return chunks


def format_prompt(prompt_template, information):
    """Support both LangChain prompt templates and normal strings."""

    values = {
        "paper_text": information,
        "paper_information": information,
        "structured_information": information,
        "extracted_information": information
    }

    if hasattr(prompt_template, "format"):
        return prompt_template.format(**values)

    return str(prompt_template).format(**values)


def analyze_llm_only(papers):
    """
    LLM-only research gap identification.

    Input:
        papers = {
            "paper1.pdf": [
                {"page": 1, "text": "..."},
                {"page": 2, "text": "..."}
            ]
        }

    Output:
        Research gaps and evaluation metrics.
    """

    start_time = time.perf_counter()

    total_input_tokens = 0
    total_output_tokens = 0
    total_gpt_latency = 0
    total_gpt_calls = 0

    extracted_papers = []

    # STEP 1: Extract information from every paper

    for paper_name, pages in papers.items():

        paper_text = "\n".join(
            page["text"] for page in pages
        )

        chunks = split_text(paper_text)

        paper_extractions = []

        for chunk_number, chunk in enumerate(chunks, start=1):

            extraction_prompt = format_prompt(
                paper_extraction_prompt,
                chunk
            )

            response = call_gpt(extraction_prompt)

            paper_extractions.append(
                f"""
                Paper: {paper_name}
                Chunk: {chunk_number}

                {response["text"]}
                """
            )

            total_input_tokens += response["input_tokens"]
            total_output_tokens += response["output_tokens"]
            total_gpt_latency += response["latency"]
            total_gpt_calls += 1

        extracted_papers.append(
            f"""
            PAPER: {paper_name}

            EXTRACTED INFORMATION:
            {"".join(paper_extractions)}
            """
        )

    # STEP 2: Identify research gaps across all papers

    combined_information = "\n\n".join(extracted_papers)

    final_prompt = format_prompt(
        gap_prompt,
        combined_information
    )

    final_response = call_gpt(final_prompt)

    total_input_tokens += final_response["input_tokens"]
    total_output_tokens += final_response["output_tokens"]
    total_gpt_latency += final_response["latency"]
    total_gpt_calls += 1

    total_latency = time.perf_counter() - start_time

    throughput = (
        total_output_tokens / total_latency
        if total_latency > 0 else 0
    )

    return {
        "result": final_response["text"],
        "extracted_information": extracted_papers,
        "metrics": {
            "total_latency": round(total_latency, 2),
            "gpt_latency": round(total_gpt_latency, 2),
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "throughput": round(throughput, 2),
            "gpt_calls": total_gpt_calls
        }
    }
