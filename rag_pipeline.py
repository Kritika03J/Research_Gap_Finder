
import os
import time
import uuid

import chromadb
from openai import OpenAI
from dotenv import load_dotenv

from prompts import paper_extraction_prompt, gap_prompt

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
EMBEDDING_MODEL = os.getenv(
    "OPENAI_EMBEDDING_MODEL",
    "text-embedding-3-small"
)


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


def get_embeddings(texts):
    """Generate embeddings for a list of text chunks."""

    start = time.perf_counter()

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )

    latency = time.perf_counter() - start

    embeddings = [
        item.embedding for item in response.data
    ]

    tokens = response.usage.total_tokens

    return embeddings, tokens, latency


def split_text(text, chunk_size=1200, overlap=200):
    """Split text into overlapping chunks."""

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])

        start += chunk_size - overlap

    return chunks


def format_prompt(prompt_template, information):

    if hasattr(prompt_template, "format"):
        formatted = prompt_template.format(
            paper_text=information,
            extracted_information=information
        )

        if hasattr(formatted, "to_string"):
            return formatted.to_string()

        return formatted

    return str(prompt_template).format(
        paper_text=information,
        extracted_information=information
    )


def analyze_rag(papers, top_k=4):
    """
    Vector RAG research gap identification.

    Input:
        papers = {
            "paper1.pdf": [
                {"page": 1, "text": "..."},
                {"page": 2, "text": "..."}
            ]
        }

    Output:
        Research gaps, retrieved evidence and metrics.
    """

    start_time = time.perf_counter()

    total_input_tokens = 0
    total_output_tokens = 0
    total_embedding_tokens = 0

    total_gpt_latency = 0
    total_embedding_latency = 0
    total_gpt_calls = 0
    total_embedding_calls = 0

    # STEP 1: Create chunks from all papers

    all_chunks = []
    all_metadata = []

    for paper_name, pages in papers.items():

        for page in pages:

            page_chunks = split_text(page["text"])

            for chunk in page_chunks:

                if not chunk.strip():
                    continue

                all_chunks.append(chunk)

                all_metadata.append({
                    "paper": paper_name,
                    "page": page["page"]
                })

    if not all_chunks:
        raise ValueError("No readable text found in the uploaded PDFs.")

    # STEP 2: Generate embeddings

    embeddings = []

    batch_size = 100

    for i in range(0, len(all_chunks), batch_size):

        batch = all_chunks[i:i + batch_size]

        batch_embeddings, tokens, latency = get_embeddings(batch)

        embeddings.extend(batch_embeddings)

        total_embedding_tokens += tokens
        total_embedding_latency += latency
        total_embedding_calls += 1

    # STEP 3: Store chunks in ChromaDB

    chroma_client = chromadb.Client()

    collection_name = "research_gaps_" + uuid.uuid4().hex[:12]

    collection = chroma_client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    ids = [
        f"chunk_{i}"
        for i in range(len(all_chunks))
    ]

    collection.add(
        ids=ids,
        documents=all_chunks,
        embeddings=embeddings,
        metadatas=all_metadata
    )

    # STEP 4: Retrieve relevant chunks

    queries = [
        "What are the research objectives and problems addressed?",
        "What methodologies, algorithms and models are used?",
        "What datasets are used and what are their limitations?",
        "What are the main experimental findings?",
        "What limitations and challenges are reported?",
        "What future research directions are suggested?",
        "What problems remain unresolved in this research area?"
    ]

    retrieved_evidence = []
    seen_chunks = set()

    for query in queries:

        query_embeddings, tokens, latency = get_embeddings([query])

        total_embedding_tokens += tokens
        total_embedding_latency += latency
        total_embedding_calls += 1

        results = collection.query(
            query_embeddings=query_embeddings,
            n_results=min(top_k, len(all_chunks)),
            include=["documents", "metadatas", "distances"]
        )

        for i in range(len(results["documents"][0])):

            chunk = results["documents"][0][i]
            metadata = results["metadatas"][0][i]

            unique_key = (
                metadata["paper"],
                metadata["page"],
                chunk
            )

            if unique_key in seen_chunks:
                continue

            seen_chunks.add(unique_key)

            retrieved_evidence.append({
                "paper": metadata["paper"],
                "page": metadata["page"],
                "text": chunk,
                "distance": results["distances"][0][i]
            })

    if not retrieved_evidence:
        raise ValueError("No relevant evidence was retrieved.")

    # STEP 5: Group retrieved evidence by paper

    evidence_by_paper = {}

    for item in retrieved_evidence:

        paper_name = item["paper"]

        if paper_name not in evidence_by_paper:
            evidence_by_paper[paper_name] = []

        evidence_by_paper[paper_name].append(
            f"""
            Page: {item["page"]}

            Evidence:
            {item["text"]}
            """
        )

    # STEP 6: Extract structured information from retrieved evidence

    extracted_papers = []

    for paper_name, evidence in evidence_by_paper.items():

        retrieved_text = "\n\n".join(evidence)

        extraction_input = f"""
        Paper name: {paper_name}

        The following text contains retrieved passages from this paper.
        Analyze only the supplied evidence.

        {retrieved_text}
        """

        extraction_prompt = format_prompt(
            paper_extraction_prompt,
            extraction_input
        )

        response = call_gpt(extraction_prompt)

        extracted_papers.append(
            f"""
            PAPER: {paper_name}

            EXTRACTED INFORMATION FROM RETRIEVED EVIDENCE:

            {response["text"]}
            """
        )

        total_input_tokens += response["input_tokens"]
        total_output_tokens += response["output_tokens"]
        total_gpt_latency += response["latency"]
        total_gpt_calls += 1

    # STEP 7: Identify research gaps

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

    # STEP 8: Clean up temporary ChromaDB collection

    chroma_client.delete_collection(collection_name)

    return {
        "result": final_response["text"],
        "extracted_information": extracted_papers,
        "retrieved_evidence": retrieved_evidence,
        "metrics": {
            "total_latency": round(total_latency, 2),
            "gpt_latency": round(total_gpt_latency, 2),
            "embedding_latency": round(total_embedding_latency, 2),
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "embedding_tokens": total_embedding_tokens,
            "throughput": round(throughput, 2),
            "gpt_calls": total_gpt_calls,
            "embedding_calls": total_embedding_calls,
            "retrieved_chunks": len(retrieved_evidence)
        }
    }
