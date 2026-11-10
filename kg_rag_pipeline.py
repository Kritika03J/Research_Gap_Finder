
import time
import uuid
from collections import defaultdict

import chromadb
import pandas as pd

from rag_pipeline import (
    split_text,
    get_embeddings,
    call_gpt,
    format_prompt
)

from prompts import (
    paper_extraction_prompt,
    gap_prompt
)

from multiple_paper_ner import process_paper_pages



from knowledge_graph.graph_builder import (
    build_knowledge_graph,
    get_graph_summary
)

from knowledge_graph.graph_retriever import retrieve_graph_evidence
# ============================================================
# CONFIGURATION
# ============================================================

RETRIEVAL_QUERIES = [
    "What are the research objectives and problems addressed?",
    "Which methods, algorithms, and models are used?",
    "What datasets and materials are used?",
    "What are the main findings and performance results?",
    "What limitations and challenges are explicitly reported?",
    "What future research directions are explicitly suggested?",
    "What unresolved problems are discussed by the authors?"
]

# Maximum evidence retained per paper after ranking.
MAX_VECTOR_EVIDENCE_PER_PAPER = 10
MAX_GRAPH_EVIDENCE_PER_PAPER = 8

# Number of vector chunks retrieved for each query.
DEFAULT_TOP_K = 4

# Embedding batch size.
EMBEDDING_BATCH_SIZE = 100


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _render_prompt(template, values):
    """
    Formats a prompt using the existing format_prompt function.

    Multiple common variable names are supplied so that the
    existing prompt templates can be reused.
    """

    try:
        return format_prompt(template, **values)

    except (TypeError, KeyError, ValueError):
        pass

    try:
        return format_prompt(template, values)

    except (TypeError, KeyError, ValueError):
        pass

    class SafeDict(dict):
        def __missing__(self, key):
            return ""

    try:
        return template.format_map(SafeDict(values))

    except Exception:
        return str(template)


def _extract_llm_output(response):
    """
    Handles common return formats from call_gpt().

    Supported examples:
    - "response text"
    - (response text, input tokens, output tokens)
    - (response text, usage dictionary)
    - {"text": ..., "input_tokens": ..., "output_tokens": ...}
    """

    result = {
        "text": "",
        "input_tokens": 0,
        "output_tokens": 0
    }

    if isinstance(response, str):
        result["text"] = response
        return result

    if isinstance(response, dict):
        result["text"] = (
            response.get("text")
            or response.get("content")
            or response.get("response")
            or response.get("output")
            or ""
        )

        result["input_tokens"] = (
            response.get("input_tokens")
            or response.get("prompt_tokens")
            or 0
        )

        result["output_tokens"] = (
            response.get("output_tokens")
            or response.get("completion_tokens")
            or 0
        )

        usage = response.get("usage", {})

        if isinstance(usage, dict):
            result["input_tokens"] = (
                result["input_tokens"]
                or usage.get("input_tokens")
                or usage.get("prompt_tokens")
                or 0
            )

            result["output_tokens"] = (
                result["output_tokens"]
                or usage.get("output_tokens")
                or usage.get("completion_tokens")
                or 0
            )

        return result

    if isinstance(response, (tuple, list)) and len(response) > 0:
        result["text"] = response[0]

        if len(response) > 1:
            second = response[1]

            if isinstance(second, dict):
                result["input_tokens"] = (
                    second.get("input_tokens")
                    or second.get("prompt_tokens")
                    or 0
                )

                result["output_tokens"] = (
                    second.get("output_tokens")
                    or second.get("completion_tokens")
                    or 0
                )

            elif isinstance(second, (int, float)):
                result["input_tokens"] = second

        if len(response) > 2 and isinstance(response[2], (int, float)):
            result["output_tokens"] = response[2]

        return result

    # Handle response objects with a content attribute.
    if hasattr(response, "content"):
        result["text"] = response.content
        return result

    result["text"] = str(response)

    return result


def _extract_embeddings(response):
    """
    Extracts embeddings from common get_embeddings() return formats.
    """

    if isinstance(response, dict):
        if "embeddings" in response:
            return response["embeddings"]

        if "data" in response:
            return response["data"]

    if isinstance(response, tuple):
        return response[0]

    return response


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _get_page_number(item):
    """
    Gets page number from an evidence item.
    """

    return (
        item.get("page")
        or item.get("subject_page")
        or item.get("object_page")
        or "Unknown"
    )


def _deduplicate_evidence(items):
    """
    Removes duplicate evidence while preserving insertion order.
    """

    seen = set()
    unique_items = []

    for item in items:
        key = (
            item.get("paper"),
            item.get("page"),
            item.get("text", "").strip()
        )

        if key in seen:
            continue

        seen.add(key)
        unique_items.append(item)

    return unique_items


# ============================================================
# STEP 1: EXTRACT NER AND RELATIONS FROM ALL PAPERS
# ============================================================

def extract_paper_relations(papers):
    """
    Runs the NER and relation extraction pipeline on every paper.

    Expected input:

    {
        "paper1.pdf": [
            {"page": 1, "text": "..."},
            {"page": 2, "text": "..."}
        ]
    }
    """

    relations_by_paper = {}
    ner_statistics = {}

    for paper_name, pages in papers.items():

        print(f"\nProcessing NER: {paper_name}")

        try:
            relations_df, ner_stats = process_paper_pages(pages)

            if relations_df is None:
                relations_df = pd.DataFrame()

            relations_by_paper[paper_name] = relations_df
            ner_statistics[paper_name] = ner_stats or {}

            print(
                f"Extracted {len(relations_df)} relations "
                f"from {paper_name}"
            )

        except Exception as e:
            print(f"NER error for {paper_name}: {e}")

            relations_by_paper[paper_name] = pd.DataFrame()

            ner_statistics[paper_name] = {
                "error": str(e)
            }

    return relations_by_paper, ner_statistics


# ============================================================
# STEP 2: BUILD KNOWLEDGE GRAPH
# ============================================================

def create_knowledge_graph(relations_by_paper):
    """
    Builds a cross-paper knowledge graph using extracted relations.
    """

    print("\nBuilding knowledge graph...")

    graph = build_knowledge_graph(relations_by_paper)

    try:
        graph_stats = get_graph_summary(graph)

    except Exception:
        graph_stats = {
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges()
        }

    print("Knowledge graph created.")
    print(f"Nodes: {graph.number_of_nodes()}")
    print(f"Edges: {graph.number_of_edges()}")

    return graph, graph_stats


# ============================================================
# STEP 3: SPLIT PAPER TEXT INTO VECTOR CHUNKS
# ============================================================

def create_chunks(papers):
    """
    Splits all paper pages into chunks.

    Each chunk retains its source paper and page number.
    """

    all_chunks = []
    all_metadata = []

    for paper_name, pages in papers.items():

        for page_item in pages:

            page_number = page_item.get("page", "Unknown")
            page_text = page_item.get("text", "")

            if not page_text or not page_text.strip():
                continue

            chunks = split_text(page_text)

            for chunk in chunks:

                if not chunk or not chunk.strip():
                    continue

                all_chunks.append(chunk)

                all_metadata.append({
                    "paper": paper_name,
                    "page": str(page_number)
                })

    print(f"\nTotal vector chunks: {len(all_chunks)}")

    return all_chunks, all_metadata


# ============================================================
# STEP 4: GENERATE EMBEDDINGS
# ============================================================

def generate_embeddings(all_chunks):
    """
    Generates embeddings in batches to avoid oversized API requests.
    """

    all_embeddings = []

    total_embedding_time = 0.0
    embedding_calls = 0

    for start in range(0, len(all_chunks), EMBEDDING_BATCH_SIZE):

        batch = all_chunks[
            start:start + EMBEDDING_BATCH_SIZE
        ]

        batch_start = time.perf_counter()

        response = get_embeddings(batch)

        batch_embeddings = _extract_embeddings(response)

        batch_time = time.perf_counter() - batch_start

        total_embedding_time += batch_time
        embedding_calls += 1

        if batch_embeddings is None:
            raise ValueError(
                "get_embeddings() returned no embeddings."
            )

        all_embeddings.extend(batch_embeddings)

        print(
            f"Embedded {min(start + len(batch), len(all_chunks))}"
            f"/{len(all_chunks)} chunks"
        )

    if len(all_embeddings) != len(all_chunks):
        raise ValueError(
            "Number of embeddings does not match number of chunks. "
            f"Chunks: {len(all_chunks)}, "
            f"Embeddings: {len(all_embeddings)}"
        )

    return (
        all_embeddings,
        total_embedding_time,
        embedding_calls
    )


# ============================================================
# STEP 5: CREATE TEMPORARY CHROMA COLLECTION
# ============================================================

def create_vector_collection(
    all_chunks,
    all_metadata,
    all_embeddings
):
    """
    Creates a temporary Chroma collection for the current analysis.
    """

    client = chromadb.Client()

    collection_name = f"kg_rag_{uuid.uuid4().hex[:12]}"

    collection = client.create_collection(
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
        metadatas=all_metadata,
        embeddings=all_embeddings
    )

    return client, collection, collection_name


# ============================================================
# STEP 6: VECTOR RETRIEVAL
# ============================================================

def retrieve_vector_evidence(
    collection,
    top_k=DEFAULT_TOP_K
):
    """
    Retrieves relevant chunks using the predefined research queries.

    Lower cosine distance indicates a closer vector match.
    """

    vector_evidence = []

    for query in RETRIEVAL_QUERIES:

        query_start = time.perf_counter()

        query_embedding_response = get_embeddings([query])

        query_embedding = _extract_embeddings(
            query_embedding_response
        )

        if not query_embedding:
            continue

        results = collection.query(
            query_embeddings=[query_embedding[0]],
            n_results=min(
                top_k,
                collection.count()
            ),
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

        query_time = time.perf_counter() - query_start

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for i in range(len(documents)):

            metadata = metadatas[i] or {}

            vector_evidence.append({
                "source": "VECTOR",
                "query": query,
                "paper": metadata.get("paper", "Unknown"),
                "page": metadata.get("page", "Unknown"),
                "text": documents[i],
                "distance": _safe_float(
                    distances[i] if i < len(distances) else None,
                    default=999.0
                ),
                "query_time": query_time
            })

    return _deduplicate_evidence(vector_evidence)


# ============================================================
# STEP 7: KNOWLEDGE GRAPH RETRIEVAL
# ============================================================

def retrieve_kg_evidence(graph):
    """
    Retrieves graph relationships using the same research queries.

    IMPORTANT:
    Graph relationships are treated as discovery signals.
    They are not automatically considered scientific evidence.
    """

    graph_evidence = []

    for query in RETRIEVAL_QUERIES:

        try:
            retrieved = retrieve_graph_evidence(
                graph,
                query,
                max_results=20
            )

        except Exception as e:
            print(f"Graph retrieval error for query '{query}': {e}")
            continue

        for item in retrieved:

            context = (
                item.get("context")
                or item.get("text")
                or ""
            ).strip()

            # Do not use a graph relation without supporting context.
            if not context:
                continue

            graph_evidence.append({
                "source": "KNOWLEDGE_GRAPH",
                "query": query,
                "paper": item.get("paper", "Unknown"),
                "page": _get_page_number(item),
                "subject": item.get("subject", ""),
                "relation": item.get("relation", ""),
                "object": item.get("object", ""),
                "confidence": _safe_float(
                    item.get("confidence"),
                    default=0.0
                ),
                "retrieval_score": _safe_float(
                    item.get("retrieval_score"),
                    default=0.0
                ),
                "text": context
            })

    return _deduplicate_evidence(graph_evidence)


# ============================================================
# STEP 8: RANK AND GROUP EVIDENCE BY PAPER
# ============================================================

def group_evidence_by_paper(
    papers,
    vector_evidence,
    graph_evidence
):
    """
    Groups evidence by its original paper.

    Vector evidence:
        Ranked by cosine distance (lower is better).

    Graph evidence:
        Ranked by retrieval relevance and relation confidence.

    Only the highest-ranked evidence is retained per paper.
    """

    raw_evidence_by_paper = {
        paper_name: []
        for paper_name in papers
    }

    vector_by_paper = defaultdict(list)
    graph_by_paper = defaultdict(list)

    # Group vector evidence.
    for item in vector_evidence:

        paper_name = item.get("paper")

        if paper_name not in raw_evidence_by_paper:
            continue

        vector_by_paper[paper_name].append(item)

    # Group graph evidence.
    for item in graph_evidence:

        paper_name = item.get("paper")

        if paper_name not in raw_evidence_by_paper:
            continue

        graph_by_paper[paper_name].append(item)

    vector_counter = 1
    graph_counter = 1

    for paper_name in papers:

        # Lower cosine distance is better.
        ranked_vector = sorted(
            vector_by_paper[paper_name],
            key=lambda x: x.get("distance", 999.0)
        )

        # Higher retrieval score and confidence are better.
        ranked_graph = sorted(
            graph_by_paper[paper_name],
            key=lambda x: (
                x.get("retrieval_score", 0.0),
                x.get("confidence", 0.0)
            ),
            reverse=True
        )

        selected_vector = ranked_vector[
            :MAX_VECTOR_EVIDENCE_PER_PAPER
        ]

        selected_graph = ranked_graph[
            :MAX_GRAPH_EVIDENCE_PER_PAPER
        ]

        # Preserve original source evidence and its provenance.
        for item in selected_vector:

            evidence_item = dict(item)

            evidence_item["evidence_id"] = (
                f"V{vector_counter}"
            )

            vector_counter += 1

            raw_evidence_by_paper[paper_name].append(
                evidence_item
            )

        for item in selected_graph:

            evidence_item = dict(item)

            evidence_item["evidence_id"] = (
                f"G{graph_counter}"
            )

            graph_counter += 1

            raw_evidence_by_paper[paper_name].append(
                evidence_item
            )

    return raw_evidence_by_paper


# ============================================================
# STEP 9: FORMAT SOURCE EVIDENCE FOR LLM EXTRACTION
# ============================================================

def format_paper_evidence(paper_name, evidence):
    """
    Formats the original retrieved passages for per-paper analysis.
    """

    formatted = []

    for item in evidence:

        source = item.get("source", "UNKNOWN")
        evidence_id = item.get("evidence_id", "N/A")
        page = item.get("page", "Unknown")
        text = item.get("text", "")

        section = [
            f"Evidence ID: {evidence_id}",
            f"Source type: {source}",
            f"Paper: {paper_name}",
            f"Page: {page}"
        ]

        if source == "KNOWLEDGE_GRAPH":

            section.extend([
                "Graph relationship:",
                f"Subject: {item.get('subject', '')}",
                f"Relation: {item.get('relation', '')}",
                f"Object: {item.get('object', '')}",
                f"Relation extraction confidence: "
                f"{item.get('confidence', 0.0)}",
                "Supporting context:"
            ])

        else:
            section.append("Retrieved source passage:")

        section.append(text)

        formatted.append("\n".join(section))

    if not formatted:
        return "No relevant evidence was retrieved for this paper."

    return "\n\n".join(formatted)


# ============================================================
# STEP 10: EXTRACT STRUCTURED INFORMATION PER PAPER
# ============================================================

def extract_information_per_paper(
    papers,
    raw_evidence_by_paper
):
    """
    Performs per-paper extraction using the original retrieved
    source passages.

    The extraction stage must distinguish:
    - Explicitly reported information
    - Reasonable inferences
    - Information not established by retrieved evidence
    """

    extracted_information = {}

    extraction_time = 0.0
    extraction_input_tokens = 0
    extraction_output_tokens = 0
    extraction_calls = 0

    for paper_name, evidence in raw_evidence_by_paper.items():

        print(f"\nExtracting information: {paper_name}")

        evidence_text = format_paper_evidence(
            paper_name,
            evidence
        )

        instructions = f"""
You are analyzing one research paper for a research gap
identification system.

Paper name:
{paper_name}

Retrieved source evidence:
{evidence_text}

INSTRUCTIONS:

1. Extract only information supported by the supplied evidence.

2. Identify the following wherever available:
   - Research objective
   - Problem addressed
   - Methods and algorithms
   - Dataset and sample size
   - Experimental setup
   - Evaluation metrics
   - Main findings and numerical results
   - Explicit limitations
   - Explicit future research directions

3. Clearly distinguish:
   - Explicitly reported information
   - Evidence-supported inference
   - Information not established by the supplied evidence

4. Do not interpret the absence of a topic in retrieved passages
   as proof that the paper did not address that topic.

5. Do not invent datasets, results, limitations, or experiments.

6. Preserve numerical values accurately.

7. Knowledge-graph relationships are candidate relationships
   extracted automatically. Verify them against their supporting
   context before using them.

8. If a detail cannot be established from the supplied evidence,
   write:
   "Not established by supplied evidence."

9. Include paper page numbers and evidence IDs wherever possible.

10. Do not suggest new research methods or research directions
    during this extraction stage.

Return a structured analysis of this paper.
"""

        prompt_values = {
            "paper_name": paper_name,
            "paper_text": instructions,
            "paper_content": instructions,
            "text": instructions,
            "content": instructions,
            "context": instructions,
            "evidence": evidence_text,
            "retrieved_evidence": evidence_text,
            "instructions": instructions
        }

        extraction_prompt = _render_prompt(
            paper_extraction_prompt,
            prompt_values
        )

        # The instructions are appended so that they remain present
        # even if the existing template uses different placeholders.
        final_extraction_prompt = (
            extraction_prompt
            + "\n\n"
            + instructions
        )

        start_time = time.perf_counter()

        response = call_gpt(final_extraction_prompt)

        elapsed = time.perf_counter() - start_time

        parsed = _extract_llm_output(response)

        extracted_information[paper_name] = parsed["text"]

        extraction_time += elapsed
        extraction_input_tokens += _safe_int(
            parsed["input_tokens"]
        )
        extraction_output_tokens += _safe_int(
            parsed["output_tokens"]
        )

        extraction_calls += 1

        print(f"Completed extraction for {paper_name}")

    extraction_metrics = {
        "extraction_time": extraction_time,
        "extraction_input_tokens": extraction_input_tokens,
        "extraction_output_tokens": extraction_output_tokens,
        "extraction_calls": extraction_calls
    }

    return extracted_information, extraction_metrics


# ============================================================
# STEP 11: PREPARE FINAL GAP ANALYSIS INPUT
# ============================================================

def prepare_final_gap_input(
    extracted_information,
    raw_evidence_by_paper
):
    """
    Combines:
    Part A: Structured per-paper LLM analysis.
    Part B: Original retrieved source evidence.

    The final LLM receives both, so it can cross-check summaries
    against the original passages.
    """

    # --------------------------------------------------------
    # PART A: Structured paper analyses
    # --------------------------------------------------------

    summary_sections = []

    for paper_name, summary in extracted_information.items():

        summary_sections.append(
            f"""
==================================================
PAPER: {paper_name}
==================================================

{summary}
"""
        )

    summaries_text = "\n".join(summary_sections)

    # --------------------------------------------------------
    # PART B: Original source evidence
    # --------------------------------------------------------

    source_sections = []

    for paper_name, evidence in raw_evidence_by_paper.items():

        if not evidence:
            continue

        evidence_text = format_paper_evidence(
            paper_name,
            evidence
        )

        source_sections.append(
            f"""
==================================================
ORIGINAL SOURCE EVIDENCE: {paper_name}
==================================================

{evidence_text}
"""
        )

    source_evidence_text = "\n".join(source_sections)

    # --------------------------------------------------------
    # FINAL INPUT
    # --------------------------------------------------------

    combined_information = f"""
You are conducting a cross-paper research gap analysis.

You have been provided with two kinds of information.

PART A: STRUCTURED PAPER ANALYSES
These are LLM-generated summaries of individual papers.

{summaries_text}


PART B: ORIGINAL RETRIEVED SOURCE EVIDENCE
These are passages retrieved directly from the supplied papers.

{source_evidence_text}


IMPORTANT EVIDENCE RULES:

1. Original source passages in PART B take priority over
   interpretations or claims in PART A.

2. Use PART A to understand the broader structure of each paper.
   Use PART B to verify important claims.

3. Every proposed research gap must have identifiable evidence
   from the supplied papers.

4. Cite the paper name and page number when explaining evidence.
   Include evidence IDs where useful.

5. A knowledge-graph relationship is an automatically extracted
   candidate relationship, not an independently verified fact.
   Verify it using its supporting context.

6. Do not treat missing graph nodes or edges as evidence that
   a paper did not investigate a topic.

7. Do not treat a topic's absence from retrieved passages as
   proof that the paper did not address that topic.

8. Distinguish different research dimensions. For example:
   - Class imbalance
   - Interpretability
   - Feature-level explanation
   - Causal inference
   - Generalizability
   - External validation
   - Sample size
   - Computational efficiency
   - Predictive performance
   - Clinical deployment

   Evidence about one dimension must not automatically be used
   to establish a gap in another dimension.

9. Do not combine unrelated limitations into one ambitious gap.

10. Suggested research directions must directly follow from
    the evidence. Do not introduce clustering, representation
    learning, causal modelling, explainability, or other methods
    unless the supplied evidence provides a reason to do so.

11. If the evidence is insufficient to establish a meaningful
    cross-paper gap, do not force one.

Now identify only the strongest defensible research gaps.
"""


    return combined_information


# ============================================================
# STEP 12: FINAL CROSS-PAPER GAP ANALYSIS
# ============================================================

def perform_gap_analysis(combined_information):

    prompt_values = {
        "combined_information": combined_information,
        "paper_information": combined_information,
        "paper_summaries": combined_information,
        "context": combined_information,
        "text": combined_information,
        "content": combined_information,
        "evidence": combined_information
    }

    final_prompt = _render_prompt(
        gap_prompt,
        prompt_values
    )

    # Append the evidence rules and original evidence explicitly.
    final_prompt = (
        final_prompt
        + "\n\n"
        + combined_information
    )

    start_time = time.perf_counter()

    response = call_gpt(final_prompt)

    elapsed = time.perf_counter() - start_time

    parsed = _extract_llm_output(response)

    metrics = {
        "gap_analysis_time": elapsed,
        "gap_analysis_input_tokens": _safe_int(
            parsed["input_tokens"]
        ),
        "gap_analysis_output_tokens": _safe_int(
            parsed["output_tokens"]
        ),
        "gap_analysis_calls": 1
    }

    return parsed["text"], metrics


# ============================================================
# MAIN SYSTEM 3 PIPELINE
# ============================================================

def analyze_kg_rag(papers, top_k=DEFAULT_TOP_K):
    """
    Complete System 3:

    NER + Relation Extraction
             |
             v
       Knowledge Graph
             |
             |
    Original PDF Text
             |
             v
       Vector Retrieval
             |
             v
      Graph Retrieval
             |
             v
    Evidence Grouping
             |
             v
    Per-Paper Extraction
             |
             v
    Cross-Paper Gap Analysis

    Parameters
    ----------
    papers : dict
        {
            "paper1.pdf": [
                {"page": 1, "text": "..."},
                {"page": 2, "text": "..."}
            ]
        }

    top_k : int
        Number of vector chunks retrieved per query.

    Returns
    -------
    dict
        Gap analysis, extracted information, evidence,
        graph statistics, and execution metrics.
    """

    total_start = time.perf_counter()

    # --------------------------------------------------------
    # Initialize metrics
    # --------------------------------------------------------

    metrics = {
        "total_papers": len(papers),
        "total_pages": sum(
            len(pages)
            for pages in papers.values()
        ),
        "total_chunks": 0,
        "total_vector_evidence": 0,
        "total_graph_evidence": 0,
        "embedding_time": 0.0,
        "embedding_calls": 0,
        "extraction_time": 0.0,
        "extraction_calls": 0,
        "extraction_input_tokens": 0,
        "extraction_output_tokens": 0,
        "gap_analysis_time": 0.0,
        "gap_analysis_calls": 0,
        "gap_analysis_input_tokens": 0,
        "gap_analysis_output_tokens": 0
    }

    client = None
    collection_name = None

    try:

        # ----------------------------------------------------
        # STEP 1: NER + RELATION EXTRACTION
        # ----------------------------------------------------

        print("\n========== STEP 1: NER ==========")

        relations_by_paper, ner_statistics = (
            extract_paper_relations(papers)
        )

        # ----------------------------------------------------
        # STEP 2: BUILD KNOWLEDGE GRAPH
        # ----------------------------------------------------

        print("\n========== STEP 2: KNOWLEDGE GRAPH ==========")

        graph, graph_stats = create_knowledge_graph(
            relations_by_paper
        )

        # ----------------------------------------------------
        # STEP 3: CREATE VECTOR CHUNKS
        # ----------------------------------------------------

        print("\n========== STEP 3: CHUNKING ==========")

        all_chunks, all_metadata = create_chunks(papers)

        metrics["total_chunks"] = len(all_chunks)

        if not all_chunks:
            raise ValueError(
                "No text chunks were generated from the supplied papers."
            )

        # ----------------------------------------------------
        # STEP 4: GENERATE EMBEDDINGS
        # ----------------------------------------------------

        print("\n========== STEP 4: EMBEDDINGS ==========")

        (
            all_embeddings,
            embedding_time,
            embedding_calls
        ) = generate_embeddings(all_chunks)

        metrics["embedding_time"] = embedding_time
        metrics["embedding_calls"] = embedding_calls

        # ----------------------------------------------------
        # STEP 5: CREATE CHROMA COLLECTION
        # ----------------------------------------------------

        print("\n========== STEP 5: VECTOR DATABASE ==========")

        (
            client,
            collection,
            collection_name
        ) = create_vector_collection(
            all_chunks,
            all_metadata,
            all_embeddings
        )

        # ----------------------------------------------------
        # STEP 6: VECTOR RETRIEVAL
        # ----------------------------------------------------

        print("\n========== STEP 6: VECTOR RETRIEVAL ==========")

        vector_evidence = retrieve_vector_evidence(
            collection,
            top_k=top_k
        )

        metrics["total_vector_evidence"] = len(
            vector_evidence
        )

        print(
            f"Vector evidence retrieved: "
            f"{len(vector_evidence)}"
        )

        # ----------------------------------------------------
        # STEP 7: GRAPH RETRIEVAL
        # ----------------------------------------------------

        print("\n========== STEP 7: GRAPH RETRIEVAL ==========")

        graph_evidence = retrieve_kg_evidence(graph)

        metrics["total_graph_evidence"] = len(
            graph_evidence
        )

        print(
            f"Graph evidence retrieved: "
            f"{len(graph_evidence)}"
        )

        # ----------------------------------------------------
        # STEP 8: RANK AND GROUP EVIDENCE
        # ----------------------------------------------------

        print("\n========== STEP 8: EVIDENCE GROUPING ==========")

        raw_evidence_by_paper = group_evidence_by_paper(
            papers,
            vector_evidence,
            graph_evidence
        )

        # ----------------------------------------------------
        # STEP 9: PER-PAPER EXTRACTION
        # ----------------------------------------------------

        print("\n========== STEP 9: PAPER EXTRACTION ==========")

        (
            extracted_information,
            extraction_metrics
        ) = extract_information_per_paper(
            papers,
            raw_evidence_by_paper
        )

        metrics.update(extraction_metrics)

        # ----------------------------------------------------
        # STEP 10: PREPARE FINAL INPUT
        # ----------------------------------------------------

        print("\n========== STEP 10: FINAL INPUT ==========")

        combined_information = prepare_final_gap_input(
            extracted_information,
            raw_evidence_by_paper
        )

        # ----------------------------------------------------
        # STEP 11: FINAL GAP ANALYSIS
        # ----------------------------------------------------

        print("\n========== STEP 11: GAP ANALYSIS ==========")

        result, gap_metrics = perform_gap_analysis(
            combined_information
        )

        metrics.update(gap_metrics)

        # ----------------------------------------------------
        # TOTAL EXECUTION TIME
        # ----------------------------------------------------

        metrics["total_time"] = (
            time.perf_counter() - total_start
        )

        # ----------------------------------------------------
        # RETURN RESULTS
        # ----------------------------------------------------

        return {
            "result": result,

            "extracted_information": extracted_information,

            "retrieved_evidence": vector_evidence,

            "graph_evidence": graph_evidence,

            "raw_evidence_by_paper": raw_evidence_by_paper,

            "ner_statistics": ner_statistics,

            "graph_stats": graph_stats,

            "metrics": metrics
        }

    finally:

        # ----------------------------------------------------
        # CLEAN UP TEMPORARY CHROMA COLLECTION
        # ----------------------------------------------------

        if client is not None and collection_name is not None:

            try:
                client.delete_collection(
                    name=collection_name
                )

                print("\nTemporary Chroma collection deleted.")

            except Exception as e:
                print(
                    f"Chroma cleanup warning: {e}"
                )
