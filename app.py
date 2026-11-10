
import os
import time

import streamlit as st
from dotenv import load_dotenv


# Load environment variables before importing pipelines
BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

load_dotenv(
    os.path.join(BASE_DIR, ".env"),
    override=True
)


if not os.getenv("OPENAI_API_KEY"):
    st.error(
        "OPENAI_API_KEY is missing. "
        "Please check your .env file."
    )
    st.stop()


from pdf_extractor import extract_uploaded_papers
from llm_only import analyze_llm_only
from rag_pipeline import analyze_rag


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Research Gap Finder",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Research Gap Finder")

st.write(
    "Identify research gaps across multiple academic "
    "papers using LLM-only, Vector RAG, and "
    "Knowledge Graph + RAG approaches."
)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "papers" not in st.session_state:
    st.session_state.papers = None

if "results" not in st.session_state:
    st.session_state.results = {}


# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------

uploaded_files = st.file_uploader(
    "Upload Research Papers",
    type=["pdf"],
    accept_multiple_files=True
)


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

st.sidebar.header("Analysis Settings")

top_k = st.sidebar.slider(
    "Vector passages per query",
    min_value=1,
    max_value=10,
    value=4
)

selected_systems = st.multiselect(
    "Select systems to compare",
    options=[
        "LLM-only",
        "Vector RAG + LLM",
        "Knowledge Graph + RAG + LLM"
    ],
    default=[
        "LLM-only",
        "Vector RAG + LLM",
        "Knowledge Graph + RAG + LLM"
    ]
)


# --------------------------------------------------
# RESULT NORMALIZATION
# --------------------------------------------------

def normalize_output(output):

    """
    Handles dictionary, tuple, and string outputs.
    """

    if isinstance(output, dict):

        result = (
            output.get("result")
            or output.get("text")
            or ""
        )

        metrics = output.get(
            "metrics", {}
        )

        evidence = output.get(
            "retrieved_evidence",
            []
        )

        graph_evidence = output.get(
            "graph_evidence",
            []
        )

        graph_stats = output.get(
            "graph_stats",
            {}
        )

        return (
            result,
            metrics,
            evidence,
            graph_evidence,
            graph_stats
        )

    if isinstance(output, tuple):

        result = (
            output[0]
            if len(output) > 0
            else ""
        )

        evidence = (
            output[1]
            if len(output) > 1
            else []
        )

        return (
            str(result),
            {},
            evidence if isinstance(evidence, list) else [],
            [],
            {}
        )

    return (
        str(output),
        {},
        [],
        [],
        {}
    )


# --------------------------------------------------
# METRICS DISPLAY
# --------------------------------------------------

def display_metrics(metrics, elapsed):

    st.subheader("Performance Metrics")

    input_tokens = metrics.get(
        "input_tokens", 0
    )

    output_tokens = metrics.get(
        "output_tokens", 0
    )

    embedding_tokens = metrics.get(
        "embedding_tokens", 0
    )

    throughput = (
        output_tokens / elapsed
        if elapsed > 0
        else 0
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Latency",
        f"{elapsed:.2f} sec"
    )

    c2.metric(
        "Input Tokens",
        f"{input_tokens:,}"
    )

    c3.metric(
        "Output Tokens",
        f"{output_tokens:,}"
    )

    c4.metric(
        "Throughput",
        f"{throughput:.2f} tokens/sec"
    )

    with st.expander("Additional Metrics"):

        st.write(
            "Embedding Tokens:",
            embedding_tokens
        )

        for key, value in metrics.items():

            if key not in {
                "input_tokens",
                "output_tokens",
                "embedding_tokens"
            }:

                st.write(
                    f"{key}: {value}"
                )


# --------------------------------------------------
# ANALYSIS BUTTON
# --------------------------------------------------

if st.button(
    "Identify Research Gaps",
    type="primary",
    use_container_width=True
):

    if not uploaded_files:

        st.warning(
            "Please upload at least one PDF."
        )

    elif not selected_systems:

        st.warning(
            "Please select at least one system."
        )

    else:

        st.session_state.results = {}

        try:

            # --------------------------------------
            # COMMON PDF EXTRACTION
            # --------------------------------------

            with st.spinner(
                "Extracting research papers..."
            ):

                papers = extract_uploaded_papers(
                    uploaded_files
                )

            papers = {
                name: pages
                for name, pages in papers.items()
                if pages
            }

            if not papers:

                st.error(
                    "No readable text was found."
                )

                st.stop()

            st.session_state.papers = papers

            st.success(
                f"Extracted {len(papers)} research papers."
            )

            with st.expander(
                "View extracted paper information"
            ):

                for paper_name, pages in papers.items():

                    st.write(
                        f"**{paper_name}**"
                    )

                    st.write(
                        f"Extracted pages: {len(pages)}"
                    )

            # --------------------------------------
            # SYSTEM 1: LLM-ONLY
            # --------------------------------------

            if "LLM-only" in selected_systems:

                st.header("System 1: LLM-only")

                start = time.perf_counter()

                with st.spinner(
                    "Running LLM-only analysis..."
                ):

                    output = analyze_llm_only(
                        papers
                    )

                elapsed = (
                    time.perf_counter() - start
                )

                (
                    result,
                    metrics,
                    evidence,
                    graph_evidence,
                    graph_stats
                ) = normalize_output(output)

                st.session_state.results[
                    "LLM-only"
                ] = {
                    "result": result,
                    "metrics": metrics,
                    "elapsed": elapsed
                }

                st.markdown(result)

                display_metrics(
                    metrics,
                    elapsed
                )

                st.download_button(
                    "Download LLM-only Report",
                    data=result,
                    file_name="llm_only_report.md",
                    mime="text/markdown"
                )

            # --------------------------------------
            # SYSTEM 2: VECTOR RAG
            # --------------------------------------

            if "Vector RAG + LLM" in selected_systems:

                st.divider()

                st.header(
                    "System 2: Vector RAG + LLM"
                )

                start = time.perf_counter()

                with st.spinner(
                    "Running Vector RAG analysis..."
                ):

                    output = analyze_rag(
                        papers,
                        top_k=top_k
                    )

                elapsed = (
                    time.perf_counter() - start
                )

                (
                    result,
                    metrics,
                    evidence,
                    graph_evidence,
                    graph_stats
                ) = normalize_output(output)

                st.session_state.results[
                    "Vector RAG + LLM"
                ] = {
                    "result": result,
                    "metrics": metrics,
                    "elapsed": elapsed
                }

                st.markdown(result)

                display_metrics(
                    metrics,
                    elapsed
                )

                st.download_button(
                    "Download Vector RAG Report",
                    data=result,
                    file_name="vector_rag_report.md",
                    mime="text/markdown"
                )

                with st.expander(
                    "View Retrieved Vector Evidence"
                ):

                    for i, item in enumerate(evidence):

                        st.markdown(
                            f"### Evidence {i + 1}"
                        )

                        st.write(
                            "Paper:", item.get("paper")
                        )

                        st.write(
                            "Page:", item.get("page")
                        )

                        st.write(
                            item.get("text", "")
                        )

                        st.divider()

            # --------------------------------------
            # SYSTEM 3: KNOWLEDGE GRAPH + RAG
            # --------------------------------------

            if (
                "Knowledge Graph + RAG + LLM"
                in selected_systems
            ):

                st.divider()

                st.header(
                    "System 3: Knowledge Graph + RAG + LLM"
                )

                with st.spinner(
                    "Running NER, relation extraction, "
                    "graph construction and retrieval..."
                ):

                    # Import only when System 3 is selected.
                    # This avoids loading the NER models for
                    # users who only run Systems 1 and 2.
                    from kg_rag_pipeline import (
                        analyze_kg_rag
                    )

                    start = time.perf_counter()

                    output = analyze_kg_rag(
                        papers,
                        top_k=top_k
                    )

                    elapsed = (
                        time.perf_counter() - start
                    )

                (
                    result,
                    metrics,
                    evidence,
                    graph_evidence,
                    graph_stats
                ) = normalize_output(output)

                st.session_state.results[
                    "Knowledge Graph + RAG + LLM"
                ] = {
                    "result": result,
                    "metrics": metrics,
                    "elapsed": elapsed
                }

                st.markdown(result)

                display_metrics(
                    metrics,
                    elapsed
                )

                # Graph statistics
                st.subheader(
                    "Knowledge Graph Statistics"
                )

                if graph_stats:

                    c1, c2 = st.columns(2)

                    c1.metric(
                        "Graph Nodes",
                        graph_stats.get("nodes", 0)
                    )

                    c2.metric(
                        "Graph Edges",
                        graph_stats.get("edges", 0)
                    )

                # Graph evidence
                with st.expander(
                    "View Knowledge Graph Evidence"
                ):

                    for i, item in enumerate(
                        graph_evidence
                    ):

                        st.markdown(
                            f"### Graph Evidence {i + 1}"
                        )

                        st.write(
                            "Paper:",
                            item.get("paper")
                        )

                        st.write(
                            "Page:",
                            item.get("page")
                        )

                        st.write(
                            "Relationship:",
                            f"{item.get('subject')} "
                            f"--{item.get('relation')}--> "
                            f"{item.get('object')}"
                        )

                        st.write(
                            "Confidence:",
                            item.get("confidence")
                        )

                        st.write(
                            item.get("text")
                        )

                        st.divider()

                st.download_button(
                    "Download KG-RAG Report",
                    data=result,
                    file_name="kg_rag_report.md",
                    mime="text/markdown"
                )

            # --------------------------------------
            # COMPARISON
            # --------------------------------------

            if len(st.session_state.results) > 1:

                st.divider()

                st.header(
                    "Comparison of Approaches"
                )

                comparison = []

                for system, data in (
                    st.session_state.results.items()
                ):

                    metrics = data["metrics"]

                    comparison.append({

                        "System": system,

                        "Latency (sec)": round(
                            data["elapsed"], 2
                        ),

                        "Input Tokens": metrics.get(
                            "input_tokens", 0
                        ),

                        "Output Tokens": metrics.get(
                            "output_tokens", 0
                        ),

                        "Embedding Tokens": metrics.get(
                            "embedding_tokens", 0
                        )
                    })

                st.dataframe(
                    comparison,
                    use_container_width=True,
                    hide_index=True
                )

        except Exception as e:

            st.error(
                f"An error occurred: {str(e)}"
            )

            st.exception(e)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.divider()

    st.subheader("About")

    st.write(
        """
        This application compares three research
        gap identification approaches.

        1. LLM-only
        2. Vector RAG + LLM
        3. Knowledge Graph + RAG + LLM
        """
    )
