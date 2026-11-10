
import re
import math
import networkx as nx


# --------------------------------------------------
# 1. ENTITY TYPE NORMALIZATION
# --------------------------------------------------

TYPE_ALIASES = {
    "method": "Method",
    "methods": "Method",
    "task": "Task",
    "tasks": "Task",
    "dataset": "Dataset",
    "datasets": "Dataset",
    "material": "Dataset",
    "materials": "Dataset",
}


def normalize_type(entity_type):
    """Convert entity types to consistent labels."""

    value = str(entity_type).strip().lower()

    return TYPE_ALIASES.get(
        value,
        str(entity_type).strip().title() if value else "Unknown"
    )


# --------------------------------------------------
# 2. ENTITY ALIASES
# --------------------------------------------------
# Add aliases here when you identify equivalent terms.
# Keep this dictionary conservative to avoid false merges.

ENTITY_ALIASES = {
    "Method": {
        "rf": "random forest",
        "random forest classifier": "random forest",
        "random forest model": "random forest",

        "svm": "support vector machine",
        "support vector machines": "support vector machine",
        "support vector classifier": "support vector machine",

        "knn": "k nearest neighbors",
        "k nearest neighbor": "k nearest neighbors",
        "k nearest neighbours": "k nearest neighbors",

        "cnn": "convolutional neural network",
        "convolutional neural networks": "convolutional neural network",

        "dcnn": "deep convolutional neural network",
        "deep convolutional neural networks": "deep convolutional neural network",

        "xgboost classifier": "xgboost",
        "extreme gradient boosting": "xgboost",

        "lr": "logistic regression",
        "logistic regression classifier": "logistic regression",
    },

    "Dataset": {
        "wdbc": "wisconsin breast cancer diagnostic dataset",
        "wisconsin diagnostic breast cancer dataset":
            "wisconsin breast cancer diagnostic dataset",
        "breast cancer wisconsin diagnostic dataset":
            "wisconsin breast cancer diagnostic dataset",
        "wisconsin breast cancer dataset":
            "wisconsin breast cancer diagnostic dataset",
    }
}


# --------------------------------------------------
# 3. BASIC TEXT NORMALIZATION
# --------------------------------------------------

def clean_entity_text(text):
    """Clean entity text without changing its meaning."""

    if text is None:
        return ""

    text = str(text).strip().lower()

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove surrounding punctuation
    text = text.strip(" .,;:!?()[]{}\"'")

    return text


def normalize_entity(text, entity_type=None):
    """
    Return a canonical entity name.

    Example:
        RF -> random forest
        Random Forest Classifier -> random forest
    """

    cleaned = clean_entity_text(text)

    if not cleaned:
        return ""

    entity_type = normalize_type(entity_type or "Unknown")

    aliases = ENTITY_ALIASES.get(entity_type, {})

    return aliases.get(cleaned, cleaned)


# --------------------------------------------------
# 4. SAFE VALUE CONVERSION
# --------------------------------------------------

def safe_float(value, default=0.0):

    try:
        result = float(value)

        if not math.isfinite(result):
            return default

        return result

    except (TypeError, ValueError):
        return default


def safe_int(value, default=0):

    try:
        if value is None:
            return default

        if isinstance(value, float) and math.isnan(value):
            return default

        return int(value)

    except (TypeError, ValueError):
        return default


def safe_text(value, default=""):

    if value is None:
        return default

    try:
        if math.isnan(value):
            return default
    except (TypeError, ValueError):
        pass

    return str(value).strip()


# --------------------------------------------------
# 5. ADD OR UPDATE ENTITY NODE
# --------------------------------------------------

def add_or_update_node(
    graph,
    original_name,
    entity_type,
    paper_name,
    page
):
    """
    Add an entity node or update an existing canonical node.

    Returns the canonical node ID.
    """

    entity_type = normalize_type(entity_type)

    canonical_name = normalize_entity(
        original_name,
        entity_type
    )

    if not canonical_name:
        return None

    # Include entity type to avoid accidental collisions.
    node_id = f"{entity_type.lower()}::{canonical_name}"

    if not graph.has_node(node_id):

        graph.add_node(
            node_id,
            label=canonical_name,
            canonical_name=canonical_name,
            type=entity_type,
            aliases=set(),
            papers=set(),
            mentions=[]
        )

    node_data = graph.nodes[node_id]

    # Preserve original wording as an alias.
    node_data["aliases"].add(
        safe_text(original_name)
    )

    # Track all papers containing this entity.
    node_data["papers"].add(paper_name)

    # Preserve where the entity was mentioned.
    mention = {
        "paper": paper_name,
        "page": safe_int(page),
        "original_name": safe_text(original_name)
    }

    if mention not in node_data["mentions"]:
        node_data["mentions"].append(mention)

    return node_id


# --------------------------------------------------
# 6. BUILD KNOWLEDGE GRAPH
# --------------------------------------------------

def build_knowledge_graph(
    relations_by_paper,
    confidence_threshold=0.60
):
    """
    Build a cross-paper directed knowledge graph.

    Input:
        {
            "paper1.pdf": relation_dataframe,
            "paper2.pdf": relation_dataframe
        }

    Output:
        NetworkX MultiDiGraph

    Features:
        - Entity normalization
        - Cross-paper entity linking
        - Multiple relationship preservation
        - Paper/page provenance
        - Confidence filtering
    """

    graph = nx.MultiDiGraph()

    graph.graph["description"] = (
        "Cross-paper research knowledge graph"
    )

    graph.graph["confidence_threshold"] = confidence_threshold

    for paper_name, relations_df in relations_by_paper.items():

        if relations_df is None or relations_df.empty:
            continue

        for _, row in relations_df.iterrows():

            confidence = safe_float(
                row.get("confidence", 0)
            )

            if confidence < confidence_threshold:
                continue

            subject = safe_text(row.get("subject", ""))
            obj = safe_text(row.get("object", ""))

            subject_type = normalize_type(
                row.get("subject_type", "Unknown")
            )

            object_type = normalize_type(
                row.get("object_type", "Unknown")
            )

            relation = safe_text(
                row.get("relation", "RELATED_TO"),
                "RELATED_TO"
            )

            if not subject or not obj:
                continue

            # Create canonical node IDs.
            subject_id = add_or_update_node(
                graph,
                subject,
                subject_type,
                paper_name,
                row.get("subject_page", 0)
            )

            object_id = add_or_update_node(
                graph,
                obj,
                object_type,
                paper_name,
                row.get("object_page", 0)
            )

            if not subject_id or not object_id:
                continue

            # Ignore self-relations.
            if subject_id == object_id:
                continue

            subject_page = safe_int(
                row.get("subject_page", 0)
            )

            object_page = safe_int(
                row.get("object_page", 0)
            )

            context = safe_text(
                row.get("context", "")
            )

            # Keep every paper's evidence as a separate edge.
            graph.add_edge(
                subject_id,
                object_id,
                relation=relation,
                confidence=confidence,
                paper=paper_name,
                subject_page=subject_page,
                object_page=object_page,
                context=context,
                subject_original=subject,
                object_original=obj
            )

    # Store useful graph-level statistics.
    graph.graph["total_nodes"] = graph.number_of_nodes()
    graph.graph["total_edges"] = graph.number_of_edges()

    graph.graph["papers"] = sorted({
        paper
        for _, data in graph.nodes(data=True)
        for paper in data.get("papers", set())
    })

    return graph


# --------------------------------------------------
# 7. CROSS-PAPER PATTERN ANALYSIS
# --------------------------------------------------

def get_cross_paper_patterns(graph, min_papers=2):
    """
    Find entities shared by multiple papers.

    Useful for identifying repeated methods, datasets
    and tasks across the reviewed literature.
    """

    patterns = []

    for node, data in graph.nodes(data=True):

        papers = sorted(data.get("papers", set()))

        if len(papers) < min_papers:
            continue

        patterns.append({
            "entity": data.get("label", node),
            "entity_type": data.get("type", "Unknown"),
            "paper_count": len(papers),
            "papers": papers,
            "aliases": sorted(data.get("aliases", set()))
        })

    patterns.sort(
        key=lambda item: item["paper_count"],
        reverse=True
    )

    return patterns


# --------------------------------------------------
# 8. GRAPH SUMMARY
# --------------------------------------------------

def get_graph_summary(graph):

    return {
        "total_nodes": graph.number_of_nodes(),
        "total_relationships": graph.number_of_edges(),
        "total_papers": len(graph.graph.get("papers", [])),
        "entity_types": {
            entity_type: sum(
                1
                for _, data in graph.nodes(data=True)
                if data.get("type") == entity_type
            )
            for entity_type in {
                data.get("type", "Unknown")
                for _, data in graph.nodes(data=True)
            }
        },
        "cross_paper_entities": len(
            get_cross_paper_patterns(graph)
        )
    }
