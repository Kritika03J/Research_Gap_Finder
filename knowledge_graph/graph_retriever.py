
import re


# --------------------------------------------------
# 1. STOPWORDS
# --------------------------------------------------

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were",
    "what", "which", "how", "why", "and", "or",
    "of", "in", "on", "to", "for", "from", "with",
    "do", "does", "did", "used", "use", "using",
    "paper", "research", "study", "studies",
    "identify", "find", "explain", "describe",
    "show", "discuss", "analyze", "analysis",
    "gap", "gaps"
}


# --------------------------------------------------
# 2. QUERY TOKENIZATION
# --------------------------------------------------

def query_tokens(text):

    words = re.findall(
        r"[a-zA-Z0-9]+",
        str(text).lower()
    )

    return {
        word
        for word in words
        if word not in STOPWORDS and len(word) > 1
    }


# --------------------------------------------------
# 3. NODE MATCHING
# --------------------------------------------------

def score_node_match(query, q_tokens, node_data):
    """
    Score a graph node against the user's query.

    Considers:
        - Exact phrase matching
        - Canonical entity name
        - Entity aliases
        - Token overlap
        - Query coverage
    """

    label = str(
        node_data.get("label", "")
    ).lower()

    aliases = node_data.get("aliases", set())

    candidate_names = {label}

    candidate_names.update(
        str(alias).lower()
        for alias in aliases
    )

    best_score = 0
    best_overlap = 0

    for name in candidate_names:

        name_tokens = query_tokens(name)

        if not name_tokens:
            continue

        overlap = len(
            q_tokens.intersection(name_tokens)
        )

        coverage = overlap / len(q_tokens)

        # Strong bonus for an exact phrase match.
        phrase_bonus = 0

        if name and name in query.lower():
            phrase_bonus = 5

        # Partial token matching.
        token_score = (
            2 * overlap
            + 3 * coverage
            + phrase_bonus
        )

        if token_score > best_score:

            best_score = token_score
            best_overlap = overlap

    return best_score, best_overlap


# --------------------------------------------------
# 4. CREATE EVIDENCE
# --------------------------------------------------

def make_evidence(
    subject,
    obj,
    edge_data,
    matched_node=None,
    retrieval_score=0
):

    relation = edge_data.get(
        "relation", "RELATED_TO"
    )

    paper = edge_data.get(
        "paper", "Unknown paper"
    )

    subject_page = edge_data.get(
        "subject_page", 0
    )

    object_page = edge_data.get(
        "object_page", 0
    )

    pages = [
        page
        for page in [subject_page, object_page]
        if page and page > 0
    ]

    page = min(pages) if pages else 0

    context = edge_data.get(
        "context", ""
    )

    confidence = float(
        edge_data.get("confidence", 0)
    )

    text = (
        f"{subject} --{relation}--> {obj}. "
        f"Context: {context}"
    )

    return {
        "paper": paper,
        "page": page,
        "subject": subject,
        "relation": relation,
        "object": obj,
        "confidence": confidence,
        "context": context,
        "text": text,
        "matched_node": matched_node,
        "retrieval_score": retrieval_score
    }


# --------------------------------------------------
# 5. EDGE RELEVANCE
# --------------------------------------------------

def score_edge_relevance(
    q_tokens,
    edge_data,
    subject,
    obj,
    seed_score
):
    """
    Rank an edge using its connected entities,
    relation and supporting context.
    """

    relation = str(
        edge_data.get("relation", "")
    )

    context = str(
        edge_data.get("context", "")
    )

    edge_text = (
        subject + " " +
        obj + " " +
        relation + " " +
        context
    )

    edge_tokens = query_tokens(edge_text)

    overlap = len(
        q_tokens.intersection(edge_tokens)
    )

    confidence = float(
        edge_data.get("confidence", 0)
    )

    return (
        seed_score
        + (2 * overlap)
        + confidence
    )


# --------------------------------------------------
# 6. RETRIEVE GRAPH EVIDENCE
# --------------------------------------------------

def retrieve_graph_evidence(
    graph,
    query,
    max_results=20,
    max_seed_nodes=10,
    min_node_score=0
):
    """
    Retrieve relevant graph relationships.

    Steps:
        1. Match query against canonical node labels
           and aliases.
        2. Rank matching nodes.
        3. Retrieve incoming and outgoing relationships.
        4. Rank evidence using query relevance and confidence.
        5. Remove duplicate evidence.

    Returns:
        List of dictionaries containing:
        paper, page, subject, relation, object,
        confidence, context, text, retrieval_score.
    """

    q_tokens = query_tokens(query)

    if not q_tokens:
        return []

    ranked_nodes = []

    # ----------------------------------------------
    # Match query against all graph nodes.
    # ----------------------------------------------

    for node, data in graph.nodes(data=True):

        score, overlap = score_node_match(
            query,
            q_tokens,
            data
        )

        if overlap > 0 and score >= min_node_score:

            ranked_nodes.append({
                "node": node,
                "score": score,
                "overlap": overlap
            })

    # Higher score first.
    ranked_nodes.sort(
        key=lambda item: (
            item["score"],
            item["overlap"]
        ),
        reverse=True
    )

    seed_nodes = ranked_nodes[:max_seed_nodes]

    evidence = []
    seen = set()

    # ----------------------------------------------
    # Retrieve connected relationships.
    # ----------------------------------------------

    for seed in seed_nodes:

        node = seed["node"]
        seed_score = seed["score"]

        subject_label = graph.nodes[node].get(
            "label", node
        )

        # Outgoing relationships.
        for _, target, edge_key, edge_data in graph.out_edges(
            node,
            keys=True,
            data=True
        ):

            obj_label = graph.nodes[target].get(
                "label", target
            )

            retrieval_score = score_edge_relevance(
                q_tokens,
                edge_data,
                subject_label,
                obj_label,
                seed_score
            )

            item = make_evidence(
                subject_label,
                obj_label,
                edge_data,
                matched_node=node,
                retrieval_score=retrieval_score
            )

            # Include provenance in the deduplication key.
            key = (
                item["paper"],
                item["page"],
                item["subject"],
                item["relation"],
                item["object"],
                item["context"]
            )

            if key not in seen:

                seen.add(key)

                evidence.append(item)

        # Incoming relationships.
        for source, _, edge_key, edge_data in graph.in_edges(
            node,
            keys=True,
            data=True
        ):

            source_label = graph.nodes[source].get(
                "label", source
            )

            retrieval_score = score_edge_relevance(
                q_tokens,
                edge_data,
                source_label,
                subject_label,
                seed_score
            )

            item = make_evidence(
                source_label,
                subject_label,
                edge_data,
                matched_node=node,
                retrieval_score=retrieval_score
            )

            key = (
                item["paper"],
                item["page"],
                item["subject"],
                item["relation"],
                item["object"],
                item["context"]
            )

            if key not in seen:

                seen.add(key)

                evidence.append(item)

    # ----------------------------------------------
    # Final evidence ranking.
    # ----------------------------------------------

    evidence.sort(
        key=lambda item: (
            item["retrieval_score"],
            item["confidence"]
        ),
        reverse=True
    )

    return evidence[:max_results]


# --------------------------------------------------
# 7. GET GRAPH NEIGHBOURS
# --------------------------------------------------

def get_node_neighbours(graph, node):

    if node not in graph:
        return []

    neighbours = set()

    neighbours.update(
        graph.successors(node)
    )

    neighbours.update(
        graph.predecessors(node)
    )

    return [
        {
            "node": neighbour,
            "label": graph.nodes[neighbour].get(
                "label", neighbour
            ),
            "type": graph.nodes[neighbour].get(
                "type", "Unknown"
            )
        }
        for neighbour in neighbours
    ]
