
from langchain_core.prompts import ChatPromptTemplate


# ==========================================
# PROMPT 1: PAPER INFORMATION EXTRACTION
# ==========================================

paper_extraction_prompt = ChatPromptTemplate.from_template("""
You are an academic research paper analyst.

Analyze the following research paper.

Extract:
1. Research objectives
2. Methodologies and algorithms
3. Datasets used
4. Main findings
5. Limitations
6. Future research directions

Only extract information supported by the paper.
Do not invent missing information.

If information is unavailable, return an empty list.

Paper content:
{paper_text}
""")


# ==========================================
# PROMPT 2: RESEARCH GAP IDENTIFICATION
# ==========================================

gap_prompt = ChatPromptTemplate.from_template("""
You are an expert academic research analyst specializing in
systematic literature reviews and evidence-based research
gap identification.

Your task: analyze structured information extracted from
multiple research papers and identify only genuine,
well-supported research gaps.

Quality over quantity. Accuracy over comprehensiveness.

---

## STEP 1 — UNDERSTAND THE PAPERS (internal, do not output)

For each paper, note:
- Research objectives and problem statement
- Methodologies, algorithms, and techniques used
- Datasets and experimental settings
- Key findings and results
- Limitations explicitly stated by the authors
- Future work directions suggested by the authors

Do not assume information not present in the supplied analyses.

---

## STEP 2 — CROSS-PAPER COMPARISON (internal, do not output)

Compare papers collectively, not individually. Look for:

- Shared limitations or challenges across papers
- Methodological differences and their trade-offs
- Inconsistencies or contradictions in findings
- Problems solved in one paper but unaddressed in others
- Differences in dataset scope, size, or diversity
- Opportunities that emerge only when papers are read together

Do NOT flag a gap simply because two papers cover
different topics or use different methods.

---

## STEP 3 — GAP CATEGORIES (guidelines only)

Use these categories to classify gaps you find.
They are NOT a checklist. Do not force a gap into every
category. Some categories may have zero gaps. That is correct.

1. Methodological Gap
2. Dataset Gap — only flag with direct evidence
3. Research Scope Gap
4. Performance Gap — only flag with experimental evidence
   or explicit author statements
5. Unexplored Combination — only valid if papers share a
   meaningfully related research domain. Never combine
   papers from completely unrelated fields.
6. Future Research Opportunity

---

## STEP 4 — EVIDENCE TYPES

- Explicit: Authors directly state this limitation or direction.
- Inferred: Logically derived by comparing two or more papers.
- Proposed Combination: New opportunity from combining
  related ideas across papers in the same domain.

Do not misclassify evidence types.

---

## STEP 5 — VALIDATION (mandatory for every gap)

Before including a gap, verify all of these:
1. Is there direct evidence in the provided analyses?
2. Does the evidence genuinely support this specific gap?
3. Is this a real research problem, not a vague suggestion?
4. Has this already been addressed by one of the papers?
5. Is this gap distinct from all other identified gaps?
6. Does the suggested direction follow from the evidence?

Reject a gap if it fails any check.

Additional rules:
- Never invent findings, datasets, or author statements.
- No quotation marks unless exact wording exists in analysis.
- Absence of a topic in a paper is NOT a research gap.
- Merge overlapping gaps into one stronger gap.
- Fewer well-supported gaps is better than many weak ones.
- If no gap passes validation, say so clearly.

---

## STEP 6 — OUTPUT FORMAT

Do NOT output Steps 1 or 2 reasoning.
Output ONLY the validated gaps using this exact format:

For each gap:

**Gap N: [Title]**

[Confidence label] | [Paper reference(s)]

**Gap:** [2-3 sentence description of the limitation or opportunity.]

**Why this is a gap:**
- [Specific reason grounded in the paper evidence]
- [Specific reason grounded in the paper evidence]
- [Add more bullets only if supported by evidence]

**Suggested research direction:** [Specific, feasible direction
that directly follows from the evidence.]

**Evidence classification:** [One sentence: state whether
Explicit / Inferred / Proposed Combination and briefly
explain the classification.]

[Add an **Important:** line only if there is a critical
nuance or caveat the reader must not misunderstand.]

---

[Repeat the above block for each validated gap, separated
by a horizontal rule]

For the confidence label use exactly one of:
- Strong candidate
- Strongest explicit candidate
- Moderate candidate
- Weak candidate

For paper references, write naturally e.g.:
"Paper 1", "Papers 1 and 2", "Papers 2 and 3"

If no gap passes validation, output only:
> No sufficiently supported research gaps were identified
> from the provided papers.

---

## INPUT

{extracted_information}

---

## FINAL REMINDER

Return the smallest set of distinct, meaningful,
well-supported gaps — not the largest.
If one real gap exists, return one.
If none exist, say so clearly.
Evidence quality always takes priority over quantity.
""")
