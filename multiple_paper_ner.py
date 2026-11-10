
"""
multiple_paper_ner.py

NER + Relation Extraction module for Research Gap Finder.

Input:
    pages = [
        {"page": 1, "text": "Page one text..."},
        {"page": 2, "text": "Page two text..."}
    ]

Main function:
    process_paper_pages(pages)

Returns:
    ner_df, relation_df, stats

The module uses the same trained NER and relation extraction
models as the original Colab notebook.
"""

import json
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import pysbd
import torch
import torch.nn as nn

from tqdm.auto import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoModel,
    AutoConfig
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

NER_MODEL_DIR = BASE_DIR / "ner_model"
RE_MODEL_DIR = BASE_DIR / "re_model"

NER_MODEL_URL = (
    "https://drive.google.com/drive/folders/"
    "1XSV5Qm_3VqooZJeis_4mwGCgdW_8RCg2?usp=sharing"
)

RE_MODEL_URL = (
    "https://drive.google.com/drive/folders/"
    "14l8qy2rScwfq-Z4tnRgHf3JE5cZ5vSfg?usp=sharing"
)

MAX_LENGTH = 512
NER_OVERLAP_WORDS = 50

MAX_SENTENCE_GAP = 1
MIN_REL_CONFIDENCE = 0.50

RE_BATCH_SIZE = 32

# Initialize device BEFORE loading either model.
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"NER/RE device: {device}")


# ============================================================
# 2. ENTITY AND RELATION CONFIGURATION
# ============================================================

ENTITY_TYPES = [
    "Method",
    "Task",
    "Dataset"
]

ENTITY_TYPE_MAPPING = {
    "Method": "Method",
    "Task": "Task",
    "Material": "Dataset"
}

SUBJ_OPEN = {
    entity_type: f"[SUBJ-{entity_type.upper()}]"
    for entity_type in ENTITY_TYPES
}

OBJ_OPEN = {
    entity_type: f"[OBJ-{entity_type.upper()}]"
    for entity_type in ENTITY_TYPES
}

SUBJ_CLOSE = "[/SUBJ]"
OBJ_CLOSE = "[/OBJ]"

MARKERS = (
    list(SUBJ_OPEN.values())
    + list(OBJ_OPEN.values())
    + [SUBJ_CLOSE, OBJ_CLOSE]
)


# ============================================================
# 3. GLOBAL MODEL VARIABLES
# ============================================================

ner_tokenizer = None
ner_model = None
ner_id2label = None

re_tokenizer = None
re_model = None
REL_LABELS = None

segmenter = pysbd.Segmenter(
    language="en",
    clean=False
)

WORD_PATTERN = re.compile(
    r"\w+(?:[-’']\w+)*|[^\w\s]",
    re.UNICODE
)

_models_loaded = False


# ============================================================
# 4. DOWNLOAD MODEL FOLDERS IF REQUIRED
# ============================================================

def download_models_if_missing():

    """
    Download the original model folders if they are absent.

    Expected folders:
        ner_model/
        re_model/
    """

    if NER_MODEL_DIR.exists() and RE_MODEL_DIR.exists():
        return

    try:
        import gdown

    except ImportError as exc:
        raise ImportError(
            "gdown is required to download the NER and "
            "relation extraction models. Install it using: "
            "pip install gdown"
        ) from exc

    if not NER_MODEL_DIR.exists():

        print("Downloading NER model...")

        NER_MODEL_DIR.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        gdown.download_folder(
            NER_MODEL_URL,
            output=str(NER_MODEL_DIR),
            quiet=False
        )

    if not RE_MODEL_DIR.exists():

        print("Downloading relation extraction model...")

        RE_MODEL_DIR.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        gdown.download_folder(
            RE_MODEL_URL,
            output=str(RE_MODEL_DIR),
            quiet=False
        )

    if not NER_MODEL_DIR.exists():
        raise FileNotFoundError(
            f"NER model folder not found: {NER_MODEL_DIR}"
        )

    if not RE_MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Relation model folder not found: {RE_MODEL_DIR}"
        )


# ============================================================
# 5. RELATION EXTRACTION MODEL ARCHITECTURE
# ============================================================

_ids = {}


def register_markers(tokenizer):

    """
    Get token IDs for the special subject/object markers.

    The saved relation tokenizer is expected to contain
    these markers already.
    """

    marker_ids = {
        marker: tokenizer.convert_tokens_to_ids(marker)
        for marker in MARKERS
    }

    unknown_id = tokenizer.unk_token_id

    missing = [
        marker
        for marker, token_id in marker_ids.items()
        if token_id is None or token_id == unknown_id
    ]

    if missing:

        raise ValueError(
            "The relation tokenizer is missing trained "
            f"special markers: {missing}. Check the original "
            "re_model folder; do not add new tokens to a "
            "trained model without updating its embeddings."
        )

    _ids["subj_open"] = {
        marker_ids[marker]
        for marker in SUBJ_OPEN.values()
    }

    _ids["obj_open"] = {
        marker_ids[marker]
        for marker in OBJ_OPEN.values()
    }

    _ids["all"] = set(marker_ids.values())


def mark_words(tokens, subj, obj):

    """
    Insert subject and object markers into the token sequence.

    subj and obj:
        (start_index, end_index, entity_type)
    """

    out = []

    for i, word in enumerate(tokens):

        if i == subj[0]:
            out.append(SUBJ_OPEN[subj[2]])

        if i == obj[0]:
            out.append(OBJ_OPEN[obj[2]])

        out.append(word)

        if i == subj[1]:
            out.append(SUBJ_CLOSE)

        if i == obj[1]:
            out.append(OBJ_CLOSE)

    return " ".join(out)


def encode_marked(text, tokenizer, max_len):

    """
    Tokenize marked text and preserve both entity markers
    when the sequence needs truncation.
    """

    ids = tokenizer(
        text,
        add_special_tokens=False
    )["input_ids"]

    budget = max_len - 2

    if len(ids) > budget:

        positions = [
            i
            for i, token_id in enumerate(ids)
            if token_id in _ids["all"]
        ]

        if not positions:
            return None

        lo = min(positions)
        hi = max(positions)

        if hi - lo + 1 > budget:
            return None

        extra = budget - (hi - lo + 1)

        start = max(
            0,
            lo - extra // 2
        )

        end = start + budget

        if end > len(ids):

            end = len(ids)
            start = end - budget

        ids = ids[start:end]

    subj_positions = [
        i for i, token_id in enumerate(ids)
        if token_id in _ids["subj_open"]
    ]

    obj_positions = [
        i for i, token_id in enumerate(ids)
        if token_id in _ids["obj_open"]
    ]

    if (
        len(subj_positions) != 1
        or len(obj_positions) != 1
    ):
        return None

    if (
        tokenizer.cls_token_id is None
        or tokenizer.sep_token_id is None
    ):
        raise ValueError(
            "Relation tokenizer must define CLS and SEP tokens."
        )

    ids = (
        [tokenizer.cls_token_id]
        + ids
        + [tokenizer.sep_token_id]
    )

    return {
        "input_ids": ids,
        "subj_pos": subj_positions[0] + 1,
        "obj_pos": obj_positions[0] + 1
    }


class REModel(nn.Module):

    """
    Original relation classification architecture.

    Uses:
        CLS representation
        Subject marker representation
        Object marker representation
    """

    def __init__(
        self,
        encoder,
        num_labels,
        dropout=0.1
    ):

        super().__init__()

        self.encoder = encoder

        hidden_size = encoder.config.hidden_size

        self.head = nn.Sequential(

            nn.Dropout(dropout),

            nn.Linear(
                3 * hidden_size,
                hidden_size
            ),

            nn.GELU(),

            nn.Dropout(dropout),

            nn.Linear(
                hidden_size,
                num_labels
            )
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        subj_pos,
        obj_pos
    ):

        hidden_states = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state

        batch_indices = torch.arange(
            hidden_states.size(0),
            device=hidden_states.device
        )

        features = torch.cat(
            [
                hidden_states[:, 0],
                hidden_states[batch_indices, subj_pos],
                hidden_states[batch_indices, obj_pos]
            ],
            dim=-1
        )

        return self.head(features)


# ============================================================
# 6. LOAD BOTH MODELS ONCE
# ============================================================

def load_models():

    """
    Load NER and relation extraction models once.

    Subsequent calls reuse the already-loaded models.
    """

    global ner_tokenizer
    global ner_model
    global ner_id2label

    global re_tokenizer
    global re_model
    global REL_LABELS

    global _models_loaded

    if _models_loaded:
        return

    download_models_if_missing()

    # ------------------------------
    # Load NER model
    # ------------------------------

    print("Loading NER tokenizer...")

    ner_tokenizer = AutoTokenizer.from_pretrained(
        str(NER_MODEL_DIR),
        use_fast=True
    )

    if not ner_tokenizer.is_fast:

        raise ValueError(
            "NER tokenizer must be a fast tokenizer "
            "because the pipeline uses word_ids()."
        )

    print("Loading NER model...")

    ner_model = AutoModelForTokenClassification.from_pretrained(
        str(NER_MODEL_DIR)
    )

    ner_model.to(device)
    ner_model.eval()

    ner_id2label = {
        int(key): value
        for key, value in ner_model.config.id2label.items()
    }

    print(
        "NER model loaded.",
        ner_id2label
    )

    # ------------------------------
    # Load relation extraction config
    # ------------------------------

    config_path = RE_MODEL_DIR / "re_config.json"

    if not config_path.exists():

        raise FileNotFoundError(
            f"Relation model configuration not found: "
            f"{config_path}"
        )

    with open(
        config_path,
        "r",
        encoding="utf-8"
    ) as f:

        re_config = json.load(f)

    # ------------------------------
    # Load relation tokenizer
    # ------------------------------

    print("Loading relation tokenizer...")

    re_tokenizer = AutoTokenizer.from_pretrained(
        str(RE_MODEL_DIR)
    )

    register_markers(re_tokenizer)

    if re_tokenizer.pad_token_id is None:

        raise ValueError(
            "Relation tokenizer has no pad_token_id. "
            "Check the saved tokenizer used to train the "
            "relation extraction model."
        )

    # ------------------------------
    # Load relation encoder
    # ------------------------------

    print("Loading relation encoder...")

    encoder_config = AutoConfig.from_pretrained(
        str(RE_MODEL_DIR)
    )

    encoder = AutoModel.from_config(
        encoder_config
    )

    encoder.resize_token_embeddings(
        len(re_tokenizer)
    )

    # ------------------------------
    # Build relation model
    # ------------------------------

    re_model = REModel(
        encoder=encoder,
        num_labels=len(re_config["labels"]),
        dropout=re_config.get("dropout", 0.1)
    )

    weights_path = RE_MODEL_DIR / "re_model.pt"

    if not weights_path.exists():

        raise FileNotFoundError(
            f"Relation model weights not found: {weights_path}"
        )

    try:

        state_dict = torch.load(
            weights_path,
            map_location="cpu",
            weights_only=True
        )

    except TypeError:

        # Compatibility with older PyTorch versions.
        state_dict = torch.load(
            weights_path,
            map_location="cpu"
        )

    re_model.load_state_dict(state_dict)

    re_model.to(device)
    re_model.eval()

    REL_LABELS = re_config["labels"]

    print(
        "Relation model loaded.",
        REL_LABELS
    )

    _models_loaded = True

    print("Both models loaded successfully.")


# ============================================================
# 7. CONVERT EXTRACTED PAGES INTO SENTENCES
# ============================================================

def pages_to_sentences(pages):

    """
    Convert the common PDF extraction output into sentences.

    This function does not read PDFs.

    Input:
        [
            {"page": 1, "text": "..."},
            {"page": 2, "text": "..."}
        ]

    Output:
        [
            {
                "page": 1,
                "text": "...",
                "words": [...]
            }
        ]
    """

    sentences = []

    for page_item in pages:

        page_num = int(
            page_item["page"]
        )

        page_text = str(
            page_item.get("text", "")
        )

        if not page_text.strip():
            continue

        sentence_list = segmenter.segment(
            page_text
        )

        for sentence_text in sentence_list:

            words = WORD_PATTERN.findall(
                sentence_text
            )

            if not words:
                continue

            sentences.append({

                "page": page_num,

                "text": sentence_text,

                "words": words

            })

    return sentences


# ============================================================
# 8. CREATE NER CHUNKS
# ============================================================

def create_chunks(
    words,
    tokenizer,
    max_length=512,
    overlap_words=50
):

    """
    Split word sequences into chunks that fit the NER model.
    """

    word_token_counts = [

        len(ids)

        for ids in tokenizer(
            words,
            add_special_tokens=False
        )["input_ids"]

    ]

    chunks = []

    start = 0

    while start < len(words):

        end = start

        # Reserve room for special tokens.
        token_count = 2

        while end < len(words):

            n = word_token_counts[end]

            if token_count + n > max_length:
                break

            token_count += n
            end += 1

        if end == start:

            # Skip an individual word that exceeds the limit.
            start += 1
            continue

        chunks.append({

            "start": start,

            "end": end,

            "words": words[start:end]

        })

        if end == len(words):
            break

        start = max(
            start + 1,
            end - overlap_words
        )

    return chunks


# ============================================================
# 9. RUN NAMED ENTITY RECOGNITION
# ============================================================

def run_ner(sentences):

    """
    Run the trained token classification model.

    Returns:
        entities
        all_words
    """

    load_models()

    all_words = []
    all_pages = []
    all_sentence_ids = []

    # Flatten sentences while retaining page and sentence info.
    for sentence_id, sentence in enumerate(sentences):

        for word in sentence["words"]:

            all_words.append(word)

            all_pages.append(
                sentence["page"]
            )

            all_sentence_ids.append(
                sentence_id
            )

    chunks = create_chunks(
        all_words,
        ner_tokenizer,
        MAX_LENGTH,
        NER_OVERLAP_WORDS
    )

    entities = []

    o_id = next(
        (
            key
            for key, value in ner_id2label.items()
            if value == "O"
        ),
        0
    )

    for chunk_number, chunk in enumerate(

        tqdm(
            chunks,
            desc="NER",
            leave=False
        ),

        start=1

    ):

        words = chunk["words"]

        encoding = ner_tokenizer(
            words,
            is_split_into_words=True,
            truncation=False,
            return_tensors="pt"
        )

        word_ids = encoding.word_ids(
            batch_index=0
        )

        inputs = {
            key: value.to(device)
            for key, value in encoding.items()
        }

        with torch.inference_mode():

            logits = ner_model(
                **inputs
            ).logits

        token_predictions = (
            logits.argmax(dim=-1)[0]
            .cpu()
            .tolist()
        )

        # Use the first subword prediction for each word.
        word_pred_ids = {}

        for token_index, word_id in enumerate(word_ids):

            if word_id is None:
                continue

            if word_id not in word_pred_ids:

                word_pred_ids[word_id] = (
                    token_predictions[token_index]
                )

        labels = [

            ner_id2label.get(
                word_pred_ids.get(i, o_id),
                "O"
            )

            for i in range(len(words))

        ]

        i = 0

        while i < len(words):

            label = labels[i]

            if not label.startswith("B-"):

                i += 1
                continue

            entity_type = label[2:]

            entity_words = [words[i]]

            j = i + 1

            while (
                j < len(words)
                and labels[j] == f"I-{entity_type}"
            ):

                entity_words.append(words[j])

                j += 1

            global_start = chunk["start"] + i
            global_end = chunk["start"] + j

            # Avoid duplicate entities from overlapping chunks.
            #
            # The first chunk keeps all its entities.
            # Later chunks keep entities starting after the
            # overlap region.
            is_first_chunk = (
                chunk_number == 1
            )

            is_new_region = (
                global_start
                >= chunk["start"] + NER_OVERLAP_WORDS
            )

            if is_first_chunk or is_new_region:

                entities.append({

                    "entity": " ".join(entity_words),

                    "type": entity_type,

                    "page": all_pages[global_start],

                    "start_word": global_start,

                    "end_word": global_end,

                    "sentence_id": all_sentence_ids[
                        global_start
                    ]

                })

            i = j

    # Remove duplicate entities.
    unique_entities = {}

    for entity in entities:

        key = (
            entity["start_word"],
            entity["end_word"],
            entity["type"]
        )

        unique_entities[key] = entity

    entities = sorted(
        unique_entities.values(),
        key=lambda entity: entity["start_word"]
    )

    return entities, all_words


# ============================================================
# 10. GENERATE RELATION CANDIDATES
# ============================================================

def generate_relation_candidates(
    entities,
    sentence_offsets,
    all_words,
    max_sentence_gap=1
):

    """
    Generate entity pairs from the same or nearby sentences.

    Only entities recognized as Method, Task, or Dataset
    are passed to this function.
    """

    candidates = []

    by_sentence = defaultdict(list)

    for entity in entities:

        by_sentence[
            entity["sentence_id"]
        ].append(entity)

    for subject in entities:

        subject_sentence = subject["sentence_id"]

        first_nearby = max(
            0,
            subject_sentence - max_sentence_gap
        )

        last_nearby = min(
            len(sentence_offsets) - 2,
            subject_sentence + max_sentence_gap
        )

        for object_sentence in range(
            first_nearby,
            last_nearby + 1
        ):

            for obj in by_sentence.get(
                object_sentence,
                []
            ):

                if subject is obj:
                    continue

                first_sentence = min(
                    subject_sentence,
                    object_sentence
                )

                last_sentence = max(
                    subject_sentence,
                    object_sentence
                )

                lo = sentence_offsets[
                    first_sentence
                ]

                hi = sentence_offsets[
                    last_sentence + 1
                ]

                context_tokens = all_words[lo:hi]

                subject_tuple = (

                    subject["start_word"] - lo,

                    subject["end_word"] - lo - 1,

                    subject["re_type"]

                )

                object_tuple = (

                    obj["start_word"] - lo,

                    obj["end_word"] - lo - 1,

                    obj["re_type"]

                )

                candidates.append({

                    "subject": subject,

                    "object": obj,

                    "tokens": context_tokens,

                    "subj": subject_tuple,

                    "obj": object_tuple,

                    "context": " ".join(context_tokens)

                })

    return candidates


# ============================================================
# 11. PREDICT RELATIONS
# ============================================================

def predict_relations(
    encoded_candidates,
    batch_size=32
):

    """
    Predict relations for encoded entity-pair candidates.

    Relations below MIN_REL_CONFIDENCE are discarded.
    """

    load_models()

    predictions = []

    if not encoded_candidates:
        return predictions

    for start in tqdm(

        range(
            0,
            len(encoded_candidates),
            batch_size
        ),

        desc="Relations",
        leave=False

    ):

        batch_items = encoded_candidates[
            start:start + batch_size
        ]

        max_len = max(
            len(item["input_ids"])
            for item in batch_items
        )

        input_ids = torch.full(

            (len(batch_items), max_len),

            re_tokenizer.pad_token_id,

            dtype=torch.long

        )

        attention_mask = torch.zeros(

            (len(batch_items), max_len),

            dtype=torch.long

        )

        subj_positions = []
        obj_positions = []

        for i, item in enumerate(batch_items):

            ids = item["input_ids"]

            input_ids[i, :len(ids)] = torch.tensor(
                ids,
                dtype=torch.long
            )

            attention_mask[i, :len(ids)] = 1

            subj_positions.append(
                item["subj_pos"]
            )

            obj_positions.append(
                item["obj_pos"]
            )

        input_ids = input_ids.to(device)

        attention_mask = attention_mask.to(device)

        subj_positions = torch.tensor(
            subj_positions,
            dtype=torch.long,
            device=device
        )

        obj_positions = torch.tensor(
            obj_positions,
            dtype=torch.long,
            device=device
        )

        with torch.inference_mode():

            logits = re_model(

                input_ids=input_ids,

                attention_mask=attention_mask,

                subj_pos=subj_positions,

                obj_pos=obj_positions

            )

            probabilities = torch.softmax(
                logits,
                dim=-1
            )

            confidence, predicted_ids = (
                probabilities.max(dim=-1)
            )

        for i, item in enumerate(batch_items):

            predicted_id = predicted_ids[i].item()

            label = REL_LABELS[predicted_id]

            score = confidence[i].item()

            if label == "NO_RELATION":
                continue

            if score < MIN_REL_CONFIDENCE:
                continue

            candidate = item["candidate"]

            predictions.append({

                "subject": candidate["subject"]["entity"],

                "subject_type": candidate["subject"]["re_type"],

                "relation": label,

                "object": candidate["object"]["entity"],

                "object_type": candidate["object"]["re_type"],

                "confidence": round(score, 4),

                "subject_page": candidate["subject"]["page"],

                "object_page": candidate["object"]["page"],

                "context": candidate["context"]

            })

    return predictions


# ============================================================
# 12. MAIN FUNCTION FOR STREAMLIT
# ============================================================

def process_paper_pages(pages):

    """
    Run NER and relation extraction on already-extracted pages.

    This is the main function called by kg_rag_pipeline.py.

    Parameters:
        pages:
            [
                {"page": 1, "text": "..."},
                {"page": 2, "text": "..."}
            ]

    Returns:
        ner_df:
            DataFrame containing recognized entities.

        rel_df:
            DataFrame containing predicted relationships.

        stats:
            Dictionary containing processing counts.
    """

    if not isinstance(pages, list):

        raise TypeError(
            "pages must be a list of dictionaries containing "
            "'page' and 'text'."
        )

    load_models()

    sentences = pages_to_sentences(
        pages
    )

    if not sentences:

        raise ValueError(
            "No readable sentences found for NER."
        )

    # -----------------------------------------
    # STEP 1: Named Entity Recognition
    # -----------------------------------------

    entities, all_words = run_ner(
        sentences
    )

    # -----------------------------------------
    # STEP 2: Keep supported entity types
    # -----------------------------------------

    re_entities = [

        {
            **entity,

            "re_type": ENTITY_TYPE_MAPPING[
                entity["type"]
            ]

        }

        for entity in entities

        if entity["type"] in ENTITY_TYPE_MAPPING

    ]

    # -----------------------------------------
    # STEP 3: Calculate sentence offsets
    # -----------------------------------------

    sentence_offsets = [0]

    for sentence in sentences:

        sentence_offsets.append(

            sentence_offsets[-1]
            + len(sentence["words"])

        )

    # -----------------------------------------
    # STEP 4: Generate relation candidates
    # -----------------------------------------

    candidates = generate_relation_candidates(

        re_entities,

        sentence_offsets,

        all_words,

        MAX_SENTENCE_GAP

    )

    # -----------------------------------------
    # STEP 5: Encode candidates
    # -----------------------------------------

    encoded_candidates = []

    for candidate in candidates:

        marked_text = mark_words(

            candidate["tokens"],

            candidate["subj"],

            candidate["obj"]

        )

        encoded = encode_marked(

            marked_text,

            re_tokenizer,

            MAX_LENGTH

        )

        if encoded is not None:

            encoded_candidates.append({

                **encoded,

                "candidate": candidate

            })

    # -----------------------------------------
    # STEP 6: Predict relations
    # -----------------------------------------

    relations = predict_relations(

        encoded_candidates,

        RE_BATCH_SIZE

    )

    # -----------------------------------------
    # STEP 7: Convert results into DataFrames
    # -----------------------------------------

    ner_df = pd.DataFrame(
        entities,
        columns=[
            "entity",
            "type",
            "page",
            "start_word",
            "end_word",
            "sentence_id"
        ]
    )

    rel_df = pd.DataFrame(
        relations,
        columns=[
            "subject",
            "subject_type",
            "relation",
            "object",
            "object_type",
            "confidence",
            "subject_page",
            "object_page",
            "context"
        ]
    )

    # -----------------------------------------
    # STEP 8: Statistics
    # -----------------------------------------

    stats = {

        "sentences": len(sentences),

        "words": len(all_words),

        "entities": len(entities),

        "re_entities": len(re_entities),

        "candidate_pairs": len(candidates),

        "encoded_pairs": len(encoded_candidates),

        "relations": len(relations)

    }

    return ner_df, rel_df, stats


# ============================================================
# 13. OPTIONAL BACKWARD-COMPATIBILITY FUNCTION
# ============================================================

def process_paper(pdf_path):

    """
    Optional standalone PDF processing function.

    The Research Gap Finder Streamlit app should use
    process_paper_pages() instead, so all systems use
    the same PDF extraction output.
    """

    import fitz

    pages = []

    with fitz.open(pdf_path) as pdf:

        for page_number, page in enumerate(
            pdf,
            start=1
        ):

            text = page.get_text("text")

            if text.strip():

                pages.append({

                    "page": page_number,

                    "text": text.strip()

                })

    return process_paper_pages(pages)
