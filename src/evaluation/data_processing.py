"""
Data Processing Module for BioEval

Postprocessing/parsing of model outputs per task, plus file-based (golds, preds)
loaders for scoring. Aligned with the BioEval web app
(inference-service/src/evaluation/parser.py):

- NER: parse <span class="...">entity</span> into token-index spans, extracting
       EVERY span class (case-insensitive), after stripping markdown code fences.
- MCQ: first-character / substring option extraction.
- MLC: label-substring matching -> binary vector.
- SLC/classification: normalized (lowercase + strip) text.
- Generation: raw text (no normalization) so cased metrics (BERTScore/BARTScore)
              match the web app, which scores raw text.
"""

import json
import os
import re
from typing import List, Tuple, Union

# =============================================================================
# NER (Named Entity Recognition) — token-index span parsing
# Ported from the web app's parser.extract_spans.
# =============================================================================

_SPAN_OPEN_RE = re.compile(r'<span\s+class="([^"]+)">')
_SPAN_CLOSE_RE = re.compile(r"</span>")


def _strip_code_fences(text: str) -> str:
    """Remove leading ```html / ``` and a trailing ``` (matches extract_spans)."""
    text = text.strip()
    if text.startswith("```html"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def process_ner_token_indices_single(html: str, entity_types=None) -> Tuple[List[str], List[List[Union[int, str]]]]:
    """Parse <span class="...">entity</span> HTML into token-index spans.

    Matches the web app (parser.extract_spans): markdown code fences are stripped
    first, EVERY span class is extracted (not only declared types), the class is
    lowercased, and end indices are inclusive. ``entity_types`` is accepted for
    backward compatibility but ignored — off-type spans are surfaced (and scored
    as false positives against a single-type reference) exactly as in the web app.

    Returns (tokens, spans) where each span is ``[start, end_inclusive, class_lower]``.
    """
    text = _strip_code_fences(html if isinstance(html, str) else "")

    tokens: List[str] = []
    spans: List[List[Union[int, str]]] = []
    tok_idx = 0
    cur_start = None
    cur_class = None

    for part in re.split(r"(</?span[^>]*>)", text):
        if not part:
            continue
        m_open = _SPAN_OPEN_RE.fullmatch(part)
        if m_open:
            cur_start = tok_idx
            cur_class = m_open.group(1).lower()
            continue
        if _SPAN_CLOSE_RE.fullmatch(part):
            if cur_start is not None and cur_class is not None:
                spans.append([cur_start, tok_idx - 1, cur_class])
            cur_start = cur_class = None
            continue
        piece_tokens = part.split()
        tokens.extend(piece_tokens)
        tok_idx += len(piece_tokens)

    return tokens, spans


def _lower_span_labels(spans) -> List[List[Union[int, str]]]:
    """Lowercase each [start, end, label] span's label — the web app lowercases
    both predicted and reference span labels before scoring."""
    out = []
    for sp in spans or []:
        if len(sp) >= 3:
            out.append([sp[0], sp[1], str(sp[2]).lower()])
        else:
            out.append(list(sp))
    return out


def process_ner_token_indices(file_path: str, entity_types=None) -> Tuple[list, list]:
    """Load a NER file and return (golds, preds) as token-index spans.

    Predictions are parsed from each record's ``output``; reference span labels
    are lowercased to match the parsed (lowercased) predictions.
    """
    with open(file_path, "r") as file:
        data = json.load(file)
    records = data.get("records", data) if isinstance(data, dict) else data

    golds, preds = [], []
    for entry in records:
        golds.append(_lower_span_labels(entry["reference"]))
        _, pred_spans = process_ner_token_indices_single(entry["output"])
        preds.append(pred_spans)
    return golds, preds


# Backward-compatible aliases
def process_ner_custom_single(html: str, entity_types=None):
    """Alias for process_ner_token_indices_single."""
    return process_ner_token_indices_single(html, entity_types)


def process_ner_custom(file_path: str, entity_types=None):
    """Alias for process_ner_token_indices."""
    return process_ner_token_indices(file_path, entity_types)


# =============================================================================
# Data loaders
# =============================================================================

def load_normalized_data(file_path: str) -> Tuple[List, List]:
    """Load (golds, preds) with lowercase + strip normalization.

    Used for SLC / single-label classification, where the label is matched as a
    normalized string.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, "r") as file:
        data = json.load(file)
        records = data if isinstance(data, list) else data.get("records", data)
        golds = [entry["reference"].lower().strip() for entry in records]
        preds = [entry["postprocessed_output"].lower().strip() for entry in records]
    return golds, preds


def load_raw_data(file_path: str) -> Tuple[List, List]:
    """Load (golds, preds) WITHOUT normalization.

    Used for generation so that case-sensitive metrics (BERTScore, BARTScore)
    match the web app, which scores the raw prediction and reference text.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, "r") as file:
        data = json.load(file)
        records = data if isinstance(data, list) else data.get("records", data)
        golds = [entry["reference"] for entry in records]
        preds = [entry["postprocessed_output"] for entry in records]
    return golds, preds


# =============================================================================
# MCQ (Multiple Choice Questions)
# =============================================================================

def process_mcq_custom_single(output: str, option_type: str) -> str:
    """Extract the selected option from a model output.

    option_type: "A-E" (a,b,c,d,e), "A-D" (a,b,c,d), or "Yes/No/Maybe".
    """
    if not output or not output.strip():
        return "missing"

    output = output.lower().strip()

    if option_type == "A-E":
        return output[0] if output[0] in {"a", "b", "c", "d", "e"} else "missing"
    elif option_type == "A-D":
        return output[0] if output[0] in {"a", "b", "c", "d"} else "missing"
    elif option_type == "Yes/No/Maybe":
        for label in ["yes", "no", "maybe"]:
            if label in output:
                return label
        return "missing"
    else:
        raise ValueError(f"Unsupported option_type: {option_type}. Use 'A-E', 'A-D', or 'Yes/No/Maybe'")


def process_mcq_custom(file_path: str, option_type: str) -> Tuple[List[str], List[str]]:
    """Load an MCQ file and return (golds, preds) of extracted options."""
    with open(file_path, "r") as file:
        data = json.load(file)

    golds = [process_mcq_custom_single(entry["reference"], option_type) for entry in data]
    preds = [process_mcq_custom_single(entry["output"], option_type) for entry in data]
    return golds, preds


# =============================================================================
# MLC (Multi-Label Classification)
# =============================================================================

def process_mlc_custom(file_path: str, label_string: str) -> Tuple[List[List[int]], List[List[int]]]:
    """Load an MLC file and return (golds, preds) as binary label vectors."""
    with open(file_path, "r") as file:
        data = json.load(file)

    golds = [process_mlc_custom_single(entry["reference"], label_string) for entry in data]
    preds = [process_mlc_custom_single(entry["output"], label_string) for entry in data]
    return golds, preds


def process_mlc_custom_single(output: str, label_string: str) -> List[int]:
    """Convert a model output to a binary label vector by matching labels
    (comma-separated) as case-insensitive substrings."""
    label_list = [label.strip().lower() for label in label_string.split(",")]
    output = output.lower()
    result = [0] * len(label_list)
    for index, choice in enumerate(label_list):
        if choice in output:
            result[index] = 1
    return result
