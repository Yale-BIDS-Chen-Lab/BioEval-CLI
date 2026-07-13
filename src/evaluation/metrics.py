"""Evaluation metrics, aligned with the BioEval web app
(inference-service/src/evaluation/callables/{individual,aggregate}.py).

Metric models are loaded lazily on first use, so importing this module does not
require torch/transformers; a task that never touches a neural metric never
loads one. Device selection (cuda > mps > cpu) with a CPU fallback matches the
web app, so generation metrics run on Mac/CPU as well as GPU.
"""
import os
import sys

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

# Vendored BARTScore lib on path (relative to project root).
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(_project_root, "BARTScore"))

# Lazily-loaded singletons — created on first use, not at import time.
_rouge = None
_bertscore = None
_meteor = None
_bart_scorer = None
_metric_device = None


def get_metric_device():
    """Best available device for metric models (cuda > mps > cpu).

    torch is imported lazily here so that importing this module needs no torch.
    """
    global _metric_device
    if _metric_device is not None:
        return _metric_device
    try:
        import torch

        if torch.cuda.is_available():
            _metric_device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            _metric_device = "mps"
        else:
            _metric_device = "cpu"
    except Exception:
        _metric_device = "cpu"
    return _metric_device


def _get_rouge():
    global _rouge
    if _rouge is None:
        import evaluate

        print("Initializing ROUGE scorer...")
        _rouge = evaluate.load("rouge")
    return _rouge


def _get_bertscore():
    global _bertscore
    if _bertscore is None:
        import evaluate

        print("Initializing BERTScore model...")
        _bertscore = evaluate.load("bertscore")
    return _bertscore


def _get_meteor():
    global _meteor
    if _meteor is None:
        import evaluate

        print("Initializing METEOR scorer...")
        _meteor = evaluate.load("meteor")
    return _meteor


def _get_bart_scorer():
    global _bart_scorer
    if _bart_scorer is None:
        from bart_score import BARTScorer

        device = get_metric_device()
        print(f"Initializing BARTScore model on {device} (this happens only once)...")
        _bart_scorer = BARTScorer(device=device, checkpoint="facebook/bart-large-cnn")
    return _bart_scorer


# ----------------------------------------------- per-example (individual) --
# ROUGE with use_aggregator=False returns the per-sample F-measure list, which
# we mean for the corpus score — matching the web app (not HF's BootstrapAggregator).
def _rouge_individual(predictions, references, variant):
    try:
        res = _get_rouge().compute(
            predictions=list(predictions), references=list(references), use_aggregator=False
        )
        if res is None:
            return [0.0] * len(predictions)
        return [float(x) for x in res[variant]]
    except Exception as e:
        print(f"ROUGE error: {e}, returning zeros for {len(predictions)} samples")
        return [0.0] * len(predictions)


def compute_rouge1_individual(predictions, references):
    return _rouge_individual(predictions, references, "rouge1")


def compute_rouge2_individual(predictions, references):
    return _rouge_individual(predictions, references, "rouge2")


def compute_rougeL_individual(predictions, references):
    return _rouge_individual(predictions, references, "rougeL")


def compute_meteor_individual(predictions, references):
    scorer = _get_meteor()
    scores = []
    for p, r in zip(predictions, references):
        try:
            res = scorer.compute(predictions=[p], references=[r])
            scores.append(float(res["meteor"]) if res else 0.0)
        except Exception:
            scores.append(0.0)
    return scores


def compute_bertscore_individual(predictions, references):
    scorer = _get_bertscore()
    device = get_metric_device()
    preds = [str(x) if x is not None else "" for x in predictions]
    refs = [str(x) if x is not None else "" for x in references]
    try:
        res = scorer.compute(
            predictions=preds, references=refs,
            model_type="bert-base-multilingual-cased", device=device,
        )
        if res is None:
            return [0.0] * len(preds)
        return [float(x) for x in res["f1"]]
    except Exception as e:
        if device != "cpu":
            print(f"BERTScore failed on {device}: {e}. Retrying on CPU...")
            try:
                res = scorer.compute(
                    predictions=preds, references=refs,
                    model_type="bert-base-multilingual-cased", device="cpu",
                )
                if res is None:
                    return [0.0] * len(preds)
                return [float(x) for x in res["f1"]]
            except Exception as retry_error:
                print(f"BERTScore retry on CPU failed: {retry_error}, returning zeros")
                return [0.0] * len(preds)
        print(f"BERTScore error: {e}, returning zeros for {len(preds)} samples")
        return [0.0] * len(preds)


def compute_bartscore_individual(predictions, references):
    device = get_metric_device()
    try:
        scorer = _get_bart_scorer()
        return [float(x) for x in scorer.score(srcs=list(predictions), tgts=list(references), batch_size=8)]
    except Exception as e:
        if device != "cpu":
            print(f"BARTScore failed on {device}: {e}. Retrying on CPU...")
            try:
                from bart_score import BARTScorer

                global _bart_scorer
                _bart_scorer = BARTScorer(device="cpu", checkpoint="facebook/bart-large-cnn")
                return [float(x) for x in _bart_scorer.score(srcs=list(predictions), tgts=list(references), batch_size=8)]
            except Exception as retry_error:
                print(f"BARTScore retry on CPU failed: {retry_error}, returning zeros")
                return [0.0] * len(predictions)
        print(f"BARTScore error: {e}, returning zeros for {len(predictions)} samples")
        return [0.0] * len(predictions)


def compute_accuracy_individual(predictions, true_labels):
    def norm(x):
        return x.lower().strip() if isinstance(x, str) else x

    return [1.0 if norm(p) == norm(t) else 0.0 for p, t in zip(predictions, true_labels)]


# ------------------------------------------------------ corpus-level (mean) --
def compute_rouge1(predictions, references):
    return float(np.mean(compute_rouge1_individual(predictions, references)))


def compute_rouge2(predictions, references):
    return float(np.mean(compute_rouge2_individual(predictions, references)))


def compute_rougeL(predictions, references):
    return float(np.mean(compute_rougeL_individual(predictions, references)))


def compute_meteor(predictions, true_labels):
    return float(np.mean(compute_meteor_individual(predictions, true_labels)))


def compute_bertscore(predictions, references):
    return float(np.mean(compute_bertscore_individual(predictions, references)))


def compute_bartscore(predictions, references):
    return float(np.mean(compute_bartscore_individual(predictions, references)))


def compute_accuracy(predictions, true_labels):
    return accuracy_score(true_labels, predictions)


def compute_macro_f1(predictions, true_labels):
    return f1_score(true_labels, predictions, average="macro", zero_division=0)


def compute_weighted_f1(predictions, true_labels):
    return f1_score(true_labels, predictions, average="weighted", zero_division=0)


# ------------------------------------ entity exact match (web app aggregate) --
# Ported from callables/aggregate.py: build a one-hot matrix over the union of
# all (start, end, label) spans, then micro precision/recall/F1 via sklearn.
def _exact_match(predictions, references, variant):
    preds = np.asarray(predictions, dtype=object).ravel()
    refs = np.asarray(references, dtype=object).ravel()

    label_set = set()
    for sample in list(preds) + list(refs):
        if sample:
            label_set.update(tuple(span) for span in sample)
    label_list = list(label_set)
    idx = {lab: i for i, lab in enumerate(label_list)}

    n_samples = max(len(preds), len(refs))
    n_labels = len(label_list)
    y_pred = np.zeros((n_samples, n_labels), dtype=int)
    y_true = np.zeros((n_samples, n_labels), dtype=int)

    for i in range(n_samples):
        for span in (preds[i] if i < len(preds) and preds[i] else []):
            y_pred[i, idx[tuple(span)]] = 1
    for i in range(n_samples):
        for span in (refs[i] if i < len(refs) and refs[i] else []):
            y_true[i, idx[tuple(span)]] = 1

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )
    return {"precision": precision, "recall": recall, "f1": f1}[variant]


def compute_exact_match_precision(predictions, references):
    return _exact_match(predictions, references, "precision")


def compute_exact_match_recall(predictions, references):
    return _exact_match(predictions, references, "recall")


def compute_exact_match_f1(predictions, references):
    return _exact_match(predictions, references, "f1")
