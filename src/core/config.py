# Dictionary to store allowed metrics for each task
TASK_METRICS = {
    "ner": ["exact_match_precision", "exact_match_recall", "exact_match_f1"],
    "slc": ["macro_f1", "weighted_f1"],
    "mlc": ["macro_f1", "weighted_f1"],
    "mcq": ["accuracy", "macro_f1", "weighted_f1"],
    "generation": ["rouge1", "rouge2", "rougeL", "bertscore", "bartscore", "meteor"],
    # Add more tasks and their allowed metrics as needed
}
