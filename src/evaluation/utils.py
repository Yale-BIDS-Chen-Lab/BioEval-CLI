import json
import numpy as np
import csv
import os
from src.evaluation import metrics
from scipy.stats import ranksums, wilcoxon
from src.core.config import TASK_METRICS
from src.evaluation.data_processing import load_normalized_data

metrics_map = {
    'rouge1': metrics.compute_rouge1,
    'rouge2': metrics.compute_rouge2,
    'rougeL': metrics.compute_rougeL,
    'bertscore': metrics.compute_bertscore,
    'bartscore': metrics.compute_bartscore,
    'accuracy': metrics.compute_accuracy,
    'macro_f1': metrics.compute_macro_f1,
    'weighted_f1': metrics.compute_weighted_f1,
    'meteor': metrics.compute_meteor,
    'exact_match_precision': metrics.compute_exact_match_precision,
    'exact_match_recall': metrics.compute_exact_match_recall,
    'exact_match_f1': metrics.compute_exact_match_f1,
}


# --------------------------------------------------------------- statistics --
# Aligned with the BioEval web app (inference-service/src/statistics/handler.py):
#   * the bootstrap resamples the PER-EXAMPLE metric values (with replacement)
#     and takes their mean; the bootstrap distribution is used ONLY for the
#     confidence interval, never for the significance test;
#   * the significance test runs on the RAW per-example paired scores;
#   * no random seed is set (results vary run to run, like the web app).

def bootstrap_stats(scores, sample_size=40, n_boot=1000):
    """Bootstrap mean/std/95% CI from a list of per-example metric values."""
    arr = np.array(scores, dtype=float)
    if len(arr) == 0:
        return {"mean": 0, "std": 0, "ci_low": 0, "ci_high": 0}

    actual_sample_size = min(sample_size, len(arr))
    boot_means = np.array([
        np.random.choice(arr, actual_sample_size, replace=True).mean()
        for _ in range(n_boot)
    ])

    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    return {
        "mean": round(float(boot_means.mean()), 4),
        "std": round(float(boot_means.std(ddof=1)), 4),
        "ci_low": round(float(ci_low), 4),
        "ci_high": round(float(ci_high), 4),
    }


def run_statistical_analysis(models, sample_size=40, n_boot=1000, test_method="signed-rank"):
    """Pairwise significance tests + bootstrap CI for each model/metric.

    Args:
        models: { modelName: { metricName: [per-example values] } }
        sample_size: bootstrap sample size (clamped to the number of values)
        n_boot: number of bootstrap iterations
        test_method: "signed-rank" (paired Wilcoxon) or "rank-sum" (unpaired)
    """
    if test_method not in {"signed-rank", "rank-sum"}:
        test_method = "signed-rank"

    model_names = list(models.keys())

    all_metric_sets = [set(m.keys()) for m in models.values()]
    if not all_metric_sets:
        return {"testMethod": test_method, "bootstrap": {}, "pairwise": []}
    common_metrics = sorted(set.intersection(*all_metric_sets))

    bootstrap_results = {}
    for model_name, model_metrics in models.items():
        bootstrap_results[model_name] = {}
        for metric in common_metrics:
            bootstrap_results[model_name][metric] = bootstrap_stats(
                model_metrics.get(metric, []), sample_size, n_boot
            )

    pairwise_results = []
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            model_a, model_b = model_names[i], model_names[j]
            for metric in common_metrics:
                scores_a = np.asarray(models[model_a].get(metric, []), dtype=float)
                scores_b = np.asarray(models[model_b].get(metric, []), dtype=float)
                if len(scores_a) < 2 or len(scores_b) < 2:
                    continue

                if test_method == "signed-rank":
                    if len(scores_a) != len(scores_b):
                        continue
                    differences = scores_a - scores_b
                    if np.allclose(differences, 0):
                        stat, p_value = 0.0, 1.0
                    else:
                        stat, p_value = wilcoxon(
                            scores_a,
                            scores_b,
                            zero_method="wilcox",
                            alternative="two-sided",
                        )
                else:
                    stat, p_value = ranksums(scores_a, scores_b)

                pairwise_results.append({
                    "modelA": model_a,
                    "modelB": model_b,
                    "metric": metric,
                    "testMethod": test_method,
                    "statistic": round(float(stat), 4),
                    "p_value": float(p_value),
                })

    return {
        "testMethod": test_method,
        "bootstrap": bootstrap_results,
        "pairwise": pairwise_results,
    }


def _find_dataset_files(dataset, directory):
    """Return sorted full paths of {dataset}_{8-hex-hash}.json files."""
    all_files = [f for f in os.listdir(directory) if f.endswith('.json')]
    dataset_files = []
    for f in all_files:
        name_without_ext = f[:-5]
        if name_without_ext.startswith(f"{dataset}_"):
            remainder = name_without_ext[len(dataset) + 1:]
            if len(remainder) == 8 and all(c in '0123456789abcdef' for c in remainder):
                dataset_files.append(f)
    if not dataset_files:
        available_datasets = set()
        for f in all_files:
            parts = f[:-5].split('_')
            if len(parts) > 1:
                available_datasets.add('_'.join(parts[:-1]))
        raise ValueError(
            f"No evaluation files found for dataset '{dataset}' in {directory}. "
            f"Available datasets: {sorted(available_datasets)}"
        )
    return sorted(os.path.join(directory, f) for f in dataset_files)


def _per_example_scores(json_file, metric):
    """Read per-example metric values from a CLI evaluation output file.

    Matches the web app, which runs statistics on per-example scores. Only
    metrics computed per example (see individual_metrics in the pipeline) are
    available here; corpus-level metrics (e.g. macro_f1, exact_match_f1)
    have no per-example values and cannot be compared this way.
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    records = data.get('records', data) if isinstance(data, dict) else data
    scores = []
    for rec in records:
        individual = (rec or {}).get('individual_metrics') or {}
        if metric not in individual:
            raise ValueError(
                f"Metric '{metric}' has no per-example scores in "
                f"{os.path.basename(json_file)}. Statistical comparison requires a "
                f"per-example metric (e.g. rouge1/2/L, bertscore, bartscore, meteor, "
                f"accuracy); available here: {sorted(individual.keys())}."
            )
        scores.append(float(individual[metric]))
    return scores


def compare_models(dataset, directory, output_csv, metric,
                   sample_size=40, n_boot=1000, test_method="signed-rank"):
    """Compare every pair of model outputs for a dataset on one per-example
    metric and write bootstrap CIs + pairwise p-values to a CSV.

    Mirrors the web app: bootstrap CI from resampled per-example means, and a
    paired (signed-rank) or unpaired (rank-sum) significance test on the raw
    per-example scores.
    """
    if metric not in metrics_map:
        raise ValueError(
            f"Unsupported metric: {metric}. Supported metrics are: {list(metrics_map.keys())}"
        )

    json_files = _find_dataset_files(dataset, directory)

    models = {}
    for jf in json_files:
        name = os.path.basename(jf).replace('.json', '')
        models[name] = {metric: _per_example_scores(jf, metric)}

    result = run_statistical_analysis(
        models, sample_size=sample_size, n_boot=n_boot, test_method=test_method
    )
    boot = result["bootstrap"]
    pairwise = result["pairwise"]

    output_dir = os.path.dirname(output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = [
            'Model 1', 'Model 2',
            f'Model 1 Mean {metric}', f'Model 1 Std {metric}', f'Model 1 95% CI {metric}',
            f'Model 2 Mean {metric}', f'Model 2 Std {metric}', f'Model 2 95% CI {metric}',
            'p-value', 'test_method',
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for pr in pairwise:
            a, b = pr['modelA'], pr['modelB']
            ba, bb = boot[a][metric], boot[b][metric]
            writer.writerow({
                'Model 1': a,
                'Model 2': b,
                f'Model 1 Mean {metric}': ba['mean'],
                f'Model 1 Std {metric}': ba['std'],
                f'Model 1 95% CI {metric}': [ba['ci_low'], ba['ci_high']],
                f'Model 2 Mean {metric}': bb['mean'],
                f'Model 2 Std {metric}': bb['std'],
                f'Model 2 95% CI {metric}': [bb['ci_low'], bb['ci_high']],
                'p-value': pr['p_value'],
                'test_method': pr['testMethod'],
            })
    return result


def validate_task_and_metric(task, metric):
    """Validate that the given task and metric are supported."""
    if task not in TASK_METRICS:
        raise ValueError(f"Invalid task: '{task}'. Valid tasks are: {list(TASK_METRICS.keys())}.")
    if metric not in TASK_METRICS[task]:
        raise ValueError(
            f"For the '{task}' task, the metric must be one of {TASK_METRICS[task]}. "
            f"'{metric}' is not allowed.")


def evaluate_model(dataset, task, json_file, metric, postprocessing=None, process_json_override=None):
    """Evaluate a single model output file by computing one corpus-level metric."""
    validate_task_and_metric(task, metric)

    metric_evaluator = metrics_map.get(metric)
    if not metric_evaluator:
        raise ValueError(f"Unsupported metric: {metric}")

    if process_json_override is not None:
        process_json = process_json_override
    elif postprocessing:
        from . import data_processing
        process_json = getattr(data_processing, postprocessing)
        if not process_json:
            raise ValueError(f"Postprocessing function '{postprocessing}' not found in data_processing.py")
    else:
        process_json = load_normalized_data

    golds, preds = process_json(json_file)
    if len(golds) != len(preds):
        raise ValueError("Mismatch between number of gold labels and predictions.")

    metric_value = metric_evaluator(preds, golds)
    return f"{metric} score: {round(metric_value, 4)}"
