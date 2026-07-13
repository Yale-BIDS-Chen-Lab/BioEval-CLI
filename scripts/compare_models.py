#!/usr/bin/env python3
"""
BioEval Model Comparison Script
===============================

Statistical comparison of model evaluation files, aligned with the BioEval web
app (inference-service/src/statistics/handler.py):

  * Bootstrap 95% confidence intervals are computed by resampling the
    PER-EXAMPLE metric values (with replacement) and taking their mean.
  * The significance test runs on the RAW per-example paired scores — either
    the paired Wilcoxon signed-rank test (default) or the unpaired
    Wilcoxon rank-sum test.
  * No random seed is set, so results vary run to run (as in the web app).

Only PER-EXAMPLE metrics can be compared (rouge1/2/L, bertscore, bartscore,
meteor, accuracy); corpus-level metrics such as macro_f1 or exact_match_f1
have no per-example scores.

Output
------
* `--output_file` – CSV of pairwise comparison results (per-model bootstrap
  mean/std/95% CI and the pairwise p-value).

Examples
--------
# Paired signed-rank comparison on ROUGE-L for the PLOS dataset (default test)
python3 scripts/compare_models.py --metric rougeL --dataset plos \
    --input_directory data/outputs --output_file plos_comparison.csv

# Unpaired rank-sum comparison on accuracy for MedQA
python3 scripts/compare_models.py --metric accuracy --dataset medqa \
    --test_method rank-sum --input_directory data/outputs \
    --output_file medqa_comparison.csv
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.utils import compare_models


def main() -> None:
    p = argparse.ArgumentParser(
        description="Statistical comparison of BioEval model evaluation files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument("--metric", required=True,
                   help="Per-example metric to compare (e.g. rougeL, bertscore, meteor, accuracy)")
    p.add_argument("--dataset", required=True,
                   help="Dataset to compare results from (e.g. plos, ncbi_disease, medqa)")
    p.add_argument("--input_directory", required=True,
                   help="Directory containing evaluation JSON files ({dataset}_{hash}.json)")
    p.add_argument("--output_file", required=True,
                   help="CSV file for pairwise comparison results")

    # Statistical settings (match the web app defaults)
    p.add_argument("--test_method", choices=["signed-rank", "rank-sum"], default="signed-rank",
                   help="Significance test: 'signed-rank' (paired Wilcoxon, default) or 'rank-sum' (unpaired)")
    p.add_argument("--n_samples", type=int, default=1000,
                   help="Number of bootstrap iterations (default: 1000)")
    p.add_argument("--sample_size", type=int, default=40,
                   help="Bootstrap sample size; clamped to the number of examples (default: 40)")

    args = p.parse_args()

    if not os.path.exists(args.input_directory):
        raise ValueError(f"Input directory does not exist: {args.input_directory}")

    print(f"Comparing models on '{args.metric}' for dataset '{args.dataset}' "
          f"using the {args.test_method} test...")

    result = compare_models(
        dataset=args.dataset,
        directory=args.input_directory,
        output_csv=args.output_file,
        metric=args.metric,
        sample_size=args.sample_size,
        n_boot=args.n_samples,
        test_method=args.test_method,
    )

    n_pairs = len(result.get("pairwise", []))
    print(f"\nModel comparison completed. {n_pairs} pairwise comparison(s) → {args.output_file}")


if __name__ == "__main__":
    main()
