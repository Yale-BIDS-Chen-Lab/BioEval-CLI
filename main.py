#!/usr/bin/env python3
"""BioEval CLI entry point.

Run the end-to-end evaluation pipeline from the repository root, e.g.:

    python main.py --config config/model_cards/mcq/benchmark/medqa.yaml

Stage flags (--do-prompt-generation / --do-inference / --do-evaluation) select a
subset of stages; omit them to run the full pipeline. This is a thin wrapper
around scripts/run_pipeline.py so the documented `python main.py` command works.
Run it from the repository root: the pipeline invokes its stage scripts by
relative path (e.g. src/cli/prompts.py).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import yaml
import run_pipeline

if __name__ == "__main__":
    ns = run_pipeline.parse_args()
    with open(ns.config, "r") as f:
        cfg = yaml.safe_load(f)
    run_pipeline.main(
        cfg,
        do_prompt_generation=ns.do_prompt_generation,
        do_inference=ns.do_inference,
        do_evaluation=ns.do_evaluation,
    )
