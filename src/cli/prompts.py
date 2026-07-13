#!/usr/bin/env python3
"""
Generate prompts for zero-shot or few-shot evaluation.

Usage
-----
# Zero-shot (omit shot_indices):
python prompts.py --task mcq --dataset medmcqa \\
    --test_file data/test.json \\
    --system_prompt config/prompts/mcq/medmcqa.txt \\
    --output_file data/outputs/medmcqa_prompts.json

# 3-shot with explicit indices:
python prompts.py --task mcq --dataset medmcqa \\
    --test_file data/test.json \\
    --train_file data/train.json \\
    --system_prompt config/prompts/mcq/medmcqa.txt \\
    --shot_indices 0,5,10 \\
    --output_file data/outputs/medmcqa_prompts.json

Prompt Template Format
----------------------
Your prompt template file (.txt) must contain:

1. Required: {{input}} placeholder - replaced with each test example

2. For few-shot: {example_N_input} and {example_N_output} placeholders (0-indexed)
   - Number of placeholders must match number of shot_indices
   - Placeholders must be sequential: 0, 1, 2, ...

Zero-shot template example:
    Answer the question: {{input}}
    Answer:

3-shot template example (with shot_indices="0,5,10"):
    Answer the question.

    Q: {example_0_input}
    A: {example_0_output}

    Q: {example_1_input}
    A: {example_1_output}

    Q: {example_2_input}
    A: {example_2_output}

    Q: {{input}}
    A:
"""
import argparse
import datetime
import json
import os
import re
import pandas as pd
from typing import List, Dict, Callable, Union

# --------------------------------------------------------------------------- #
#                               I/O UTILITIES                                 #
# --------------------------------------------------------------------------- #

def _read_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in {".csv"}:
        return pd.read_csv(path)
    if ext in {".tsv"}:
        return pd.read_csv(path, sep="\t")
    if ext in {".xls", ".xlsx"}:
        return pd.read_excel(path)
    if ext == ".jsonl":
        return pd.read_json(path, orient="records", lines=True)
    if ext == ".json":                             
        return pd.read_json(path, orient="records")
    raise ValueError(f"Unsupported file extension: {ext}")


def _write_json(data: Union[List, Dict], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    else:
        raise ValueError("`output_file` must end with .json")


# --------------------------------------------------------------------------- #
#                       PROMPT-BUILDING FUNCTION FACTORY                      #
# --------------------------------------------------------------------------- #

def build_template(task_prompt: str,
                   train_examples: List[Dict[str, str]]) -> Callable[[str], str]:
    """
    Returns a function that renders a single prompt.

    Parameters
    ----------
    task_prompt     : Prompt template with placeholders.
    train_examples  : Few-shot examples, each a dict with keys `input` & `reference`.

    Returns
    -------
    template_func(task_input) -> str
    
    Required placeholder in task_prompt:
    - {{input}} : Will be replaced with the actual test input
    
    Optional placeholders for few-shot (0-indexed):
    - {example_N_input} : Will be replaced with train_examples[N]['input']
    - {example_N_output} : Will be replaced with train_examples[N]['reference']
    
    Example prompt template (zero-shot):
        "Answer the question.
        Q: {{input}}
        A:"
    
    Example prompt template (3-shot):
        "Answer the question.
        
        Example 0:
        Q: {example_0_input}
        A: {example_0_output}
        
        Example 1:
        Q: {example_1_input}
        A: {example_1_output}
        
        Example 2:
        Q: {example_2_input}
        A: {example_2_output}
        
        Now answer:
        Q: {{input}}
        A:"
    """
    # Validate that prompt template contains required {{input}} placeholder
    if "{{input}}" not in task_prompt:
        raise ValueError("Prompt template must contain {{input}} placeholder")

    # Count example placeholders in prompt template
    # Match patterns like {example_0_input}, {example_1_input}, etc.
    example_placeholders = set(re.findall(r'\{example_(\d+)_input\}', task_prompt))
    num_placeholders = len(example_placeholders)
    num_examples = len(train_examples)
    
    # Validate placeholder count matches shot_indices count
    if num_placeholders != num_examples:
        if num_placeholders > 0 and num_examples == 0:
            raise ValueError(
                f"Prompt template has {num_placeholders} example placeholder(s) "
                f"({{example_N_input}}) but no shot_indices provided. "
                f"Either add shot_indices or remove example placeholders from prompt."
            )
        elif num_placeholders == 0 and num_examples > 0:
            raise ValueError(
                f"shot_indices has {num_examples} index(es) but prompt template "
                f"has no example placeholders ({{example_N_input}}). "
                f"Add placeholders like {{example_0_input}}, {{example_0_output}}, etc."
            )
        else:
            raise ValueError(
                f"Mismatch: prompt template has {num_placeholders} example placeholder(s) "
                f"but shot_indices has {num_examples} index(es). They must match."
            )
    
    # Validate placeholder indices are sequential starting from 0
    if num_placeholders > 0:
        expected_indices = set(str(i) for i in range(num_placeholders))
        if example_placeholders != expected_indices:
            raise ValueError(
                f"Example placeholders must be sequential starting from 0. "
                f"Found: {sorted(example_placeholders, key=int)}, "
                f"expected: {sorted(expected_indices, key=int)}"
            )

    def template_func(task_input: str) -> str:
        prompt = task_prompt
        
        # Replace example placeholders: {example_N_input} and {example_N_output} (0-indexed)
        for i, ex in enumerate(train_examples):
            prompt = prompt.replace(f"{{example_{i}_input}}", ex.get('input', ''))
            prompt = prompt.replace(f"{{example_{i}_output}}", ex.get('reference', ''))
        
        # Replace the main input placeholder
        prompt = prompt.replace("{{input}}", task_input)
        
        return prompt

    return template_func


# --------------------------------------------------------------------------- #
#                                  MAIN                                       #
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt generator (zero-shot or few-shot)")
    parser.add_argument("--task", required=True, help="Task type (e.g., mcq, ner)")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g., medmcqa, ncbi_disease)")
    parser.add_argument("--test_file", required=True, help="Path to test data")
    parser.add_argument("--train_file", help="Path to training data (required for few-shot)")
    parser.add_argument("--system_prompt", required=True, help="Path to prompt template file")
    parser.add_argument("--output_file", required=True, help="Output file path (.json)")
    parser.add_argument(
        "--shot_indices",
        type=str,
        default=None,
        help=(
            "Comma-separated zero-based indices into train_file, e.g. '0,2,4'. "
            "Omit for zero-shot. Number of shots inferred from indices."
        ),
    )

    args = parser.parse_args()

    # -------------------------- load data ---------------------------------- #
    test_df = _read_table(args.test_file)
    
    # Determine shots from shot_indices
    if args.shot_indices:
        # Few-shot: parse indices and load train examples
        if not args.train_file:
            raise ValueError("--train_file required when using --shot_indices")
        
        train_df = _read_table(args.train_file)
        records = train_df.to_dict("records")
        
        # Parse indices
        idxs = [int(i) for i in args.shot_indices.split(",")]
        num_shots = len(idxs)
        
        # Validate indices are within range
        max_idx = max(idxs)
        if max_idx >= len(records):
            raise ValueError(
                f"shot_indices contains index {max_idx} but train_file only has {len(records)} rows"
            )
        
        train_examples = [records[i] for i in idxs]
    else:
        # Zero-shot: no train examples
        train_examples = []
        num_shots = 0

    # Load and validate prompt template
    with open(args.system_prompt, "r", encoding="utf-8") as f:
        task_prompt = f.read()

    # Build prompt function (validates {{input}} placeholder exists)
    tmpl = build_template(task_prompt, train_examples)

    # --------------------- generate and write prompts ---------------------- #
    prompts = []
    for row in test_df.to_dict("records"):
        prompts.append({
            "id": row.get("id"),
            "input": row.get("input"),
            "reference": row.get("reference"),
            "prompt": tmpl(row["input"])
        })

    # Create structured output with metadata
    output_data = {
        "dataset_info": {
            "task": args.task,
            "dataset": args.dataset,
            "test_file": args.test_file,
            "train_file": args.train_file if num_shots > 0 else None
        },
        "prompt_generation_metadata": {
            "generation_timestamp": datetime.datetime.now().isoformat(),
            "system_prompt_file": args.system_prompt,
            "shots": num_shots,
            "shot_indices": args.shot_indices
        },
        "records": prompts
    }

    _write_json(output_data, args.output_file)
    print(f"Wrote {len(prompts)} prompts ({num_shots}-shot) → {args.output_file}")

if __name__ == "__main__":
    main()
