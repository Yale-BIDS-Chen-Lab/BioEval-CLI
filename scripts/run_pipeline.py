#!/usr/bin/env python3
"""
BioEval Evaluation Pipeline

This script orchestrates the complete evaluation pipeline for biomedical NLP tasks.
All stages progressively update a single output file: {dataset}_{hash}.json

Pipeline stages:
1. Prompt Generation: Creates test prompts with optional few-shot examples
2. Model Inference: Runs LLM inference using various providers (OpenAI, vLLM, HuggingFace)
3. Postprocessing & Evaluation: Processes outputs and computes metrics (accuracy, F1, ROUGE, etc.)

Usage examples:
  Full pipeline (default) ............. python run_pipeline.py --config cfg.yml
  Prompt generation only .............. python run_pipeline.py --config cfg.yml --do-prompt-generation
  Prompt generation + inference ....... python run_pipeline.py --config cfg.yml --do-prompt-generation --do-inference
  Inference + evaluation .............. python run_pipeline.py --config cfg.yml --do-inference --do-evaluation
  Evaluation only ..................... python run_pipeline.py --config cfg.yml --do-evaluation

Output:
  Single file per configuration containing all pipeline data and progressive metadata updates.
"""

import argparse
import hashlib
import json
import os
import yaml
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.pipeline import (
    run_prompt_generation,
    run_model_inference,
    run_postprocessing,
    run_evaluation
)


def generate_config_hash(config, stage):
    """Generate a unique hash for the configuration based on the stage."""
    if stage == "prompts":
        # Hash based on prompt generation parameters
        # Derive shots from shot_indices if provided
        shot_indices = config['prompt'].get('shot_indices')
        shots = len(shot_indices.split(',')) if shot_indices else 0
        prompt_params = f"{config['dataset']}_{shots}_{config['prompt'].get('prompt_template', '')}_{config['prompt'].get('system_prompt', '')}"
        return hashlib.md5(prompt_params.encode()).hexdigest()[:8]
    
    elif stage == "inference":
        # Hash based on prompt hash + inference parameters
        prompt_hash = generate_config_hash(config, "prompts")
        model_name = config["model"]["name"]
        temperature = config["inference"].get("temperature", 0.7)
        max_tokens = config["inference"].get("max_new_tokens", config["inference"].get("max_tokens", 50))
        provider = config["model"].get("provider", "hf")
        
        inference_params = f"{prompt_hash}_{model_name}_{temperature}_{max_tokens}_{provider}"
        return hashlib.md5(inference_params.encode()).hexdigest()[:8]
    
    elif stage == "eval":
        # Hash based on inference hash + evaluation parameters
        inference_hash = generate_config_hash(config, "inference")
        metrics = sorted(config["evaluation"]["metrics"])
        postprocessing = config["evaluation"].get("postprocessing", "")
        
        eval_params = f"{inference_hash}_{'-'.join(metrics)}_{postprocessing}"
        return hashlib.md5(eval_params.encode()).hexdigest()[:8]
    
    else:
        raise ValueError(f"Unknown stage: {stage}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments with stage selection logic."""
    p = argparse.ArgumentParser(description="BioEval evaluation pipeline")
    p.add_argument("--config", required=True, help="YAML configuration file")
    
    # Stage selection flags
    p.add_argument("--do-prompt-generation", action="store_true",
                   help="Run prompt generation stage")
    p.add_argument("--do-inference", action="store_true", 
                   help="Run model inference stage")
    p.add_argument("--do-evaluation", action="store_true",
                   help="Run evaluation stage")
    
    args = p.parse_args()
    
    # Default: run the full pipeline if no do-flags were given
    if not (args.do_prompt_generation or args.do_inference or args.do_evaluation):
        args.do_prompt_generation = args.do_inference = args.do_evaluation = True
    
    # Only allow logical stage sequences (prompt → inference → evaluation)
    allowed_patterns = [
        (True,  True,  True),    # full pipeline
        (True,  True,  False),   # prompt + inference 
        (True,  False, False),   # prompt only
        (False, True,  True),    # inference + evaluation
        (False, False, True),    # evaluation only
    ]
    pattern = (args.do_prompt_generation, args.do_inference, args.do_evaluation)
    if pattern not in allowed_patterns:
        p.error(f"Illegal combination of --do-* flags: {pattern}. "
                "See header comments for allowed mixes.")

    return args


def main(config: dict,
         do_prompt_generation: bool,
         do_inference: bool,
         do_evaluation: bool) -> None:
    """Execute the BioEval evaluation pipeline.
    
    This function orchestrates the pipeline stages, progressively updating a single output file
    by default, or separate files if custom output_file paths are specified in the config.
    
    Pipeline stages:
    1. Prompt Generation: Creates test prompts and saves with metadata
    2. Model Inference: Adds model outputs (reads from prompt file, writes to inference file)
    3. Postprocessing & Evaluation: Adds processed outputs and metrics (reads from inference file)
    
    Args:
        config: Configuration dictionary loaded from YAML
        do_prompt_generation: Whether to run prompt generation stage
        do_inference: Whether to run model inference stage  
        do_evaluation: Whether to run evaluation stage (includes postprocessing)
        
    Note:
        By default, all stages use a single file: {dataset}_{hash}.json
        Users can specify custom output_file in prompt, inference, or evaluation sections
        for more granular control over intermediate and final outputs.
    """
    # Extract configuration parameters
    task = config["task"]
    dataset = config["dataset"]
    model_name = config["model"]["name"]
    
    # Generate default output filename using inference hash (includes prompt config)
    inference_hash = generate_config_hash(config, "inference")
    default_output_file = f"data/outputs/{dataset}_{inference_hash}.json"
    
    # Determine output files for each stage (allowing custom paths)
    prompt_output_file = config["prompt"].get("output_file") or default_output_file
    inference_output_file = config["inference"].get("output_file") or prompt_output_file
    eval_output_file = config["evaluation"].get("output_file") or inference_output_file
    
    provider = config["model"].get("provider", "vllm")

    # Validate local model path early
    if provider == "local":
        local_path = config["model"].get("path", model_name)
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Local model path not found: {local_path}")

    # ========== Prompt Generation ==========
    if do_prompt_generation:
        run_prompt_generation(config, prompt_output_file)
        print(f"Prompts generated and saved to: {prompt_output_file}")
    else:
        print(f"Skipping prompt generation; using existing file: {prompt_output_file}")

    # ========== Model Inference ==========
    if do_inference:
        # Input is from prompt stage, output goes to inference file
        run_model_inference(config, prompt_output_file, inference_output_file)
        print(f"Inference completed and saved to: {inference_output_file}")
    else:
        print(f"Skipping inference; using existing file: {inference_output_file}")

    # ========== Postprocessing & Evaluation ==========
    if do_evaluation:
        # Postprocessing is done as part of evaluation
        records = run_postprocessing(config, inference_output_file)
        eval_data = run_evaluation(config, records, inference_output_file)
        
        # Save evaluation results to specified file
        os.makedirs(os.path.dirname(os.path.abspath(eval_output_file)), exist_ok=True)
        with open(eval_output_file, 'w') as f:
            json.dump(eval_data, f, indent=2)
        
        print(f"Evaluation completed and saved to: {eval_output_file}")
    else:
        print("Evaluation skipped")


if __name__ == "__main__":
    # Parse command line arguments and load configuration
    ns = parse_args()
    with open(ns.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Execute the pipeline with specified stages
    main(cfg,
         do_prompt_generation=ns.do_prompt_generation,
         do_inference=ns.do_inference,
         do_evaluation=ns.do_evaluation)