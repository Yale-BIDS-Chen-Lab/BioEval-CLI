# Core pipeline orchestration logic

import subprocess
import tempfile
import json
import os
import datetime
from typing import Dict, List, Any

from src.tasks import get_task_handler
from src.evaluation import data_processing
from src.evaluation import metrics as metrics_module


def run_prompt_generation(config: Dict[str, Any], prompt_output_file: str) -> None:
    """Execute the prompt generation stage of the pipeline."""
    task = config["task"]
    dataset = config["dataset"]
    shot_indices = config["prompt"].get("shot_indices")
    
    # Build command line arguments for prompt generation script
    prompt_args = [
        "python3", "src/cli/prompts.py",
        "--task", task,
        "--dataset", dataset,
        "--test_file", config["data_files"]["test_file"],
        "--output_file", prompt_output_file,
        "--system_prompt", config["prompt"]["system_prompt"],
    ]

    # Add few-shot configuration if shot_indices provided
    if shot_indices:
        train_file = config["data_files"].get("train_file")
        if not train_file:
            raise ValueError("Few-shot requires train_file in data_files")
        
        prompt_args += ["--train_file", train_file]
        
        # Convert shot_indices to string format
        idx_str = (",".join(str(i) for i in shot_indices)
                   if isinstance(shot_indices, (list, tuple))
                   else str(shot_indices))
        prompt_args += ["--shot_indices", idx_str]
    
    subprocess.run(prompt_args, check=True)


def run_model_inference(config: Dict[str, Any], input_file: str, output_file: str) -> None:
    """Execute the model inference stage of the pipeline.
    
    Note: input_file and output_file can be the same file for progressive updates.
    """
    provider = config["model"].get("provider", "vllm")
    model_name = config["model"]["name"]
    
    if provider == "azure":
        _run_azure_inference(config, input_file, output_file, model_name)
    elif provider == "vllm":
        _run_vllm_inference(config, input_file, output_file, model_name)
    elif provider == "hf":
        _run_hf_inference(config, input_file, output_file, model_name)
    elif provider == "google":
        _run_google_inference(config, input_file, output_file, model_name)
    elif provider == "anthropic":
        _run_anthropic_inference(config, input_file, output_file, model_name)
    elif provider == "local":
        _run_local_inference(config, input_file, output_file, model_name)
    else:
        raise ValueError(f"Unknown provider '{provider}'")


def _run_azure_inference(config: Dict[str, Any], input_file: str, output_file: str, model_name: str) -> None:
    """Run Azure OpenAI API inference."""
    inference_script = "src/providers/azure.py"
    cfg = config["inference"]
    inference_args = [
        "python3", inference_script,
        "--endpoint", cfg["endpoint"],
        "--api_key", cfg["api_key"],
        "--api_version", cfg["api_version"],
        "--model", model_name,
        "--input_file", input_file,
        "--output_file", output_file,
    ]
    # Add optional OpenAI parameters
    for key in (
        "temperature", "top_p", "max_tokens", "max_completion_tokens",
        "n", "best_of", "stop", "echo", "stream", "suffix",
        "presence_penalty", "frequency_penalty", "logit_bias", "user"
    ):
        val = cfg.get(key)
        if val is not None:
            inference_args += [f"--{key}", str(val)]
    subprocess.run(inference_args, check=True)


def _run_vllm_inference(config: Dict[str, Any], input_file: str, output_file: str, model_name: str) -> None:
    """Run vLLM inference."""
    inference_script = "src/providers/vllm_provider.py"
    cfg = config["inference"]
    inference_args = [
        "python3", inference_script,
        "--model", model_name,
        "--input_file", input_file,
        "--output_file", output_file,
    ]
    # Add optional vLLM parameters
    for key in [
        "temperature", "top_p", "top_k", "min_p", "repetition_penalty",
        "length_penalty", "presence_penalty", "frequency_penalty",
        "stop", "stop_token_ids", "include_stop_str_in_output",
        "ignore_eos", "max_tokens", "min_tokens", "logprobs",
        "prompt_logprobs", "detokenize", "skip_special_tokens",
        "spaces_between_special_tokens", "logits_processors",
        "guided_json", "guided_regex", "guided_choice",
        "guided_grammar", "guided_decoding_backend", "guided_whitespace_pattern",
        "use_beam_search", "best_of", "early_stopping",
        "n", "use_tqdm", "lora_request", "chat_template",
        "add_generation_prompt", "continue_final_message",
        "tools", "tool_choice", "guided_json_object", "parallel_tool_calls",
        "truncate_prompt_tokens", "allowed_token_ids", "disallowed_token_ids",
        "min_tokens", "truncate_prompt_tokens", "output_kind",
        "renormalize_logits"
    ]:
        val = cfg.get(key)
        if val is not None:
            inference_args += [f"--{key}", str(val)]
    subprocess.run(inference_args, check=True)


def _run_hf_inference(config: Dict[str, Any], input_file: str, output_file: str, model_name: str) -> None:
    """Run Hugging Face inference."""
    inference_script = "src/providers/huggingface.py"
    cfg = config["inference"]
    inference_args = [
        "python3", inference_script,
        "--model", model_name,
        "--input_file", input_file,
        "--output_file", output_file,
    ]
    # Add optional Hugging Face parameters
    for key in [
        "max_new_tokens", "do_sample", "temperature", "top_k", "top_p",
        "typical_p", "epsilon_cutoff", "eta_cutoff", "diversity_penalty",
        "repetition_penalty", "encoder_repetition_penalty", "length_penalty",
        "no_repeat_ngram_size", "bad_words_ids", "force_words_ids",
        "renormalize_logits", "constraints", "forced_bos_token_id",
        "forced_eos_token_id", "remove_invalid_values", "exponential_decay_length_penalty",
        "suppress_tokens", "begin_suppress_tokens", "forced_decoder_ids",
        "sequence_bias", "guidance_scale", "low_memory", "num_beams",
        "num_beam_groups", "penalty_alpha", "use_cache"
    ]:
        val = cfg.get(key)
        if val is not None:
            inference_args += [f"--{key}", str(val)]
    subprocess.run(inference_args, check=True)


def _run_google_inference(config: Dict[str, Any], input_file: str, output_file: str, model_name: str) -> None:
    """Run Google Gemini inference."""
    cfg = config["inference"]
    inference_args = [
        "python3", "src/providers/google.py",
        "--model", model_name,
        "--input_file", input_file,
        "--output_file", output_file,
    ]
    if cfg.get("api_key"):
        inference_args += ["--api_key", str(cfg["api_key"])]
    for key in ("temperature", "top_p", "max_tokens"):
        val = cfg.get(key)
        if val is not None:
            inference_args += [f"--{key}", str(val)]
    subprocess.run(inference_args, check=True)


def _run_anthropic_inference(config: Dict[str, Any], input_file: str, output_file: str, model_name: str) -> None:
    """Run Anthropic (Claude) inference."""
    cfg = config["inference"]
    inference_args = [
        "python3", "src/providers/anthropic.py",
        "--model", model_name,
        "--input_file", input_file,
        "--output_file", output_file,
    ]
    if cfg.get("api_key"):
        inference_args += ["--api_key", str(cfg["api_key"])]
    for key in ("temperature", "max_tokens"):
        val = cfg.get(key)
        if val is not None:
            inference_args += [f"--{key}", str(val)]
    subprocess.run(inference_args, check=True)


def _run_local_inference(config: Dict[str, Any], input_file: str, output_file: str, model_name: str) -> None:
    """Run local model inference."""
    inference_script = "src/providers/local.py"
    cfg = config["inference"]
    inference_args = [
        "python3", inference_script,
        "--model", model_name,
        "--input_file", input_file,
        "--output_file", output_file,
    ]
    # Add optional local parameters
    for key in [
        "max_new_tokens", "do_sample", "temperature", "top_k", "top_p",
        "num_beams", "repetition_penalty", "length_penalty"
    ]:
        val = cfg.get(key)
        if val is not None:
            inference_args += [f"--{key}", str(val)]
    subprocess.run(inference_args, check=True)


def run_postprocessing(config: Dict[str, Any], inference_output_file: str) -> List[Dict[str, Any]]:
    """Execute the postprocessing stage and return processed records."""
    task = config["task"]
    dataset = config["dataset"]
    
    # Load inference results
    with open(inference_output_file, 'r') as f:
        data = json.load(f)
    
    records = data["records"]
    
    # Determine postprocessing function and label string
    postprocessing = config["evaluation"].get("postprocessing")
    label_string = config["evaluation"].get("labels")
    if postprocessing is None:
        raise ValueError("postprocessing must be specified in evaluation config for all datasets")
    
    # Select the appropriate postprocessing function based on config
    if postprocessing == "load_normalized_data":
        # Simple normalization: lowercase + strip
        postprocess_func = lambda text, labels: text.lower().strip() if text else ""
    elif postprocessing == "process_mlc_custom":
        # Multi-label classification with custom labels
        from src.evaluation.data_processing import process_mlc_custom_single
        postprocess_func = lambda text, labels: process_mlc_custom_single(text, labels)
    elif postprocessing == "process_mcq_custom":
        # Multiple choice questions with custom option types
        from src.evaluation.data_processing import process_mcq_custom_single
        postprocess_func = lambda text, labels: process_mcq_custom_single(text, labels)
    elif postprocessing == "process_ner_custom" or postprocessing == "process_ner_token_indices":
        # Named entity recognition with custom entity types (TOKEN INDICES)
        from src.evaluation.data_processing import process_ner_token_indices_single
        postprocess_func = lambda text, labels: process_ner_token_indices_single(text, labels)[1]  # Return entities only
    elif postprocessing == "process_ner_char_offsets":
        # Named entity recognition with custom entity types (CHARACTER OFFSETS)
        from src.evaluation.data_processing import process_ner_char_offsets_single
        # For char offsets, we need the input text, so we handle this differently in the loop below
        postprocess_func = None
    else:
        raise ValueError(f"Unknown postprocessing function: {postprocessing}")
    
    # Add postprocessed outputs and references using the selected function
    if postprocessing == "process_ner_char_offsets":
        # Special handling for character offsets (needs input)
        from src.evaluation.data_processing import process_ner_char_offsets_single
        for rec in records:
            input = rec.get("input", "")
            rec["postprocessed_output"] = process_ner_char_offsets_single(rec["output"], label_string, input)[1]
            # For reference in char offsets, it's already in the correct format
            rec["postprocessed_reference"] = rec["reference"]
    elif postprocessing in ["process_ner_custom", "process_ner_token_indices"]:
        # Special handling for token indices (reference is already in correct format)
        from src.evaluation.data_processing import process_ner_token_indices_single
        for rec in records:
            rec["postprocessed_output"] = process_ner_token_indices_single(rec["output"], label_string)[1]
            # For reference in token indices, it's already in the correct format
            rec["postprocessed_reference"] = rec["reference"]
    else:
        for rec in records:
            rec["postprocessed_output"] = postprocess_func(rec["output"], label_string)
            rec["postprocessed_reference"] = postprocess_func(rec["reference"], label_string)
    
    return records


def run_evaluation(config: Dict[str, Any], records: List[Dict[str, Any]], inference_output_file: str) -> Dict[str, Any]:
    """Execute the evaluation stage and return results."""
    task = config["task"]
    dataset = config["dataset"]
    metrics = config["evaluation"].get("metrics", [])
    
    # Get task handler for metadata processing
    task_handler = get_task_handler(task)
    
    # Get processing configuration - now required for all datasets
    postprocessing = config["evaluation"].get("postprocessing")
    label_string = config["evaluation"].get("labels")
    if postprocessing is None:
        raise ValueError("postprocessing must be specified in evaluation config for all datasets")
    
    # Add example-level metadata using task handler
    for rec in records:
        rec = task_handler.add_example_metadata(rec)
    
    # Save temporary evaluation file for processing
    base_name = os.path.splitext(inference_output_file)[0]
    temp_eval_file = f"{base_name}_temp_eval.json"
    with open(temp_eval_file, 'w') as f:
        json.dump(records, f, indent=2)
    
    # Compute all metrics and store results
    metrics_results = {}
    for metric in metrics:
        eval_args = [
                            "python3", "src/cli/evaluate_cli.py",
            "--task", task,
            "--dataset", dataset,
            "--metric", metric,
            "--input", temp_eval_file,
            "--postprocessing", postprocessing,
        ]
        if label_string:
            eval_args += ["--label_string", label_string]
        
        # Capture metric result
        env = os.environ.copy()
        env['PYTHONPATH'] = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        result = subprocess.run(eval_args, check=True, capture_output=True, text=True, env=env)
        metric_value = float(result.stdout.strip().split(": ")[-1])
        metrics_results[metric] = metric_value
        print(f"{metric} score: {metric_value}")
    
    # Compute individual metrics for each record (where applicable)
    # Create a mapping for metrics that support individual computation
    individual_metrics_map = {
        'rouge1': 'compute_rouge1_individual',
        'rouge2': 'compute_rouge2_individual',
        'rougeL': 'compute_rougeL_individual',
        'bertscore': 'compute_bertscore_individual',
        'meteor': 'compute_meteor_individual',
        'bartscore': 'compute_bartscore_individual',
        'accuracy': 'compute_accuracy_individual',
    }
    
    # Compute individual metrics for metrics that support it
    for metric in metrics:
        if metric in individual_metrics_map:
            # Extract predictions and references from records
            predictions = [rec["postprocessed_output"] for rec in records]
            references = [rec["postprocessed_reference"] for rec in records]
            
            # Get the individual metric function
            individual_metric_func = getattr(metrics_module, individual_metrics_map[metric])
            
            # Compute individual scores
            individual_scores = individual_metric_func(predictions, references)
            
            # Add scores to each record (rounded to 4 decimal places)
            for i, rec in enumerate(records):
                if "individual_metrics" not in rec:
                    rec["individual_metrics"] = {}
                rec["individual_metrics"][metric] = round(individual_scores[i], 4)
    
    # Structure the evaluation output
    with open(inference_output_file, 'r') as f:
        data = json.load(f)
    
    dataset_info_from_inference = data.get("dataset_info")
    inference_metadata = data.get("inference_metadata")
    
    eval_data = {
        "dataset_info": dataset_info_from_inference or {
            "task": task,
            "dataset": dataset,
        },
        "inference_metadata": inference_metadata,
        "evaluation_metadata": {
            "evaluation_timestamp": datetime.datetime.now().isoformat(),
            "metrics": metrics_results,
            "postprocessing": postprocessing,
            "labels": label_string,
        },
        "records": records
    }
    
    # Clean up temporary file
    if os.path.exists(temp_eval_file):
        os.remove(temp_eval_file)
    
    return eval_data 