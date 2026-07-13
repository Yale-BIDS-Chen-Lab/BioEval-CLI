#!/usr/bin/env python3
"""
Docs:
Pipelines parameters: https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.TextGenerationPipeline
GenerationConfig: https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
"""
import argparse
import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)

# Diagnostics
print("CUDA available:", torch.cuda.is_available())
print("GPU count   :", torch.cuda.device_count())


def str2bool(v):
    if isinstance(v, bool):
        return v
    return v.lower() in ("yes", "true", "t", "y", "1")


def parse_json(v):
    return json.loads(v) if v else None


def main():
    parser = argparse.ArgumentParser(description="HF-based text generation with all pipeline parameters")
    # Required arguments
    parser.add_argument("--model",               required=True, help="Model name or path")
    parser.add_argument("--input_file",          required=True, help="JSON list of prompts")
    parser.add_argument("--output_file",         required=True, help="Where to write outputs (.json) or .jsonl")

    # Sampling parameters
    parser.add_argument("--do_sample",           type=str2bool, default=None, help="Enable sampling")
    parser.add_argument("--temperature",         type=float,   default=None, help="Sampling temperature")
    parser.add_argument("--top_k",               type=int,     default=None, help="Top-K filtering")
    parser.add_argument("--top_p",               type=float,   default=None, help="Nucleus sampling")
    parser.add_argument("--typical_p",           type=float,   default=None, help="Typical P sampling")

    # Beam search parameters
    parser.add_argument("--num_beams",           type=int,     default=None, help="Number of beams for beam search")
    parser.add_argument("--num_beam_groups",     type=int,     default=None, help="Number of beam groups for diverse beam search")
    parser.add_argument("--penalty_alpha",       type=float,   default=None, help="Diverse beam search penalty alpha")
    parser.add_argument("--early_stopping",      type=str2bool, default=None, help="Early stopping for beam search")
    parser.add_argument("--length_penalty",      type=float,   default=None, help="Length penalty for beams")
    parser.add_argument("--num_return_sequences",type=int,     default=None, help="Number of sequences to return")

    # Generation length
    parser.add_argument("--max_length",          type=int,     default=None, help="Max total tokens (prompt + generated)")
    parser.add_argument("--min_length",          type=int,     default=None, help="Min total tokens")
    parser.add_argument("--max_new_tokens",      type=int,     default=None, help="Max tokens to generate")

    # Repetition and no-repeat
    parser.add_argument("--repetition_penalty",  type=float,   default=None, help="Repetition penalty")
    parser.add_argument("--no_repeat_ngram_size",type=int,     default=None, help="No repeat n-gram size")
    parser.add_argument("--encoder_no_repeat_ngram_size",type=int,default=None, help="No repeat n-gram on encoder-decoder")

    # Bad words and constraints
    parser.add_argument("--bad_words_ids",       type=parse_json, default=None, help="JSON list of token ID lists to block")
    parser.add_argument("--force_words_ids",     type=parse_json, default=None, help="JSON list of token ID lists to force include")
    parser.add_argument("--prefix_allowed_tokens_fn", type=str, default=None, help="Function path for prefix_allowed_tokens_fn")

    # Special tokens
    parser.add_argument("--eos_token_id",        type=int,     default=None, help="End-of-sequence token ID")
    parser.add_argument("--pad_token_id",        type=int,     default=None, help="Padding token ID")
    parser.add_argument("--bos_token_id",        type=int,     default=None, help="Bos token ID")
    parser.add_argument("--forced_bos_token_id", type=int,     default=None, help="Force bos token ID at generation start")
    parser.add_argument("--forced_eos_token_id", type=int,     default=None, help="Force eos token ID at generation end")

    # Logits processing
    parser.add_argument("--renormalize_logits",  type=str2bool, default=None, help="Renormalize logits during sampling")

    args = parser.parse_args()

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )

    # Prepare pipeline kwargs for special tokens
    pipeline_kwargs = {}
    for t in ["pad_token_id","eos_token_id","bos_token_id"]:
        v = getattr(args, t)
        if v is not None:
            pipeline_kwargs[t] = v

    text_gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
        **pipeline_kwargs
    )

    # Load prompts
    with open(args.input_file, 'r') as f:
        data = json.load(f)
    
    # Handle both old format (list) and new format (dict with metadata/records)
    if isinstance(data, list):
        prompts = data
        prompt_metadata = None
        dataset_info = None
    else:
        prompts = data["records"]
        prompt_metadata = data.get("prompt_generation_metadata")
        dataset_info = data.get("dataset_info")

    # Generate outputs
    outputs = []
    for item in prompts:
        prompt_text = item.get("prompt")
        if prompt_text is None:
            raise KeyError("Missing 'prompt'")

        gen_kwargs = {}
        if args.max_new_tokens        is not None: gen_kwargs["max_new_tokens"]      = args.max_new_tokens
        if args.temperature           is not None: gen_kwargs["temperature"]         = args.temperature
        if args.top_p                 is not None: gen_kwargs["top_p"]               = args.top_p
        if args.top_k                 is not None: gen_kwargs["top_k"]               = args.top_k
        if args.do_sample             is not None: gen_kwargs["do_sample"]           = args.do_sample
        if args.num_beams             is not None: gen_kwargs["num_beams"]           = args.num_beams
        if args.early_stopping        is not None: gen_kwargs["early_stopping"]      = args.early_stopping
        if args.length_penalty        is not None: gen_kwargs["length_penalty"]      = args.length_penalty
        if args.num_return_sequences  is not None: gen_kwargs["num_return_sequences"]= args.num_return_sequences
        if args.max_length            is not None: gen_kwargs["max_length"]          = args.max_length
        if args.min_length            is not None: gen_kwargs["min_length"]          = args.min_length
        if args.repetition_penalty    is not None: gen_kwargs["repetition_penalty"]  = args.repetition_penalty
        if args.no_repeat_ngram_size  is not None: gen_kwargs["no_repeat_ngram_size"]= args.no_repeat_ngram_size
        if args.bad_words_ids         is not None: gen_kwargs["bad_words_ids"]       = args.bad_words_ids
        if args.force_words_ids       is not None: gen_kwargs["force_words_ids"]     = args.force_words_ids
        if args.prefix_allowed_tokens_fn:
            module, func = args.prefix_allowed_tokens_fn.rsplit(':', 1)
            mod = __import__(module, fromlist=[func])
            gen_kwargs['prefix_allowed_tokens_fn'] = getattr(mod, func)
        if args.renormalize_logits     is not None: gen_kwargs["renormalize_logits"]  = args.renormalize_logits

        result = text_gen(prompt_text, **gen_kwargs)[0]["generated_text"]
        outputs.append({
            "id":        item.get("id"),
            "input": item.get("input"),
            "reference": item.get("reference"),
            "prompt":    prompt_text,
            "output":    result
        })

    # Create structured output with metadata
    import datetime
    
    inference_metadata = {
        "inference_timestamp": datetime.datetime.now().isoformat(),
        "model_name": args.model,
        "provider": "huggingface",
        "inference_parameters": {
            k: getattr(args, k) for k in [
                "temperature", "top_k", "top_p", "do_sample", "num_beams", 
                "early_stopping", "length_penalty", "max_new_tokens", "max_length",
                "min_length", "repetition_penalty", "no_repeat_ngram_size"
            ] if getattr(args, k) is not None
        },
        "prompt_generation_metadata": prompt_metadata
    }
    
    # Write outputs
    ext = args.output_file.lower().split('.')[-1]
    with open(args.output_file, 'w') as out_f:
        if ext == 'jsonl':
            for o in outputs:
                out_f.write(json.dumps(o) + "\n")
        else:
            output_data = {
                "dataset_info": dataset_info,
                "inference_metadata": inference_metadata,
                "records": outputs
            }
            json.dump(output_data, out_f, indent=2)

if __name__ == "__main__":
    main()