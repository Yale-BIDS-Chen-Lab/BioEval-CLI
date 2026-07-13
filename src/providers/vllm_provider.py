#!/usr/bin/env python
"""
vLLM-based text generation with full SamplingParams support (v0.8.4)
Docs: https://docs.vllm.ai/en/v0.8.4/api/inference_params.html
"""
import argparse
import json
import torch
from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutputKind


def str2bool(v):
    if isinstance(v, bool):
        return v
    return v.lower() in ("yes", "true", "t", "y", "1")


def parse_list(v):
    return json.loads(v) if v is not None else None


def main():
    parser = argparse.ArgumentParser(description="vLLM-based text generation with all inference parameters")
    # Required args
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--input_file", required=True, help="JSON list of prompts")
    parser.add_argument("--output_file", required=True, help="Where to write outputs")

    # SamplingParams args
    parser.add_argument("--n", type=int, default=1, help="Number of output sequences per prompt")
    parser.add_argument("--best_of", type=int, default=None, help="Number of sequences to generate and then choose best n")
    parser.add_argument("--presence_penalty", type=float, default=0.0)
    parser.add_argument("--frequency_penalty", type=float, default=0.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--stop", type=parse_list, default=None,
                        help="JSON list or string of stop sequences")
    parser.add_argument("--stop_token_ids", type=parse_list, default=None,
                        help="JSON list of stop token IDs")
    parser.add_argument("--ignore_eos", type=str2bool, default=False)
    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument("--min_tokens", type=int, default=0)
    parser.add_argument("--logprobs", type=int, default=None)
    parser.add_argument("--prompt_logprobs", type=int, default=None)
    parser.add_argument("--detokenize", type=str2bool, default=True)
    parser.add_argument("--skip_special_tokens", type=str2bool, default=True)
    parser.add_argument("--spaces_between_special_tokens", type=str2bool, default=True)
    parser.add_argument("--include_stop_str_in_output", type=str2bool, default=False)
    parser.add_argument("--truncate_prompt_tokens", type=int, default=None)
    parser.add_argument("--output_kind", type=str, default="CUMULATIVE",
                        choices=[k for k in RequestOutputKind.__members__.keys()])
    parser.add_argument("--output_text_buffer_length", type=int, default=0)
    parser.add_argument("--logit_bias", type=json.loads, default=None,
                        help="JSON dict of token_id: bias")
    parser.add_argument("--allowed_token_ids", type=parse_list, default=None,
                        help="JSON list of token ID lists")
    parser.add_argument("--bad_words", type=parse_list, default=None,
                        help="JSON list of forbidden strings")
    parser.add_argument("--extra_args", type=json.loads, default=None,
                        help="JSON dict of extra sampling arguments")
    # Guided decoding and logits_processors require custom handling; parsed as JSON
    parser.add_argument("--guided_decoding", type=json.loads, default=None,
                        help="JSON for GuidedDecodingParams")
    parser.add_argument("--logits_processors", type=json.loads, default=None,
                        help="JSON list specifying custom logits processors")

    args = parser.parse_args()

    # Diagnostics
    print("CUDA available:", torch.cuda.is_available())
    print("GPU count   :", torch.cuda.device_count())

    # Initialize vLLM engine
    llm = LLM(model=args.model)

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
        
    prompt_texts = [item["prompt"] for item in prompts]

    # Build SamplingParams
    sampling_params = SamplingParams(
        n=args.n,
        best_of=args.best_of,
        presence_penalty=args.presence_penalty,
        frequency_penalty=args.frequency_penalty,
        repetition_penalty=args.repetition_penalty,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        seed=args.seed,
        stop=args.stop,
        stop_token_ids=args.stop_token_ids,
        ignore_eos=args.ignore_eos,
        max_tokens=args.max_tokens,
        min_tokens=args.min_tokens,
        logprobs=args.logprobs,
        prompt_logprobs=args.prompt_logprobs,
        detokenize=args.detokenize,
        skip_special_tokens=args.skip_special_tokens,
        spaces_between_special_tokens=args.spaces_between_special_tokens,
        include_stop_str_in_output=args.include_stop_str_in_output,
        truncate_prompt_tokens=args.truncate_prompt_tokens,
        output_kind=RequestOutputKind[args.output_kind],
        output_text_buffer_length=args.output_text_buffer_length,
        logit_bias=args.logit_bias,
        allowed_token_ids=args.allowed_token_ids,
        bad_words=args.bad_words,
        extra_args=args.extra_args,
        guided_decoding=args.guided_decoding,
        logits_processors=args.logits_processors,
    )

    # Run generation
    outputs = llm.generate(prompt_texts, sampling_params=sampling_params)

    # Collect results
    results = []
    for output in outputs:
        idx = int(output.request_id)
        item = prompts[idx]
        generated_text = output.outputs[0].text
        results.append({
            "id":        item.get("id"),
            "input": item.get("input"),
            "reference": item.get("reference"),
            "prompt":    output.prompt,
            "output":    generated_text
        })

    # Create structured output with metadata
    import datetime
    
    inference_metadata = {
        "inference_timestamp": datetime.datetime.now().isoformat(),
        "model_name": args.model,
        "provider": "vllm",
        "inference_parameters": {
            k: getattr(args, k) for k in [
                "temperature", "top_p", "top_k", "min_p", "presence_penalty", 
                "frequency_penalty", "repetition_penalty", "max_tokens", "min_tokens",
                "n", "best_of", "seed"
            ] if getattr(args, k) is not None
        },
        "prompt_generation_metadata": prompt_metadata
    }
    
    # Write outputs
    output_data = {
        "dataset_info": dataset_info,
        "inference_metadata": inference_metadata,
        "records": results
    }
    
    with open(args.output_file, 'w') as f:
        json.dump(output_data, f, indent=2)


if __name__ == "__main__":
    main()