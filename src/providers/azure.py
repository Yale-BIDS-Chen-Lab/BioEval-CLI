#!/usr/bin/env python3
"""
Azure OpenAI Chat Completions Inference Script

Based on Azure OpenAI REST API reference:
https://learn.microsoft.com/azure/ai-services/openai/reference

Supports comprehensive parameter set for chat completions endpoint with
API version 2024-10-21 and later versions.
"""
import argparse
import json
import os
import sys
import time
from typing import List, Dict, Any, Optional, Union

import tqdm
from openai import AzureOpenAI


def _read_prompts(path: str) -> tuple[List[Dict], Optional[Dict], Optional[Dict]]:
    """Read prompts from JSON or JSONL file, return (prompts, metadata, dataset_info)"""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()], None, None
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Handle both old format (list) and new format (dict with metadata/records)
        if isinstance(data, list):
            return data, None, None
        else:
            return data["records"], data.get("prompt_generation_metadata"), data.get("dataset_info")
    raise ValueError("Input must be .json or .jsonl")


def _write_output(data: List[Dict], path: str) -> None:
    """Write output to JSON or JSONL file"""
    dirname = os.path.dirname(path)
    if dirname:  # Only create directory if path contains one
        os.makedirs(dirname, exist_ok=True)
    ext = os.path.splitext(path)[1].lower()
    mode = "w"
    if ext == ".jsonl":
        with open(path, mode, encoding="utf-8") as f:
            for rec in data:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    elif ext == ".json":
        with open(path, mode, encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    else:
        raise ValueError("Output must be .json or .jsonl")


def _parse_json(v: str) -> Optional[Any]:
    """Parse JSON string or return None if empty"""
    return json.loads(v) if v else None


def _str_to_bool(v: str) -> bool:
    """Convert string to boolean"""
    if isinstance(v, bool):
        return v
    if v.lower() in {'yes', 'true', 't', 'y', '1'}:
        return True
    elif v.lower() in {'no', 'false', 'f', 'n', '0'}:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def _parse_tools(v: str) -> Optional[List[Dict]]:
    """Parse tools JSON string for function calling"""
    if not v:
        return None
    try:
        tools = json.loads(v)
        if not isinstance(tools, list):
            raise ValueError("Tools must be a list")
        return tools
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"Invalid JSON for tools: {e}")


def _parse_response_format(v: str) -> Optional[Dict]:
    """Parse response format specification"""
    if not v:
        return None
    try:
        if v in ['text', 'json_object']:
            return {"type": v}
        else:
            # Try to parse as JSON for more complex response format
            return json.loads(v)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"Invalid response format: {e}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Azure OpenAI Chat Completions with comprehensive parameter support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Basic usage:
    python infer_azure.py --endpoint https://myresource.openai.azure.com/ \\
                         --api_key $AZURE_OPENAI_API_KEY \\
                         --api_version 2024-10-21 \\
                         --model gpt-4o \\
                         --input_file prompts.json \\
                         --output_file responses.json

  With advanced parameters:
    python infer_azure.py --endpoint https://myresource.openai.azure.com/ \\
                         --api_key $AZURE_OPENAI_API_KEY \\
                         --api_version 2024-10-21 \\
                         --model gpt-4o \\
                         --input_file prompts.json \\
                         --output_file responses.json \\
                         --temperature 0.7 \\
                         --max_tokens 2048 \\
                         --top_p 0.95 \\
                         --frequency_penalty 0.5 \\
                         --presence_penalty 0.5 \\
                         --stop '["\\n\\n", "END"]' \\
                         --response_format json_object
        """
    )
    
    # Required parameters
    required = p.add_argument_group('Required Arguments')
    required.add_argument("--endpoint", required=True, 
                         help="Azure OpenAI resource endpoint (e.g., https://myresource.openai.azure.com/)")
    required.add_argument("--api_key", required=True, 
                         help="Azure OpenAI API key (can be env var name with $ prefix)")
    required.add_argument("--api_version", required=True, 
                         help="Azure OpenAI API version (e.g., 2024-10-21)")
    required.add_argument("--model", required=True, 
                         help="Deployment name of the model")
    required.add_argument("--input_file", required=True, 
                         help="Input JSON/JSONL file with prompts")
    required.add_argument("--output_file", required=True, 
                         help="Output JSON/JSONL file for responses")

    # Core generation parameters
    generation = p.add_argument_group('Generation Parameters')
    generation.add_argument("--temperature", type=float, default=1.0,
                           help="Sampling temperature (0.0-2.0). Higher = more random")
    generation.add_argument("--top_p", type=float, default=1.0,
                           help="Nucleus sampling probability (0.0-1.0)")
    generation.add_argument("--max_tokens", type=int, default=None,
                           help="Maximum tokens to generate (legacy parameter)")
    generation.add_argument("--max_completion_tokens", type=int, default=None,
                           help="Maximum completion tokens to generate")
    generation.add_argument("--min_tokens", type=int, default=None,
                           help="Minimum tokens to generate")
    generation.add_argument("--n", type=int, default=1,
                           help="Number of completions to generate")

    # Penalty parameters
    penalties = p.add_argument_group('Penalty Parameters')
    penalties.add_argument("--frequency_penalty", type=float, default=0.0,
                          help="Frequency penalty (-2.0 to 2.0)")
    penalties.add_argument("--presence_penalty", type=float, default=0.0,
                          help="Presence penalty (-2.0 to 2.0)")
    penalties.add_argument("--repetition_penalty", type=float, default=None,
                          help="Repetition penalty (experimental)")

    # Stop parameters
    stop_group = p.add_argument_group('Stop Parameters')
    stop_group.add_argument("--stop", type=_parse_json, default=None,
                           help="Stop sequences as JSON list (e.g., '[\"\\n\", \"END\"]')")

    # Advanced parameters
    advanced = p.add_argument_group('Advanced Parameters')
    advanced.add_argument("--response_format", type=_parse_response_format, default=None,
                         help="Response format: 'text', 'json_object', or JSON schema")
    advanced.add_argument("--tools", type=_parse_tools, default=None,
                         help="Function calling tools as JSON list")
    advanced.add_argument("--tool_choice", type=str, default=None,
                         help="Tool choice: 'none', 'auto', 'required', or specific tool")
    advanced.add_argument("--parallel_tool_calls", type=_str_to_bool, default=None,
                         help="Enable parallel tool calls (true/false)")

    # Logit manipulation
    logits = p.add_argument_group('Logit Parameters')
    logits.add_argument("--logit_bias", type=_parse_json, default=None,
                       help="Logit bias as JSON dict (e.g., '{\"50256\": -100}')")
    logits.add_argument("--logprobs", type=_str_to_bool, default=None,
                       help="Return log probabilities (true/false)")
    logits.add_argument("--top_logprobs", type=int, default=None,
                       help="Number of top logprobs to return (0-20)")

    # Streaming and output
    output_group = p.add_argument_group('Output Parameters')
    output_group.add_argument("--stream", type=_str_to_bool, default=False,
                             help="Stream responses (true/false)")
    output_group.add_argument("--stream_options", type=_parse_json, default=None,
                             help="Stream options as JSON")

    # Safety and content filtering
    safety = p.add_argument_group('Safety Parameters')
    safety.add_argument("--user", type=str, default=None,
                       help="Unique user identifier for monitoring")

    # Seed and determinism
    determinism = p.add_argument_group('Determinism Parameters')
    determinism.add_argument("--seed", type=int, default=None,
                            help="Random seed for deterministic generation")

    # System and operational
    system = p.add_argument_group('System Parameters')
    system.add_argument("--timeout", type=float, default=60.0,
                       help="Request timeout in seconds")
    system.add_argument("--retry_pause", type=float, default=5.0,
                       help="Pause between retries in seconds")
    system.add_argument("--max_retries", type=int, default=3,
                       help="Maximum number of retries")

    # Experimental parameters
    experimental = p.add_argument_group('Experimental Parameters')
    experimental.add_argument("--reasoning_effort", type=str, default=None,
                             help="Reasoning effort level (experimental)")
    experimental.add_argument("--include_usage", type=_str_to_bool, default=True,
                             help="Include usage statistics in response")

    args = p.parse_args()

    # Validate and resolve API key
    key = args.api_key.strip()
    if key.startswith("$"):
        key = os.getenv(key[1:])
        if not key:
            sys.exit(f"Environment variable {args.api_key[1:]} not found")
    elif key.isidentifier():
        key = os.getenv(key, key)
    args.api_key = key

    # Validate endpoint
    if not args.endpoint.startswith('https://'):
        sys.exit("Endpoint must be a valid HTTPS URL")
    
    if not args.endpoint.endswith('/'):
        args.endpoint += '/'

    # Initialize Azure OpenAI client
    client = AzureOpenAI(
        api_version=args.api_version,
        azure_endpoint=args.endpoint.rstrip("/"),
        api_key=args.api_key,
        timeout=args.timeout
    )

    # Load prompts
    try:
        prompts, prompt_metadata, dataset_info = _read_prompts(args.input_file)
        print(f"Loaded {len(prompts)} prompts from {args.input_file}")
    except Exception as e:
        sys.exit(f"Error loading prompts: {e}")

    outputs = []
    failed_requests = 0

    for i, rec in enumerate(tqdm.tqdm(prompts, desc="Processing prompts", unit="prompt")):
        text = rec.get("prompt") or rec.get("input")
        if text is None:
            print(f"Warning: Record {i} missing 'prompt' or 'input' field", file=sys.stderr)
            continue

        # Build the request parameters
        params = {
            "model": args.model,
            "messages": [{"role": "user", "content": text}],
        }

        # Add core parameters
        if args.temperature is not None:
            params["temperature"] = args.temperature
        if args.top_p is not None:
            params["top_p"] = args.top_p
        if args.max_tokens is not None:
            params["max_tokens"] = args.max_tokens
        if args.max_completion_tokens is not None:
            params["max_completion_tokens"] = args.max_completion_tokens
        if args.min_tokens is not None:
            params["min_tokens"] = args.min_tokens
        if args.n is not None and args.n != 1:
            params["n"] = args.n

        # Add penalty parameters
        if args.frequency_penalty != 0.0:
            params["frequency_penalty"] = args.frequency_penalty
        if args.presence_penalty != 0.0:
            params["presence_penalty"] = args.presence_penalty
        if args.repetition_penalty is not None:
            params["repetition_penalty"] = args.repetition_penalty

        # Add stop sequences
        if args.stop is not None:
            params["stop"] = args.stop

        # Add advanced parameters
        if args.response_format is not None:
            params["response_format"] = args.response_format
        if args.tools is not None:
            params["tools"] = args.tools
        if args.tool_choice is not None:
            params["tool_choice"] = args.tool_choice
        if args.parallel_tool_calls is not None:
            params["parallel_tool_calls"] = args.parallel_tool_calls

        # Add logit parameters
        if args.logit_bias is not None:
            params["logit_bias"] = args.logit_bias
        if args.logprobs is not None:
            params["logprobs"] = args.logprobs
        if args.top_logprobs is not None:
            params["top_logprobs"] = args.top_logprobs

        # Add streaming parameters
        if args.stream:
            params["stream"] = args.stream
        if args.stream_options is not None:
            params["stream_options"] = args.stream_options

        # Add determinism parameters
        if args.seed is not None:
            params["seed"] = args.seed

        # Add user tracking
        if args.user is not None:
            params["user"] = args.user

        # Add experimental parameters
        if args.reasoning_effort is not None:
            params["reasoning_effort"] = args.reasoning_effort

        # Make the API call with retries
        retries = 0
        while retries <= args.max_retries:
            try:
                response = client.chat.completions.create(**params)
                
                # Extract response content
                if hasattr(response, 'choices') and len(response.choices) > 0:
                    rec["output"] = response.choices[0].message.content
                    
                    # Add usage information if available and requested
                    if args.include_usage and hasattr(response, 'usage'):
                        rec["usage"] = {
                            "completion_tokens": response.usage.completion_tokens,
                            "prompt_tokens": response.usage.prompt_tokens,
                            "total_tokens": response.usage.total_tokens
                        }
                    
                    # Add model information
                    if hasattr(response, 'model'):
                        rec["model_used"] = response.model
                        
                    # Add any additional response metadata
                    if hasattr(response, 'id'):
                        rec["response_id"] = response.id
                        
                else:
                    rec["output"] = ""
                    print(f"Warning: Empty response for prompt {i}", file=sys.stderr)
                
                break  # Success, exit retry loop
                
            except Exception as e:
                retries += 1
                if retries > args.max_retries:
                    print(f"Failed to process prompt {i} after {args.max_retries} retries: {e}", file=sys.stderr)
                    rec["output"] = ""
                    rec["error"] = str(e)
                    failed_requests += 1
                    break
                else:
                    print(f"Retry {retries}/{args.max_retries} for prompt {i}: {e}", file=sys.stderr)
                    time.sleep(args.retry_pause)

        outputs.append(rec)

    # Create structured output with metadata
    import datetime
    
    inference_metadata = {
        "inference_timestamp": datetime.datetime.now().isoformat(),
        "model_name": args.model,
        "provider": "azure",
        "failed_requests": failed_requests,
        "inference_parameters": {
            k: getattr(args, k) for k in [
                "temperature", "top_p", "max_tokens", "max_completion_tokens",
                "presence_penalty", "frequency_penalty", "min_tokens", 
                "repetition_penalty", "seed", "n"
            ] if getattr(args, k) is not None
        },
        "prompt_generation_metadata": prompt_metadata
    }
    
    # Write results
    try:
        ext = os.path.splitext(args.output_file)[1].lower()
        if ext == ".jsonl":
            _write_output(outputs, args.output_file)
        else:
            output_data = {
                "dataset_info": dataset_info,
                "inference_metadata": inference_metadata,
                "records": outputs
            }
            with open(args.output_file, 'w', encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"Completed: {len(outputs)} total, {failed_requests} failed → {args.output_file}")
    except Exception as e:
        sys.exit(f"Error writing output: {e}")


if __name__ == "__main__":
    main()