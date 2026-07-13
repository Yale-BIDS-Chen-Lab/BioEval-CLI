#!/usr/bin/env python3
"""Anthropic (Claude) inference (REST).

Ported to match the BioEval web app (inference-service/src/inference/anthropic.py).
Reads prompts from a JSON/JSONL file, generates a completion per prompt via the
Anthropic Messages endpoint, and writes the outputs back.

Allowed hyperparameters (mirroring the web app): temperature, max_tokens
(default 4096; required by the Anthropic API).
"""
import argparse
import datetime
import json
import os
import sys
from urllib import error, request

import tqdm

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"


def _read_prompts(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()], None, None
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data, None, None
        return data["records"], data.get("prompt_generation_metadata"), data.get("dataset_info")
    raise ValueError("Input must be .json or .jsonl")


def _generate(api_key, model, text, temperature, max_tokens):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": text}],
        "max_tokens": int(max_tokens) if max_tokens is not None else 4096,
    }
    if temperature is not None:
        payload["temperature"] = float(temperature)
    req = request.Request(
        ANTHROPIC_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "content-type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_VERSION,
        },
        method="POST",
    )
    with request.urlopen(req, timeout=300) as response:
        payload = json.loads(response.read().decode("utf-8"))
    blocks = payload.get("content", [])
    return "".join(
        block.get("text", "")
        for block in blocks
        if isinstance(block, dict) and block.get("type") == "text"
    )


def main():
    p = argparse.ArgumentParser(description="Anthropic (Claude) inference")
    p.add_argument("--model", required=True, help="Anthropic model id (e.g. claude-sonnet-4-5)")
    p.add_argument("--api_key", default=None, help="API key; falls back to ANTHROPIC_API_KEY")
    p.add_argument("--input_file", required=True)
    p.add_argument("--output_file", required=True)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--max_tokens", type=int, default=None)
    args = p.parse_args()

    api_key = args.api_key or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("Anthropic API key required (--api_key or ANTHROPIC_API_KEY)")

    prompts, prompt_metadata, dataset_info = _read_prompts(args.input_file)
    print(f"Loaded {len(prompts)} prompts from {args.input_file}")

    outputs, failed = [], 0
    for i, rec in enumerate(tqdm.tqdm(prompts, desc="Claude", unit="prompt")):
        text = rec.get("prompt") or rec.get("input")
        if text is None:
            print(f"Warning: record {i} missing 'prompt'/'input'", file=sys.stderr)
            continue
        try:
            rec["output"] = _generate(api_key, args.model, text, args.temperature, args.max_tokens)
        except Exception as e:
            rec["output"] = ""
            rec["error"] = str(e)
            failed += 1
            print(f"Failed prompt {i}: {e}", file=sys.stderr)
        outputs.append(rec)

    inference_metadata = {
        "inference_timestamp": datetime.datetime.now().isoformat(),
        "model_name": args.model,
        "provider": "anthropic",
        "failed_requests": failed,
        "inference_parameters": {
            k: getattr(args, k) for k in ("temperature", "max_tokens")
            if getattr(args, k) is not None
        },
        "prompt_generation_metadata": prompt_metadata,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(
            {"dataset_info": dataset_info, "inference_metadata": inference_metadata, "records": outputs},
            f, indent=2, ensure_ascii=False,
        )
    print(f"Completed: {len(outputs)} total, {failed} failed → {args.output_file}")


if __name__ == "__main__":
    main()
