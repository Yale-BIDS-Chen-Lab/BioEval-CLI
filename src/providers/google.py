#!/usr/bin/env python3
"""Google Gemini inference (REST).

Ported to match the BioEval web app (inference-service/src/inference/google.py).
Reads prompts from a JSON/JSONL file, generates a completion per prompt via the
Gemini generateContent endpoint, and writes the outputs back.

Allowed hyperparameters (mirroring the web app): temperature, top_p, max_tokens
(mapped to maxOutputTokens; default 4096).
"""
import argparse
import datetime
import json
import os
import sys
from urllib import error, parse, request

import tqdm

GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"


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


def _generation_config(temperature, top_p, max_tokens):
    cfg = {"maxOutputTokens": int(max_tokens) if max_tokens is not None else 4096}
    if temperature is not None:
        cfg["temperature"] = float(temperature)
    if top_p is not None:
        cfg["topP"] = float(top_p)
    return cfg


def _generate(api_key, model, text, gen_cfg):
    url = f"{GEMINI_API_BASE_URL}/{model}:generateContent?key={parse.quote(api_key, safe='')}"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": text}]}],
        "generationConfig": gen_cfg,
    }
    req = request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=300) as response:
        payload = json.loads(response.read().decode("utf-8"))
    candidates = payload.get("candidates", [])
    if not candidates:
        block_reason = payload.get("promptFeedback", {}).get("blockReason")
        raise RuntimeError(
            "Gemini returned no candidates" + (f" (blockReason={block_reason})" if block_reason else "")
        )
    parts = candidates[0].get("content", {}).get("parts", [])
    return "".join(
        part.get("text", "")
        for part in parts
        if isinstance(part, dict) and isinstance(part.get("text"), str)
    )


def main():
    p = argparse.ArgumentParser(description="Google Gemini inference")
    p.add_argument("--model", required=True, help="Gemini model id (e.g. gemini-2.5-pro)")
    p.add_argument("--api_key", default=None, help="API key; falls back to GOOGLE_API_KEY / GEMINI_API_KEY")
    p.add_argument("--input_file", required=True)
    p.add_argument("--output_file", required=True)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top_p", type=float, default=None)
    p.add_argument("--max_tokens", type=int, default=None)
    args = p.parse_args()

    api_key = args.api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        sys.exit("Gemini API key required (--api_key or GOOGLE_API_KEY / GEMINI_API_KEY)")

    prompts, prompt_metadata, dataset_info = _read_prompts(args.input_file)
    print(f"Loaded {len(prompts)} prompts from {args.input_file}")
    gen_cfg = _generation_config(args.temperature, args.top_p, args.max_tokens)

    outputs, failed = [], 0
    for i, rec in enumerate(tqdm.tqdm(prompts, desc="Gemini", unit="prompt")):
        text = rec.get("prompt") or rec.get("input")
        if text is None:
            print(f"Warning: record {i} missing 'prompt'/'input'", file=sys.stderr)
            continue
        try:
            rec["output"] = _generate(api_key, args.model, text, gen_cfg)
        except Exception as e:
            rec["output"] = ""
            rec["error"] = str(e)
            failed += 1
            print(f"Failed prompt {i}: {e}", file=sys.stderr)
        outputs.append(rec)

    inference_metadata = {
        "inference_timestamp": datetime.datetime.now().isoformat(),
        "model_name": args.model,
        "provider": "google",
        "failed_requests": failed,
        "inference_parameters": {
            k: getattr(args, k) for k in ("temperature", "top_p", "max_tokens")
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
