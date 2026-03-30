#!/usr/bin/env python3
"""
Part 4: Inference Comparison — Base Model vs Fine-Tuned TinyLlama

Runs 5 insurance questions through:
  1. Base TinyLlama-1.1B (no fine-tuning)
  2. Fine-Tuned TinyLlama-1.1B (LoRA adapter loaded from tinyllama_lora_adapter/)

Results are printed side-by-side and saved to inference_results.json
for inclusion in the README.

Usage:
    python inference_tiny.py
"""

import json
import sys
import time
import textwrap
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_DIR     = "tinyllama_lora_adapter"
OUTPUT_FILE     = "inference_results.json"
MAX_NEW_TOKENS  = 120

SYSTEM_MSG = (
    "You are an expert insurance assistant. "
    "Always respond with valid JSON containing exactly: "
    "answer (string), confidence (high/medium/low), source (policy/general_knowledge)."
)

# ---------------------------------------------------------------------------
# 5 evaluation questions (diverse, unseen at training time)
# ---------------------------------------------------------------------------
TEST_QUESTIONS = [
    "What is the collision deductible on an auto comprehensive policy?",
    "Does my homeowner's insurance cover flood damage?",
    "What is the difference between term life and whole life insurance?",
    "How do I file an insurance claim after a car accident?",
    "What does an insurance deductible mean?",
]


def build_prompt(question: str) -> str:
    """Build the same prompt format used during training."""
    return (
        f"[SYS] {SYSTEM_MSG} [/SYS]\n"
        f"[INST] Question: {question} [/INST]\n"
    )


def generate(model, tokenizer, prompt: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    """Run inference and return the generated text (only new tokens)."""
    inputs = tokenizer(prompt, return_tensors="pt")
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,        # greedy — deterministic for comparison
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][input_len:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def try_parse_json(text: str) -> dict:
    """Try to parse first JSON object found in text."""
    try:
        # find first { ... }
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(text[start:end])
    except Exception:
        pass
    return {}


def section(title: str):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


def compare(base_model, ft_model, tokenizer) -> list:
    results = []

    for i, question in enumerate(TEST_QUESTIONS, 1):
        section(f"Q{i}: {question}")
        prompt = build_prompt(question)

        # Base model
        print("\n[BASE MODEL]")
        t0     = time.time()
        base_out = generate(base_model, tokenizer, prompt)
        base_ms  = int((time.time() - t0) * 1000)
        print(textwrap.fill(base_out, width=65, initial_indent="  ", subsequent_indent="  "))
        base_json = try_parse_json(base_out)
        print(f"  >> Valid JSON fields: {list(base_json.keys()) or 'NONE'}  ({base_ms}ms)")

        # Fine-tuned model
        print("\n[FINE-TUNED MODEL]")
        t0    = time.time()
        ft_out = generate(ft_model, tokenizer, prompt)
        ft_ms  = int((time.time() - t0) * 1000)
        print(textwrap.fill(ft_out, width=65, initial_indent="  ", subsequent_indent="  "))
        ft_json = try_parse_json(ft_out)
        print(f"  >> Valid JSON fields: {list(ft_json.keys()) or 'NONE'}  ({ft_ms}ms)")

        results.append({
            "question":        question,
            "base_output":     base_out,
            "base_json":       base_json,
            "base_valid_json": bool(base_json),
            "ft_output":       ft_out,
            "ft_json":         ft_json,
            "ft_valid_json":   bool(ft_json),
        })

    return results


def main():
    section("Part 4 — Inference Comparison: Base vs Fine-Tuned TinyLlama")

    if not Path(ADAPTER_DIR).exists():
        print(f"\nERROR: Adapter not found at '{ADAPTER_DIR}/'")
        print("Run: python train_tiny.py   first\n")
        sys.exit(1)

    print(f"\nLoading tokenizer from base model: {BASE_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading BASE model (fp32, CPU) …")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    base_model.eval()
    print("Base model loaded.\n")

    print("Loading FINE-TUNED model (base + LoRA adapter) …")
    ft_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    ft_model = PeftModel.from_pretrained(ft_model, ADAPTER_DIR)
    ft_model.eval()
    print("Fine-tuned model loaded.\n")

    results = compare(base_model, ft_model, tokenizer)

    # Save results to JSON
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n\nResults saved to: {OUTPUT_FILE}")

    # Summary
    base_valid = sum(1 for r in results if r["base_valid_json"])
    ft_valid   = sum(1 for r in results if r["ft_valid_json"])
    section("SUMMARY")
    print(f"  Base model   valid JSON: {base_valid}/{len(results)}")
    print(f"  Fine-tuned   valid JSON: {ft_valid}/{len(results)}")
    print()


if __name__ == "__main__":
    main()
