#!/usr/bin/env python3
"""
Compare Base vs Fine-Tuned TinyLlama-1.1B

Runs 5 insurance QA examples on both models and compares outputs.
Shows:
- Base model output
- Fine-tuned model output
- JSON validity
- Confidence levels

This demonstrates the improvement from LoRA fine-tuning.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Config
BASE_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_PATH = "tinyllama_lora_adapter"

SYSTEM_MSG = (
    "You are an expert insurance assistant. "
    "Always respond with valid JSON containing exactly: "
    "answer (string), confidence (high/medium/low), source (policy/general_knowledge)."
)

# 5 test questions covering diverse insurance topics
TEST_QUESTIONS = [
    "What is the collision deductible on an auto comprehensive policy?",
    "Does my homeowner's insurance cover flood damage?",
    "What is the difference between term life and whole life insurance?",
    "How do I file an insurance claim after a car accident?",
    "What does an insurance deductible mean?",
]


def build_prompt(question: str) -> str:
    """Build a chat-style prompt consistent with training format."""
    return (
        f"[SYS] {SYSTEM_MSG} [/SYS]\n"
        f"[INST] {question} [/INST]\n"
    )


def load_base_model() -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load the base TinyLlama model without any adapter."""
    logger.info("Loading base model: %s", BASE_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    model.eval()
    logger.info("Base model loaded ✓")
    return model, tokenizer


def load_finetuned_model(base_model, adapter_path: str) -> PeftModel:
    """Load the LoRA adapter on top of the base model."""
    if not Path(adapter_path).exists():
        raise FileNotFoundError(
            f"LoRA adapter not found at: {adapter_path}\n"
            "Run: python train_tiny.py"
        )
    logger.info("Loading LoRA adapter: %s", adapter_path)
    ft_model = PeftModel.from_pretrained(base_model, adapter_path)
    ft_model.eval()
    logger.info("Fine-tuned model ready ✓")
    return ft_model


def generate_response(model, tokenizer, question: str, max_new_tokens: int = 150) -> str:
    """Generate a response from either model."""
    prompt = build_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only newly generated tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def extract_json_safely(text: str) -> Optional[Dict]:
    """Try to extract and parse the first valid JSON object from text."""
    # First try: direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Second try: find first { ... } block
    start = text.find("{")
    if start == -1:
        return None
    end = text.rfind("}") + 1
    if end <= start:
        return None
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        return None


def run_comparison():
    """Compare base vs fine-tuned model on 5 questions."""
    print("\n" + "=" * 80)
    print("  BASE vs FINE-TUNED TinyLlama-1.1B COMPARISON")
    print("  LoRA adapter:", ADAPTER_PATH)
    print("=" * 80)

    # Load models — share tokenizer
    base_model, tokenizer = load_base_model()

    try:
        ft_model = load_finetuned_model(base_model, ADAPTER_PATH)
    except FileNotFoundError as e:
        logger.error(str(e))
        return

    results = []

    for idx, question in enumerate(TEST_QUESTIONS, 1):
        print(f"\n{'─' * 80}")
        print(f"  EXAMPLE {idx}: {question}")
        print("─" * 80)

        # Base model
        logger.info("Generating BASE response...")
        base_raw = generate_response(base_model, tokenizer, question)
        base_json = extract_json_safely(base_raw)

        # Fine-tuned model
        logger.info("Generating FINE-TUNED response...")
        ft_raw = generate_response(ft_model, tokenizer, question)
        ft_json = extract_json_safely(ft_raw)

        # Display
        print(f"\n📋 Question: {question}\n")

        print("🔹 BASE MODEL OUTPUT:")
        if base_json:
            print(json.dumps(base_json, indent=2))
            print("  ✅ Valid JSON")
        else:
            snippet = base_raw[:200].replace("\n", " ")
            print(f"  ❌ Not valid JSON: {snippet}...")

        print("\n🔹 FINE-TUNED MODEL OUTPUT:")
        if ft_json:
            print(json.dumps(ft_json, indent=2))
            print("  ✅ Valid JSON")
        else:
            snippet = ft_raw[:200].replace("\n", " ")
            print(f"  ⚠️  Not valid JSON: {snippet}...")

        print("\n📊 ANALYSIS:")
        if ft_json and not base_json:
            print("  ✅ Fine-tuned produces valid JSON, base does NOT — clear improvement")
        elif ft_json and base_json:
            print("  ✅ Both produce valid JSON")
            if ft_json.get("confidence"):
                print(f"     Fine-tuned confidence: {ft_json['confidence']}")
            if ft_json.get("source"):
                print(f"     Fine-tuned source: {ft_json['source']}")
        elif not ft_json and not base_json:
            print("  ⚠️  Neither produced valid JSON — more fine-tuning may help")

        results.append({
            "question": question,
            "base_output": base_raw,
            "base_json": base_json or {},
            "base_valid_json": base_json is not None,
            "ft_output": ft_raw,
            "ft_json": ft_json or {},
            "ft_valid_json": ft_json is not None,
        })

    # Summary
    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)

    base_valid = sum(1 for r in results if r["base_valid_json"])
    ft_valid = sum(1 for r in results if r["ft_valid_json"])

    print(f"\n  📊 JSON Validity out of {len(results)} examples:")
    print(f"     Base model  : {base_valid}/{len(results)} valid JSON outputs")
    print(f"     Fine-tuned  : {ft_valid}/{len(results)} valid JSON outputs")

    if ft_valid > base_valid:
        print(f"\n  ✅ IMPROVEMENT: Fine-tuned model produces "
              f"{ft_valid - base_valid} more valid structured outputs")
    elif ft_valid == base_valid:
        print("\n  ℹ️  Same count — check qualitative differences above")

    # Save results
    out_path = Path("inference_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n  📁 Full results saved to: {out_path}\n")

    return results


if __name__ == "__main__":
    run_comparison()
