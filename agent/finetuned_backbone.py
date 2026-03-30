#!/usr/bin/env python3
"""
Fine-Tuned TinyLlama Backbone for the Insurance Agent

This module provides a drop-in LLM callable that uses the locally
fine-tuned TinyLlama-1.1B + LoRA adapter instead of calling the
Ollama model server via HTTP.

Usage (set environment variable before starting agent/server.py):
    USE_FINETUNED_BACKBONE=true python agent/server.py

The model is loaded lazily on the first call to generate_response()
so the import itself is cheap.
"""

import logging
import os
import json
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Paths
_THIS_DIR = Path(__file__).parent
ADAPTER_PATH = str(_THIS_DIR.parent / "finetune" / "tinyllama_lora_adapter")
BASE_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Lazy-loaded singletons
_model = None
_tokenizer = None


def _load_model():
    """Load base model + LoRA adapter on first call (lazy init)."""
    global _model, _tokenizer

    if _model is not None:
        return _model, _tokenizer

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except ImportError as e:
        raise RuntimeError(
            f"Missing dependency for fine-tuned backbone: {e}\n"
            "Run: pip install torch transformers peft"
        )

    if not Path(ADAPTER_PATH).exists():
        raise FileNotFoundError(
            f"LoRA adapter not found at: {ADAPTER_PATH}\n"
            "Run: cd finetune && python train_tiny.py"
        )

    logger.info("Loading TinyLlama base model: %s", BASE_MODEL_NAME)
    _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, use_fast=True)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    logger.info("Loading base model on CPU (float32)...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="cpu",
    )

    logger.info("Loading LoRA adapter: %s", ADAPTER_PATH)
    _model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    _model.eval()

    total_params = sum(p.numel() for p in _model.parameters()) / 1e9
    logger.info(
        "Fine-tuned backbone ready — %.2fB params (base + LoRA adapter)", total_params
    )
    return _model, _tokenizer


def generate_response(
    history_prompt: str,
    system_prompt: Optional[str] = None,
    max_new_tokens: int = 256,
) -> str:
    """
    Generate a response using the fine-tuned TinyLlama + LoRA adapter.

    Args:
        history_prompt: Formatted conversation history (from format_history_for_prompt)
        system_prompt:  Optional system context prepended to the prompt
        max_new_tokens: Maximum tokens to generate (default 256)

    Returns:
        Generated response string (decoded, stripped)
    """
    import torch

    model, tokenizer = _load_model()

    # Build full prompt
    if system_prompt:
        full_prompt = f"[SYS] {system_prompt} [/SYS]\n{history_prompt}"
    else:
        full_prompt = history_prompt

    inputs = tokenizer(
        full_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=False,
    )

    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    new_token_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()


def is_available() -> bool:
    """Check if the fine-tuned adapter exists and dependencies are installed."""
    try:
        import torch          # noqa
        from peft import PeftModel  # noqa
        from transformers import AutoModelForCausalLM  # noqa
    except ImportError:
        return False
    return Path(ADAPTER_PATH).exists()
