#!/usr/bin/env python3
"""
Part 4: Fine-Tune TinyLlama-1.1B for Insurance QA (CPU-only, LoRA/PEFT)

Model     : TinyLlama/TinyLlama-1.1B-Chat-v1.0  (~2.2 GB download)
Technique : LoRA via HuggingFace PEFT — NOT full fine-tuning
Output    : adapter weights saved separately in  tinyllama_lora_adapter/

LoRA  : rank=8, alpha=16, target=[q_proj, v_proj], dropout=0.05
Train : epochs=3, batch=1, grad_accum=4, lr=2e-4
"""

import json
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from peft import LoraConfig, TaskType, get_peft_model

# ---------------------------------------------------------------------------
# Logging – writes to stdout AND training_log.txt
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("training_log.txt", mode="w"),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_NAME    = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATASET_PATH  = "insurance_qa_dataset.json"
OUTPUT_DIR    = "tinyllama_lora_adapter"
MAX_LENGTH    = 256
LORA_RANK     = 8
LORA_ALPHA    = 16
LORA_DROPOUT  = 0.05
LEARNING_RATE = 2e-4
NUM_EPOCHS    = 3
BATCH_SIZE    = 1
GRAD_ACCUM    = 4

SYSTEM_MSG = (
    "You are an expert insurance assistant. "
    "Always respond with valid JSON containing exactly: "
    "answer (string), confidence (high/medium/low), source (policy/general_knowledge)."
)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class InsuranceQADataset(Dataset):
    """
    Builds a chat-style prompt per sample:

        [SYS] {system message} [/SYS]
        [INST] {question} [/INST]
        {json answer}

    Labels == input_ids so the model is trained with causal-LM loss over
    the entire sequence (standard instruction-tuning approach on CPU).
    """

    def __init__(self, data_path: str, tokenizer, max_length: int = MAX_LENGTH):
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer  = tokenizer
        self.max_length = max_length
        logger.info("Dataset loaded: %d samples from %s", len(self.data), data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item        = self.data[idx]
        question    = item.get("input", "")
        output_text = json.dumps(item.get("output", {}))

        # Build the prompt using plain markers (works with any tokenizer)
        prompt = (
            f"[SYS] {SYSTEM_MSG} [/SYS]\n"
            f"[INST] {question} [/INST]\n"
            f"{output_text}"
        )

        enc = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids      = enc["input_ids"].squeeze()
        attention_mask = enc["attention_mask"].squeeze()

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         input_ids.clone(),   # standard CLM objective
        }


# ---------------------------------------------------------------------------
# Loss-logging callback
# ---------------------------------------------------------------------------
class LossLogger(TrainerCallback):
    """Prints loss + visual progress bar every logging step."""

    def __init__(self):
        self.loss_history = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        loss = logs.get("loss")
        if loss is None:
            return

        self.loss_history.append(loss)
        step       = state.global_step
        total      = state.max_steps or 1
        pct        = step / total * 100
        bar_done   = int(30 * step / total)
        bar        = "#" * bar_done + "-" * (30 - bar_done)
        trend      = ""
        if len(self.loss_history) >= 2:
            trend = " ▼" if self.loss_history[-1] < self.loss_history[-2] else " ▲"

        logger.info(
            "Step %3d/%d  |%s| %5.1f%%  loss=%.4f%s",
            step, total, bar, pct, loss, trend,
        )

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(state.epoch)
        recent = self.loss_history[-5:] if self.loss_history else []
        if recent:
            logger.info(
                "=== Epoch %d complete.  Recent losses: %s ===",
                epoch, [f"{l:.4f}" for l in recent],
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_model_and_tokenizer():
    logger.info("=" * 60)
    logger.info("Loading tokenizer: %s", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info("Tokenizer ready | vocab=%d", len(tokenizer))

    logger.info("Loading base model (fp32, CPU) ...")
    logger.info("  >> ~2.2 GB download on first run — please wait <<")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    logger.info("Base model loaded | %.2fB parameters", total_params)
    return model, tokenizer


def apply_lora(model):
    logger.info("=" * 60)
    logger.info("Applying LoRA adapters (rank=%d, alpha=%d)", LORA_RANK, LORA_ALPHA)

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()   # shows how few params we actually train
    return model


def train(model, tokenizer):
    logger.info("=" * 60)
    logger.info("Preparing dataset …")

    dataset = InsuranceQADataset(DATASET_PATH, tokenizer)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_steps=10,
        logging_steps=5,
        save_steps=9999,          # save only at end
        save_total_limit=1,
        report_to="none",
        use_cpu=True,
        seed=42,
        disable_tqdm=False,
        logging_first_step=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[LossLogger()],
    )

    logger.info("=" * 60)
    logger.info("STARTING TRAINING")
    logger.info("  model      : %s", MODEL_NAME)
    logger.info("  samples    : %d", len(dataset))
    logger.info("  epochs     : %d", NUM_EPOCHS)
    logger.info("  batch      : %d  (grad_accum=%d)", BATCH_SIZE, GRAD_ACCUM)
    logger.info("  lr         : %s", LEARNING_RATE)
    logger.info("  output_dir : %s", OUTPUT_DIR)
    logger.info("=" * 60)

    trainer.train()

    # Save adapter weights SEPARATELY from the base model
    logger.info("Saving LoRA adapter to: %s/", OUTPUT_DIR)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    logger.info("Adapter saved.  Base model weights are NOT included — only the LoRA delta.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    print("\n" + "=" * 60)
    print("  Part 4: TinyLlama-1.1B Fine-Tuning with LoRA (CPU)")
    print("=" * 60 + "\n")

    if not Path(DATASET_PATH).exists():
        logger.error("Dataset not found: %s", DATASET_PATH)
        logger.error("Run: python generate_dataset.py")
        sys.exit(1)

    model, tokenizer = load_model_and_tokenizer()
    model = apply_lora(model)
    train(model, tokenizer)

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print(f"  Adapter saved to: {OUTPUT_DIR}/")
    print("  Next step: python inference_tiny.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
