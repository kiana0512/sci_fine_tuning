# sci_fine_tuning/evaluate_generation.py

import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk
from tqdm import tqdm
from prompt_template import build_prompt
from metrics import compute_metrics
from collections import Counter

# 定义合法标签集合（根据你的任务）
VALID_LABELS = {"MTD", "BAC", "PUR", "GAP", "RST", "CLN", "CTN", "IMP"}


def generate_prediction(model, tokenizer, prompt, max_new_tokens=3):
    """
    Generate prediction from model given prompt.
    Only returns label if in VALID_LABELS, else returns "UNKNOWN".
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id
        )
    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    generated = decoded[len(prompt):].strip()
    pred = generated.split()[0].strip(":：,，.。\n ")
    if pred not in VALID_LABELS:
        return "UNKNOWN"
    return pred


def main():
    base_model_path = "./gemma_local"
    lora_path = "checkpoints/example_lora/checkpoint-570"
    data_path = "./processed/example_split"

    # Load validation dataset
    dataset = load_from_disk(data_path)["validation"]

    # Load tokenizer and LoRA-adapted model
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(base_model_path, device_map="auto", torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(model, lora_path)
    model.eval()

    pred_labels = []
    true_labels = []

    print("🔍 Generating predictions...")
    for example in tqdm(dataset):
        prompt = build_prompt(example)
        pred = generate_prediction(model, tokenizer, prompt)
        pred_labels.append(pred)
        true_labels.append(example["output"])

    # Display top 10 predictions
    print("\n👀 Top 10 Predictions vs Ground Truth:")
    for i in range(min(10, len(pred_labels))):
        print(f"{i + 1}. Pred: {pred_labels[i]} | Label: {true_labels[i]}")

    print("\n📊 Prediction label distribution:", Counter(pred_labels))
    print("📊 True label distribution:", Counter(true_labels))

    # Optionally remove UNKNOWN before computing metrics
    filtered_preds = []
    filtered_labels = []
    for pred, label in zip(pred_labels, true_labels):
        if pred != "UNKNOWN":
            filtered_preds.append(pred)
            filtered_labels.append(label)

    print("\n✅ Evaluation Metrics (excluding UNKNOWN):")
    results = compute_metrics((filtered_preds, filtered_labels))
    for k, v in results.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
