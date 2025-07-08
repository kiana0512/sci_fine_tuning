# sci_fine_tuning/evaluate_generation.py

import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk
from tqdm import tqdm
from prompt_template import build_prompt
from metrics import compute_metrics  # 使用你原来的 metrics.py
from collections import Counter


def generate_prediction(model, tokenizer, prompt, max_new_tokens=8):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id
        )
    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    generated = decoded[len(prompt):].strip()
    prediction = generated.split()[0].strip("：:，,。 \n")
    return prediction


def main():
    base_model_path = "./gemma_local"       # ✅ 原始基座模型
    lora_path = "checkpoints/context_lora/checkpoint-570"       # ✅ LoRA 参数路径
    data_path = "./processed/context_split"        # ✅ 你的数据路径

    # 加载数据集（train_test_split 后）
    dataset = load_from_disk(data_path)["validation"]  # 使用 test 分支做验证
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(base_model_path, device_map="auto", torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(model, lora_path)  # ✅ 应用 LoRA 参数

    pred_labels = []
    true_labels = []

    print("🔍 正在生成预测...")
    for example in tqdm(dataset):
        prompt = build_prompt(example)
        pred = generate_prediction(model, tokenizer, prompt)
        pred_labels.append(pred)
        true_labels.append(example["output"])  # ✅ output 是真实标签

    # 输出前10对比
    print("\n👀 前10个预测 vs 实际标签：")
    for i in range(min(10, len(pred_labels))):
        print(f"{i + 1}. Pred: {pred_labels[i]} | Label: {true_labels[i]}")

    print("\n📊 预测标签分布：", Counter(pred_labels))
    print("📊 真实标签分布：", Counter(true_labels))

    print("\n✅ 验证集评估结果：")
    results = compute_metrics((pred_labels, true_labels))
    for k, v in results.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
