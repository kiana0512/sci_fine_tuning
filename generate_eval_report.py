# generate_eval_report.py

import os
import csv
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm


model_path = "checkpoints/context_lora"
dataset_path = "./processed/context"
output_file = "eval_report.csv"

# === 加载验证集（这里我们仍使用训练集模拟）===
dataset = load_from_disk(dataset_path)["train"]

# 构造 prompt 字段
def format_input(example):
    prompt = f"{example['instruction']}\n{example['input']}"
    return {"prompt": prompt, "label": example["output"]}

dataset = dataset.map(format_input)

# 加载 tokenizer 和 model
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)

model.eval()

# 生成预测
rows = []
for i, item in enumerate(tqdm(dataset, desc="Evaluating")):
    input_text = item["prompt"]
    label = item["label"]

    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10)
    pred = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    rows.append({
        "index": i,
        "input": input_text,
        "ground_truth": label,
        "prediction": pred,
        "is_correct": label == pred
    })

# 保存为 CSV
with open(output_file, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"✅ 预测结果保存到 {output_file}")
