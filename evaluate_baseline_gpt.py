import openai
import csv
import time
from datasets import load_from_disk
from prompt_template import build_prompt
from metrics import compute_metrics
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm
import re

# 设置 API KEY
openai.api_key = "你的OPENAI_API_KEY"  # ✅ 替换为你的 Key

# GPT模型
MODEL = "gpt-4.1"  # 或 "gpt-4o"

VALID_LABELS = {"MTD", "BAC", "PUR", "GAP", "RST", "CLN", "CTN", "IMP"}

def extract_label(response_text):
    # 提取第一个合法标签作为预测
    match = re.search(r"\b(?:MTD|BAC|PUR|GAP|RST|CLN|CTN|IMP)\b", response_text)
    return match.group(0) if match else "UNKNOWN"

def gpt_infer(prompt):
    try:
        response = openai.ChatCompletion.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10
        )
        reply = response.choices[0].message.content.strip()
        return reply
    except Exception as e:
        print(f"[⚠️ GPT Error] {e}")
        return "ERROR"

def main():
    dataset = load_from_disk("./processed/context_split")["validation"]

    pred_labels = []
    true_labels = []
    raw_outputs = []

    master_csv_path = "gpt_prediction_analysis.csv"
    qual_csv_path = "gpt_raw_outputs.csv"

    with open(master_csv_path, "w", newline="") as f_master, open(qual_csv_path, "w", newline="") as f_qual:
        master_writer = csv.DictWriter(f_master, fieldnames=[
            "Input_Sentence", "True_Label", "Predicted_Label_GPT"
        ])
        qual_writer = csv.DictWriter(f_qual, fieldnames=[
            "Test_Sample_ID", "Input_Sentence", "True_Label", "Raw_Output_GPT"
        ])
        master_writer.writeheader()
        qual_writer.writeheader()

        print("🔍 Calling OpenAI GPT API for Zero-Shot predictions...")
        for idx, example in enumerate(tqdm(dataset)):
            prompt = build_prompt(example)
            response = gpt_infer(prompt)
            pred = extract_label(response)

            pred_labels.append(pred)
            true_labels.append(example["output"])
            raw_outputs.append(response)

            master_writer.writerow({
                "Input_Sentence": example["input"],
                "True_Label": example["output"],
                "Predicted_Label_GPT": pred
            })

            qual_writer.writerow({
                "Test_Sample_ID": idx,
                "Input_Sentence": example["input"],
                "True_Label": example["output"],
                "Raw_Output_GPT": response
            })

            time.sleep(0.5)  # 避免速率过快触发限制

    print("\n👀 Top 10 GPT Predictions vs Ground Truth:")
    for i in range(min(10, len(pred_labels))):
        print(f"{i + 1}. Pred: {pred_labels[i]} | Label: {true_labels[i]}")

    print("\n📊 Prediction label distribution:", Counter(pred_labels))
    print("📊 True label distribution:", Counter(true_labels))

    # 排除 UNKNOWN
    filtered_preds = []
    filtered_labels = []
    for p, l in zip(pred_labels, true_labels):
        if p != "UNKNOWN":
            filtered_preds.append(p)
            filtered_labels.append(l)

    print("\n✅ Evaluation Metrics (excluding UNKNOWN):")
    results = compute_metrics((filtered_preds, filtered_labels))
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    metrics_to_plot = {k: v for k, v in results.items() if isinstance(v, (int, float))}
    plt.figure(figsize=(6, 4))
    plt.bar(metrics_to_plot.keys(), metrics_to_plot.values(), color='green')
    plt.title("GPT Baseline Evaluation Metrics (Excl. UNKNOWN)")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("gpt_metrics.png")
    print("\n📈 GPT图像已保存为 gpt_metrics.png")

if __name__ == "__main__":
    main()
