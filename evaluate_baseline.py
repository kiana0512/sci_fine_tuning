import csv
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk
from tqdm import tqdm
from prompt_template import build_prompt
from metrics import compute_metrics
from collections import Counter
import matplotlib.pyplot as plt

VALID_LABELS = {"MTD", "BAC", "PUR", "GAP", "RST", "CLN", "CTN", "IMP"}

def generate_prediction(model, tokenizer, prompt, max_new_tokens=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    model = model.to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True
        )
    output_ids = outputs.sequences[0]
    decoded = tokenizer.decode(output_ids, skip_special_tokens=True)
    raw_output = decoded[len(prompt):].strip()
    pred_label = raw_output.split()[0].strip(":：,，.。\n ")

    scores = outputs.scores[0]
    probs = F.softmax(scores, dim=-1)[0]
    pred_token_id = output_ids[len(inputs['input_ids'][0])]
    confidence = probs[pred_token_id].item()

    if pred_label not in VALID_LABELS:
        pred_label = "UNKNOWN"
        confidence = 0.0

    return raw_output, pred_label, confidence

def main():
    base_model_path = "./gemma_local"
    data_path = "./processed/context_split"
    dataset = load_from_disk(data_path)["validation"]

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(base_model_path, device_map="auto", torch_dtype=torch.float16)
    model.eval()

    pred_labels = []
    true_labels = []
    confidences = []
    raw_outputs = []

    master_csv_path = "baseline_prediction_analysis.csv"
    qual_csv_path = "baseline_raw_outputs.csv"

    with open(master_csv_path, "w", newline="") as f_master, open(qual_csv_path, "w", newline="") as f_qual:
        master_writer = csv.DictWriter(f_master, fieldnames=[
            "Input_Sentence", "True_Label", "Predicted_Label_Baseline", "Confidence_Baseline"
        ])
        qual_writer = csv.DictWriter(f_qual, fieldnames=[
            "Test_Sample_ID", "Input_Sentence", "True_Label", "Raw_Output_Baseline"
        ])
        master_writer.writeheader()
        qual_writer.writeheader()

        print("🔍 Running baseline predictions...")
        for idx, example in enumerate(tqdm(dataset)):
            try:
                prompt = build_prompt(example)
                raw_output, pred, confidence = generate_prediction(model, tokenizer, prompt)
            except Exception as e:
                print(f"[⚠️ Error @ idx {idx}] {e}")
                raw_output, pred, confidence = "", "UNKNOWN", 0.0

            pred_labels.append(pred)
            true_labels.append(example["output"])
            confidences.append(confidence)
            raw_outputs.append(raw_output)

            master_writer.writerow({
                "Input_Sentence": example["input"],
                "True_Label": example["output"],
                "Predicted_Label_Baseline": pred,
                "Confidence_Baseline": confidence
            })

            qual_writer.writerow({
                "Test_Sample_ID": idx,
                "Input_Sentence": example["input"],
                "True_Label": example["output"],
                "Raw_Output_Baseline": raw_output
            })

    print("\n👀 Top 10 Predictions vs Ground Truth:")
    for i in range(min(10, len(pred_labels))):
        print(f"{i + 1}. Pred: {pred_labels[i]} | Label: {true_labels[i]}")

    print("\n📊 Prediction label distribution:", Counter(pred_labels))
    print("📊 True label distribution:", Counter(true_labels))

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
    plt.bar(metrics_to_plot.keys(), metrics_to_plot.values(), color='orange')
    plt.title("Baseline Evaluation Metrics (Excl. UNKNOWN)")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("baseline_metrics.png")
    print("\n📈 Baseline图像已保存为 baseline_metrics.png")

if __name__ == "__main__":
    main()
