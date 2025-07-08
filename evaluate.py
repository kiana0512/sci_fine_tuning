import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq
from peft import PeftModel
from metrics import compute_metrics  # ✅ 你已有的指标函数


def format_input(example):
    return {"prompt": f"{example['instruction']}\n{example['input']}", "output": example["output"]}


def main():
    model_dir = "checkpoints/context_lora"  # ✅ 你训练保存的模型目录
    dataset_path = "./processed/context"  # ✅ 原始数据集路径
    max_length = 256

    # 加载 tokenizer 和数据
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    raw_dataset = load_from_disk(dataset_path)
    raw_dataset = raw_dataset.map(format_input)
    raw_dataset = raw_dataset.remove_columns(["instruction", "input"])

    # ✅ 用 train 集合划分测试集
    test_dataset = raw_dataset["train"].train_test_split(test_size=0.1, seed=42)["test"]

    def tokenize(example):
        return tokenizer(
            example["prompt"],
            text_target=example["output"],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )

    tokenized = test_dataset.map(tokenize, batched=True)

    # ✅ 加载 LoRA 模型（你训练过的）
    base_model = AutoModelForCausalLM.from_pretrained(
        "gemma_local",
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        local_files_only=True
    )
    model = PeftModel.from_pretrained(base_model, model_dir)

    model.eval()

    # ✅ 数据准备（小 batch，避免爆内存）
    dataloader = torch.utils.data.DataLoader(
        tokenized.remove_columns(["prompt", "output"]),
        batch_size=2,
        collate_fn=DataCollatorForSeq2Seq(tokenizer, model=model),
    )

    preds = []
    labels = []

    # ✅ 推理
    for batch in dataloader:
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        labels_batch = batch["labels"].to(model.device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_batch)
            logits = outputs.logits.argmax(dim=-1)

        preds.extend(logits.cpu().tolist())
        labels.extend(labels_batch.cpu().tolist())

    # ✅ 评估
    results = compute_metrics({"predictions": preds, "labels": labels})
    print("✅ 验证集评估结果：")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
