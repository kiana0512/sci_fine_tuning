# ✅ 修复后的 train.py：兼容 Gemma 和 transformers 最新版（开发版）

import os
import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    DefaultFlowCallback,
)
from peft import get_peft_model, LoraConfig, TaskType
from metrics import compute_metrics
from dotenv import load_dotenv
load_dotenv()

def format_input(example):
    prompt = f"{example['instruction']}\n{example['input']}"
    return {"prompt": prompt, "output": example["output"]}


def main():
    # 从环境变量中安全地获取 Hugging Face Token
    HF_TOKEN = os.getenv("HF_TOKEN")
    if HF_TOKEN is None:
        raise EnvironmentError("❌ 环境变量 HF_TOKEN 未设置，请通过 export HF_TOKEN=your_token 设置。")

    model_name_or_path = "google/gemma-3n-E4B-it"
    dataset_path = "processed/context"
    output_dir = "checkpoints/context_lora"

    dataset = load_from_disk(dataset_path)
    dataset = dataset.map(format_input)
    dataset = dataset.remove_columns(["instruction", "input"])

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        token=HF_TOKEN,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    def tokenize_function(examples):
        return tokenizer(
            examples["prompt"],
            text_target=examples["output"],
            padding="max_length",
            truncation=True,
            max_length=512
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        token=HF_TOKEN,
        trust_remote_code=True
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)

    # ✅ 不再使用 evaluation_strategy 参数，手动使用 callback 控制评估频率
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-5,
        bf16=torch.cuda.is_available(),
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=10,
        save_total_limit=2,
        report_to=[],
        # ❌ 这些都不要加
        # evaluation_strategy="steps",
        # save_strategy="steps",
        # load_best_model_at_end=True,
        # eval_steps=50,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        compute_metrics=compute_metrics,
        callbacks=[DefaultFlowCallback, EarlyStoppingCallback(early_stopping_patience=2)]
    )

    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("✅ 训练完成，模型已保存到:", output_dir)


if __name__ == "__main__":
    main()
