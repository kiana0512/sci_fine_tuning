import os
import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import get_peft_model, LoraConfig, TaskType

# ✅ 格式化输入：拼接 prompt
def format_input(example):
    prompt = f"{example['instruction']}\n{example['input']}"
    return {"prompt": prompt, "output": example["output"]}


def main():
    model_path = "./gemma_local"
    dataset_path = "./processed/example"
    output_dir = "checkpoints/example_lora"

    # ✅ 加载和预处理数据集
    dataset = load_from_disk(dataset_path)
    dataset = dataset.map(format_input)
    dataset = dataset.remove_columns(["instruction", "input"])

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(
            examples["prompt"],
            text_target=examples["output"],
            padding="max_length",
            truncation=True,
            max_length=256
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # ✅ 加载模型并应用 LoRA
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
        local_files_only=True
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

    # ✅ 训练参数（无评估，稳定）
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-5,
        bf16=torch.cuda.is_available(),
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    )

    trainer.train()

    print("✅ 训练完成，模型已保存至:", output_dir)


if __name__ == "__main__":
    main()
