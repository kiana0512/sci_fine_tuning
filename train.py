import os
import math
import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    TrainerCallback,
)
from peft import get_peft_model, LoraConfig, TaskType
from torch.utils.tensorboard import SummaryWriter


def format_input(example):
    prompt = f"{example['instruction']}\n{example['input']}"
    return {"prompt": prompt, "output": example["output"]}


# ✅ 自定义 TensorBoard 日志记录器
class MetricLoggerCallback(TrainerCallback):
    def __init__(self, log_dir):
        self.records = []
        self.writer = SummaryWriter(log_dir=log_dir, flush_secs=1)

    def on_log(self, args, state, control, logs=None, **kwargs):
        step = state.global_step
        if logs is not None and "loss" in logs:
            loss = logs.get("loss")
            lr = logs.get("learning_rate", None)

            self.records.append({"step": step, "loss": loss, "learning_rate": lr})
            self.writer.add_scalar("Loss/train", loss, step)
            if lr is not None:
                self.writer.add_scalar("LearningRate", lr, step)
            print(f"[TensorBoard] step={step} loss={loss:.4f} lr={lr:.6e}")

    def on_step_end(self, args, state, control, model=None, **kwargs):
        step = state.global_step
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = math.sqrt(total_norm)
        self.writer.add_scalar("GradientNorm", total_norm, step)
        if self.records:
            self.records[-1]["grad_norm"] = total_norm

    def on_train_end(self, args, state, control, **kwargs):
        self.writer.export_scalars_to_json(os.path.join(args.output_dir, "all_scalars.json"))
        self.writer.close()


def main():
    # ===== 路径设置 =====
    model_path = "./gemma_local"
    dataset_path = "./processed/example"
    output_dir = "checkpoints/example_lora"
    log_dir = os.path.join(output_dir, "tblog")
    os.makedirs(log_dir, exist_ok=True)

    # ===== 加载数据集 =====
    dataset = load_from_disk(dataset_path)
    dataset = dataset.map(format_input)
    dataset = dataset.remove_columns(["instruction", "input"])

    # ===== 加载分词器 =====
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

    # ===== 加载模型并应用 LoRA =====
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

    # ===== 训练参数设置 =====
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-5,
        bf16=torch.cuda.is_available(),

        logging_dir=log_dir,
        logging_strategy="steps",
        logging_steps=10,

        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,

        report_to=[],  # ❌ 禁用 transformers 默认日志
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        callbacks=[MetricLoggerCallback(log_dir)],
    )

    # ===== 开始训练 =====
    trainer.train()
    print("✅ 训练完成，TensorBoard 日志保存在:", log_dir)


if __name__ == "__main__":
    main()
