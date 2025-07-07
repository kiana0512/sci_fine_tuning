import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import argparse


def load_model(model_path: str, use_lora: bool = False):
    if use_lora:
        # 加载 LoRA 配置
        config = PeftConfig.from_pretrained(model_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        # 直接加载原始模型
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    return tokenizer, model


def generate_response(tokenizer, model, prompt: str, max_new_tokens: int = 100):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to your fine-tuned model or LoRA adapter")
    parser.add_argument("--use_lora", action="store_true", help="Use this flag if model_path is a LoRA adapter")
    parser.add_argument("--prompt", type=str, help="Prompt to test", default="What is the capital of France?")
    args = parser.parse_args()

    tokenizer, model = load_model(args.model_path, use_lora=args.use_lora)
    response = generate_response(tokenizer, model, args.prompt)

    print("\n=== Prompt ===")
    print(args.prompt)
    print("\n=== Model Output ===")
    print(response)
