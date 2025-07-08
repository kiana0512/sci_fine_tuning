# sci_fine_tuning/prompt_template.py

def build_prompt(example):
    """
    从 instruction 和 input 构造 prompt。
    当前 example 来自 LoRA 格式化后的数据集，字段为：
    - instruction: 提示词，例如“请根据上下文为当前句子分类。”
    - input: 拼接后的上下文
    """
    return f"{example['instruction']}\n{example['input']}"
