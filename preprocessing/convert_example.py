# preprocessing/convert_example.py

import json
from typing import List, Dict
from datasets import Dataset, DatasetDict


def load_example_jsonl(path: str) -> List[Dict]:
    """
    加载 example-based 数据集，每行包含 examples、target_sentence、target_label。
    """
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def convert_example_format(data: List[Dict]) -> List[Dict]:
    """
    将带 few-shot 示例的 JSON 转换为指令微调格式。
    """
    formatted = []
    for item in data:
        few_shots = item["examples"]
        target = item["target_sentence"]
        label = item["target_label"]

        # 构造 few-shot 示例文本
        example_strs = [f"{ex['sentence']} → {ex['label']}" for ex in few_shots]
        example_block = "\n".join(example_strs)

        formatted.append({
            "instruction": "请为目标句子分类，参考以下示例。",
            "input": f"{example_block}\n目标句子: {target}",
            "output": label
        })

    return formatted


def preprocess_example_dataset(path: str) -> DatasetDict:
    """
    读取 example 数据集并转换为 HuggingFace DatasetDict，仅返回训练集。
    """
    raw_data = load_example_jsonl(path)
    processed = convert_example_format(raw_data)
    dataset = Dataset.from_list(processed)

    return DatasetDict({"train": dataset})
