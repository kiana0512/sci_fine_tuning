# preprocessing/convert_definition.py

import json
from typing import List, Dict
from datasets import Dataset, DatasetDict


def load_definition_jsonl(path: str) -> List[Dict]:
    """
    加载 definition-based 数据集，每行为一个 JSON 对象。
    """
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def convert_definition_format(data: List[Dict]) -> List[Dict]:
    """
    将 definition 数据转换为指令微调格式。
    每条数据使用标签定义作为背景知识。
    """
    formatted = []
    for item in data:
        sentence = item["sentence"]
        label = item["label"]
        definition = item["label_definition"]

        formatted.append({
            "instruction": "请根据标签定义为以下句子分类。",
            "input": f"句子: {sentence}\n定义: {definition}",
            "output": label
        })

    return formatted


def preprocess_definition_dataset(path: str) -> DatasetDict:
    """
    读取 definition 数据集，转换为 HuggingFace DatasetDict，仅返回训练集。
    """
    raw_data = load_definition_jsonl(path)
    processed = convert_definition_format(raw_data)
    dataset = Dataset.from_list(processed)

    return DatasetDict({"train": dataset})
