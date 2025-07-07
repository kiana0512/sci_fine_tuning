# preprocessing/convert_context.py

import json
from typing import List, Dict
from datasets import Dataset, DatasetDict


def load_context_jsonl(path: str) -> List[Dict]:
    """
    从 JSONL 文件中读取 context-based 数据集。
    每行是一个 JSON 对象，包含 previous、current_sentence、current_label、next。
    """
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def convert_context_format(data: List[Dict]) -> List[Dict]:
    """
    将原始数据转换为指令微调格式，确保 previous 和 next 字段为 None 时可处理。
    """
    converted = []
    for item in data:
        previous_obj = item.get("previous")
        next_obj = item.get("next")

        previous = previous_obj["sentence"] if previous_obj else "无"
        next_ = next_obj["sentence"] if next_obj else "无"
        current = item["current_sentence"]
        label = item["current_label"]

        converted.append({
            "instruction": "请根据上下文为当前句子分类。",
            "input": f"上文: {previous}\n当前: {current}\n下文: {next_}",
            "output": label
        })

    return converted



def preprocess_context_dataset(path: str) -> DatasetDict:
    """
    主函数：读取并转换 context 数据集，只返回训练集部分。
    """
    raw_data = load_context_jsonl(path)
    processed_data = convert_context_format(raw_data)
    dataset = Dataset.from_list(processed_data)

    return DatasetDict({"train": dataset})
