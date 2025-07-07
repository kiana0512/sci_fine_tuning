# prepare_all_datasets.py

from preprocessing.convert_context import preprocess_context_dataset
from preprocessing.convert_definition import preprocess_definition_dataset
from preprocessing.convert_example import preprocess_example_dataset

def main():
    context_path = "../dataset/fine_tuning_data_process/context_based.jsonl"
    definition_path = "../dataset/fine_tuning_data_process/definition_based.jsonl"
    example_path = "../dataset/fine_tuning_data_process/example_based.jsonl"

    context_ds = preprocess_context_dataset(context_path)
    definition_ds = preprocess_definition_dataset(definition_path)
    example_ds = preprocess_example_dataset(example_path)

    # 保存为 Hugging Face 格式
    context_ds.save_to_disk("processed/context")
    definition_ds.save_to_disk("processed/definition")
    example_ds.save_to_disk("processed/example")

    print("✅ 所有数据集已处理并保存完成。")

if __name__ == "__main__":
    main()
