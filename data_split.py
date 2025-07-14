from datasets import load_from_disk, DatasetDict, Dataset
from sklearn.model_selection import train_test_split

def split_existing_hf_dataset(
    data_path="processed/context",       # ✅ 数据集路径
    test_size=0.1,                       # ✅ 验证集占比
    seed=42,                             # ✅ 随机种子
    save_path=None                       # ✅ 是否保存切分后的数据
) -> DatasetDict:
    print(f"📂 加载数据集：{data_path}")
    dataset = load_from_disk(data_path)

    # 如果已经是 DatasetDict 且包含 validation，不重复切分
    if isinstance(dataset, DatasetDict):
        if "validation" in dataset:
            print("✅ 已含 validation，跳过切分。")
            return dataset
        else:
            dataset = dataset["train"]

    print(f"✂️ 正在按比例 {test_size} 切分验证集...")

    # ❗ 修正关键点：将 dataset 转为 list 再切分
    dataset_list = dataset.to_list()
    train_data, val_data = train_test_split(dataset_list, test_size=test_size, random_state=seed)

    result = DatasetDict({
        "train": Dataset.from_list(train_data),
        "validation": Dataset.from_list(val_data)
    })

    if save_path:
        print(f"💾 保存到：{save_path}")
        result.save_to_disk(save_path)

    return result

if __name__ == "__main__":
    split_existing_hf_dataset(
        data_path="processed/example",
        test_size=0.1,
        seed=42,
        save_path="processed/example_split"
    )
