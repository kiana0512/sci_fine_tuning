# 📘 Scientific Concept Understanding via Fine-tuning LLM

## 🧠 Project Overview

This project fine-tunes the `google/gemma-2b-it` or `gemma-2b` model for **scientific concept understanding**, targeting multi-format supervised tasks like:

- 🧾 **Context-based understanding**
- 📖 **Definition-based classification**
- 🧪 **Example-based prediction**

All training is LoRA-compatible and Hugging Face Transformers-based. The model can be deployed on local CPU/GPU or remote cloud servers for inference and evaluation.

## 📊 Supported Tasks

| Task Type         | Input Format                              | Output (Label/Text) |
|-------------------|-------------------------------------------|----------------------|
| Context-based     | Scientific context paragraph               | Classification label |
| Definition-based  | Technical definition of a scientific term | Classification label |
| Example-based     | Example description sentence              | Classification label |

## 🗂️ Dataset Format (JSON)

Each data sample should be formatted as follows:

```json
{
  "input": "This is a scientific context or definition...",
  "label": "correct/incorrect"
}
```

## 💻 Environment Setup

```bash
git clone https://github.com/yourname/sci-finetune.git
cd sci-finetune
conda create -n sci python=3.10 -y
conda activate sci
pip install -r requirements.txt
```

### 🔧 requirements.txt

```txt
torch>=2.1.0
transformers>=4.40.0
datasets
scikit-learn
peft
accelerate
huggingface_hub
sentencepiece
```

## 🚀 Training

```bash
python train.py \
    --model_name_or_path google/gemma-2b-it \
    --train_file data/context_train.json \
    --validation_file data/context_val.json \
    --output_dir checkpoints/context_lora \
    --use_lora
```

## 🧪 Evaluation Report Generation

```bash
python generate_eval_report.py \
    --model_path checkpoints/context_lora \
    --data_file data/context_val.json
```

## 🤖 Inference Script

```bash
python predict.py \
    --model_path checkpoints/context_lora \
    --use_lora \
    --prompt "Define quantum entanglement."
```

## 🧩 File Structure

```
sci_fine_tuning/
├── train.py
├── predict.py
├── generate_eval_report.py
├── metrics.py
├── requirements.txt
├── data/
│   ├── context_train.json
│   └── context_val.json
├── checkpoints/
│   └── context_lora/
```

## 🔒 Hugging Face Auth

```bash
huggingface-cli login
```

## 🧯 Troubleshooting

| Error | Solution |
|-------|----------|
| `No module named sklearn` | `pip install scikit-learn` |
| `Token not accepted` | Use `huggingface-cli login` or set `HF_TOKEN` manually |
| `Unexpected argument: evaluation_strategy` | Downgrade Transformers or use correct class (`TrainingArguments`) |
| `OSError: not a valid model identifier` | Use local path instead of HF repo URL |
| `--load_best_model_at_end` error | Ensure `evaluation_strategy == save_strategy` |

## 📍 Next Steps

- [ ] Add multi-GPU support
- [ ] ONNX / TensorRT conversion
- [ ] Web UI deployment
- [ ] Batch API

## 👩‍💻 Maintainer

> Author: [@kiana0512](https://github.com/kiana0512)  
> Email: contact@yourdomain.com