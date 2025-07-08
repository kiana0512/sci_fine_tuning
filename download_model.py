import os
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

# ✅ 加载 .env 文件中的环境变量
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    raise ValueError("❌ 未在 .env 中找到 HF_TOKEN，请检查是否配置正确。")

# 模型 ID 和保存路径
model_id = "google/gemma-3n-E4B-it"
save_dir = "gemma_local"

# ✅ 使用 token 认证方式下载（不会加载到内存）
snapshot_download(
    repo_id=model_id,
    local_dir=save_dir,
    token=hf_token,
    resume_download=True,
)

print(f"✅ 模型已成功下载到目录: {save_dir}")
