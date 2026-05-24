import os
from modelscope import snapshot_download

# 1. 明确指定你希望模型下载到本地的哪个路径
# 这里我们把它定在项目下的 models 文件夹中
local_dir_path = "./models/Qwen2.5-1.5B-Instruct"

print("🚀 开始下载 Qwen2.5-1.5B-Instruct 模型...")
print("📦 模型文件较大（约3GB），请保持网络畅通，耐心等待...\n")

# 2. 呼叫魔搭的下载函数
model_dir = snapshot_download(
    'Qwen/Qwen2.5-1.5B-Instruct', # 魔搭网上的模型ID
    local_dir=local_dir_path      # 强行指定下载到你刚设定的本地目录
)

print(f"\n🎉 下载成功！模型已完整保存在: {model_dir}")