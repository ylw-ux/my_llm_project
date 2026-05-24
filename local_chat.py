import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==========================================
# 1. 指定本地模型的路径（刚才下载的战利品）
# ==========================================
MODEL_PATH = "./models/Qwen2.5-1.5B-Instruct"

print("⏳ 正在从本地硬盘加载分词器和模型权重...")
print("首次加载需要将 3GB 的矩阵数据读入内存/显存，请稍候...")

# ==========================================
# 2. 核心初始化：加载翻译官(Tokenizer)和大楼(Model)
# ==========================================
# 自动读取本地的 tokenizer_config.json
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH) 

# 自动读取本地的 config.json 并把 model.safetensors 的权重填进去
# device_map="auto" 会自动判断：如果你有 NVIDIA 显卡就用显卡加速，没有就用 CPU 运行
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype="auto"  # 自动选择最适合你硬件的数据精度（如 float16）
)

print("\n🎉 本地大模型加载成功！Legion 服务器已就绪！")
print("输入 'quit' 退出。")
print("-" * 50)


# ==========================================
# 3. 核心生成循环（自回归流式输出）
# ==========================================
while True:
    user_input = input("\n👤 我: ").strip()
    if not user_input or user_input.lower() == 'quit':
        break
        
    # 模拟标准的对话模板（给模型规范格式，打上 user 标签）
    messages = [
        {"role": "system", "content": "你是一个运行在本地的AI助手。"},
        {"role": "user", "content": user_input}
    ]
    
    # 这一步在底层把你的话变成了类似 <|im_start|>user\n你好<|im_end|> 的标准结构
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # 【工程闭环 1】: Encode —— 文本转成数字张量 (Tensor)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    print("🤖 思考中...\n🤖 本地大模型: ", end="")
    
    # 【工程闭环 2】: Generate —— 呼叫模型进行自回归生成
    # 这里我们不用等模型全部写完，而是用 Streamer 实现“吐出一个字，打印一个字”的流式效果
    from transformers import TextIteratorStreamer #流式输出包
    from threading import Thread #后台线程包
    
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    # 把复杂的生成过程丢进后台线程，防止主线程卡死
    generation_kwargs = dict(model_inputs, streamer=streamer, max_new_tokens=512)
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    # 【工程闭环 3】: Decode —— 实时从后台拿到生成的数字，翻成汉字蹦在屏幕上
    for new_text in streamer:
        print(new_text, end="", flush=True)
    print() # 换行

print(messages)
print("\n服务已关闭。")