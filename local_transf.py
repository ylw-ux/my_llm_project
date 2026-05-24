import os
import torch
# 导入 transformers 的核心组件：自动分词器和因果语言模型类
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==========================================
# 1. 明确指定你本地的模型路径（刚下载的战利品）
# ==========================================
MODEL_PATH = "./models/Qwen2.5-1.5B-Instruct"

print("⏳ 正在通过 transformers 加载本地模型和分词器...")

# ==========================================
# 2. 核心实例化：调用类方法加载模型对象
# ==========================================
# [学以致用]: 这里的 .from_pretrained 是面向对象里典型的“类方法”
# 它会去 MODEL_PATH 路径下自动寻找并解析 config.json、vocab.json 等文件

# 加载分词器对象（负责 Encode 和 Decode）
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# 加载大模型对象（负责矩阵计算生成）
# device_map="auto" 会让 transformers 自动侦测你的硬件：有 NVIDIA 显卡就上显卡，没有就用 CPU
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype="auto"  # 自动匹配最适合你显卡/CPU 的数据精度（如 float16）
)

print("\n🎉 本地大模型加载成功！你可以开始和它对话了。")
print("输入 'quit' 退出。")
print("-" * 50)

# ==========================================
# 3. 对话与推理循环
# ==========================================
while True:
    user_input = input("\n👤 我: ").strip()
    if not user_input or user_input.lower() == 'quit':
        break
        
    # 定义符合标准 Chat 格式的字典列表
    messages = [
        {"role": "system", "content": "你是一个运行在本地的AI助理。"},
        {"role": "user", "content": user_input}
    ]
    
    # 【步骤 A】: 套用模型的对话模板
    # 这步会把字典列表加上特殊符号，拼成一行类似 <|im_start|>user\n... 的连续字符串
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # 【步骤 B】: Encode（编码）
    # 把文本转成 PyTorch 的 Tensor（张量数字），并送入对应的计算设备（显卡或CPU）
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    print("🤖 思考中...")
    
    # 【步骤 C】: Generate（调用模型对象的方法生成预测数字）
    # 这里的 max_new_tokens 限制了自回归循环的最大次数
    # 传入model输出预测的token
    generated_ids = model.generate(
        **model_inputs,   #上下文传入
        max_new_tokens=512, #最大输出token
        min_new_tokens = 10,
        temperature=0.7, #温度 随机性，越大输出越多变， 低———理性 高——浪漫
        top_p = 50,  #控制query和key向量取值的范围，
        top_k = 50, #这里是取前50%
        do_sample = True, #采样 如果打开 model不管什么概率的样本都可能选中 会导致每次回答基本不一样
        #开启采样也是实现自我一致性的基础
        num_return_sequences = 5, #让model一次返回5个不同的回复
        
    )
    '''
    # 【步骤 D】: 把模型新吐出来的 Token 截取出来
    # 因为生成的数字里包含了你输入的 Prompt，我们需要把新吐出的字切片出来
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    # 【步骤 E】: Decode（解码）
    # 把数字重新翻译回人类看懂的汉字字符串，并跳过特殊控制符（skip_special_tokens=True）
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    # for rep in response:
    #     print(f"🤖 本地模型: {rep}")
    print(response)
    '''

  

    # 【步骤 D】: 把模型新吐出来的 Token 截取出来
    # 因为生成的数字里包含了你输入的 Prompt，我们需要把新吐出的字切片出来
    cut_len  = len(model_inputs.input_ids[0])
    generated_ids = [
        output_ids[cut_len:] for output_ids in  generated_ids
    ]
    
    # 【步骤 E】: Decode（解码）
    # 把数字重新翻译回人类看懂的汉字字符串，并跳过特殊控制符（skip_special_tokens=True）
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    cnt = 1
    for rep in response:
        print(f"🤖 本地模型:第{cnt}条采样是{rep}")
        cnt+=1
    # print(response)

'''
    # 【步骤 D】: 批量切片，去掉每条回复前面的 Prompt 提示词
    # 因为输入是一句话，输出了 5 句话，model_inputs.input_ids[0] 的长度就是 Prompt 的长度
    input_len = len(model_inputs.input_ids[0])
    pruned_generated_ids = [output_ids[input_len:] for output_ids in generated_ids]
    
    # 【步骤 E】: Decode 批量解码
    # batch_decode 会直接返回一个包含 5 个字符串的 Python 列表
    responses = tokenizer.batch_decode(pruned_generated_ids, skip_special_tokens=True)
    
    # 【步骤 F】: 遍历列表，把 5 条回复整整齐齐地打印出来
    print(f"\n💡 自我一致性（Self-Consistency）生成的 5 个候选答案：")
    print("=" * 50)
    for index, resp in enumerate(responses):
        print(f"📄 路径 {index + 1}：")
        print(resp.strip())
        print("-" * 50)
'''