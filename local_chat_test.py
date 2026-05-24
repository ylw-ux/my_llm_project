import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import set_seed

set_seed(42) # 设置随机种子初始化
# 计算机 所有计算权重、随机权重dropout的随机生成都是伪随机 让初始的随机权重为定值，
# 这样后面的随机变量就会更加固定，能保证相同的问题回答同一个答案


MODEL_PATH = "./models/Qwen2.5-1.5B-Instruct"

print("⏳ 正在通过 transformers 加载本地模型和分词器...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype="auto"  #
)

print("\n🎉 本地大模型加载成功！你可以开始和它对话了。")
print("输入 'quit' 退出。")
print("-" * 50)

while True:
    user_input = input("\n👤 我: ").strip()
    if not user_input or user_input.lower() == 'quit':
        break
        

    messages = [
        {"role": "system", "content": "你是一个运行在本地的AI助理。"},
        {"role": "user", "content": user_input}
    ]
    

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    print("🤖 思考中...")
    

    generated_ids = model.generate(
        **model_inputs,   #上下文传入
        max_new_tokens=200, #最大输出token
        min_new_tokens = 20,
        temperature=0.7, #温度 随机性，越大输出越多变， 低———理性 高——浪漫
        top_p = 50,  #控制query和key向量取值的范围，
        top_k = 50, #这里是取前50%
        #do_sample = True, #采样 如果打开 model不管什么概率的样本都可能选中 会导致每次回答基本不一样
        #开启采样也是实现自我一致性的基础
       # num_return_sequences = 5, #让model一次返回5个不同的回复
        
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

  

    imput_cut_len  = len(model_inputs.input_ids[0])
    generated_ids = [
        output_ids[imput_cut_len:30] for output_ids in  generated_ids
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    cnt = 1
    for rep in response:
        print(f"🤖 本地模型:第{cnt}条采样是{rep}")
        cnt+=1


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