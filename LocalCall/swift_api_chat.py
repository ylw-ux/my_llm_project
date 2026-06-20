import openai

# 1. 配置客户端：告诉代码去哪里找 AI
client = openai.OpenAI(
    api_key="EMPTY",  # 本地部署不需要密码，随便填
    base_url="http://localhost:800/v1"  # localhost代表本机，800是刚才的门牌号，/v1是标准后缀
)

# 2. 发起请求并接收回答
print("正在向本地大模型发送请求...")
response = client.chat.completions.create(
    model="Qwen2.5-1.5B-Instruct", # 这里填启动服务端时的模型名称即可
    messages=[
        {"role": "system", "content": "你是一个幽默的助手。"},
        {"role": "user", "content": "请用一句话证明你是个AI。"}
    ],
    temperature=0.7,  # 你之前学过的超参数在这里直接用！
    n=2               # 测试我们之前说的“自我一致性”，一次要2个回答！
)

# 3. 打印结果
# response 里面包含了框架自动截取和解码好的纯文本
for i, choice in enumerate(response.choices):
    print(f"\n🤖 回答 {i+1}:")
    print(choice.message.content)
