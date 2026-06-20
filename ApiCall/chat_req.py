
import os
import requests
import json
from dotenv import load_dotenv

# 1. 准备密钥
load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")

# 2. 准备 Endpoint (目标地址)
url = "https://api.deepseek.com/chat/completions"

# 3. 组装 Headers (把 API 密钥藏进请求头)
# 注意这里的格式是官方文档严格规定的
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

# 4. 组装 Payload (打包你要对大模型说的话和超参数)
data = {
    "model": "deepseek-chat",
    "messages": [
        {"role": "user", "content": "用一句话解释什么是Python的字典？"}
    ],
    "temperature": 0.7
}

print("发送请求中...")

# 5. 发送网络请求 (相当于邮递员发车)
# 使用刚才学过的异常处理来应对网络断开的情况
try:
    # 使用 requests.post 方法，把 url, 头信息, 和转换成 JSON 格式的包裹发出去
    response = requests.post(url, headers=headers, json=data)
    
    # 检查如果状态码不是 200 (成功)，就抛出异常
    response.raise_for_status()
    
    # 把收到的复杂 JSON 回复，解析成 Python 字典
    result_dict = response.json()
    
    # 手动剥洋葱，一层层提取出回复文本
    reply = result_dict['choices'][0]['message']['content']
    print(f"🤖 收到回复: {reply}")

except requests.exceptions.RequestException as e:
    print(f"❌ 网络请求出错了: {e}")



# from dotenv import load_dotenv
# import requests
# import os
# 第二遍
# load_dotenv
# key = os.getenv("DEEPSEEK_API_KEY")
# r = requests.post(
#     "https://api.deepseek.com/v1/chat/completions",
#     headers={
#         "Authorization" : f"Bearer {key}",
#         "Content-Type" : "application/json"
#     },
#     json={
#         "model":"deepseek-chat",
#         "messages":[
#             {"role":"assistant","content":"hello"},
#             {"role":"user","content":"用一句话解释什么是闭包"},

#         ]
#     },
#     timeout=30
# )

# ans = r.json()
# anss= ans["choices"][0]["message"]["content"]
# print(anss)

# 第一遍
# load_dotenv()

# resp = requests.post(
#     "https://api.deepseek.com/v1/chat/completions",
#     headers={
#         "Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY')}",
#         "Content-Type": "application/json",
#     },
#     json={
#         "model": "deepseek-chat",
#         "messages": [
#             {"role": "user", "content": "用一句话解释什么是闭包"}
#         ],
#     },
#     timeout=30,
# )

# data = resp.json()
# print(data["choices"][0]["message"]["content"])
