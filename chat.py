from openai import OpenAI
import json
from dotenv import load_dotenv
import os



# ==========================================
# 1. 基础拼图：加载 .env 里的隐藏密钥
# ==========================================
load_dotenv()  # 这行代码会自动去读你创建的 .env 文件
api_key = os.getenv("DEEPSEEK_API_KEY")

# 异常处理：如果你忘记配置密钥，程序会友好地提醒你，而不是直接崩溃
if not api_key or "这里填入" in api_key:
    print("❌ 错误：未能在 .env 文件中检测到有效的 API 密钥！")
    print("请检查 .env 文件中的 DEEPSEEK_API_KEY 是否填写正确。")
    exit()

# ==========================================
# 2. 初始化大模型客户端
# ==========================================
# 这里我们用的是 DeepSeek 的官方接口，它完美兼容 openai 库
# 这里创建了一个api的实例
client = OpenAI(
    api_key=api_key, 
    base_url="https://api.deepseek.com"
)

# ==========================================
# 3. 基础拼图：文件处理（读取历史聊天记录）
# ==========================================
HISTORY_FILE = "chat_history.json"#聊天记录储存的文件地址
messages = [] #一个字典，用来接收这次对话所有的数据，最后再把它存到目标文件地址的文件里
# 初始化一个保存的判断参数
save = 1

# 如果之前聊过天，就把历史记录读出来，让大模型拥有“记忆”
if os.path.exists(HISTORY_FILE): #os.path.exists(HISTORY_FILE)检查路径是否存在
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            # 基础拼图：将文件里的 JSON 字符串转回 Python 的列表/字典
            messages = json.load(f)
            print(f" 成功加载历史聊天记录（共 {len(messages)} 条）...")
    except Exception as e:
        print(f" 读取历史记录失败（可能文件损坏）: {e}，将开启新对话。")

# 如果是没有历史记录的新对话，给大模型设定一个“人设”
if not messages:
    messages = [
        {"role": "system", "content": "你是一个幽默、耐心的编程导师，喜欢用通俗易懂的比喻解释问题。"}
    ]

print("\n🤖 机器人已上线！输入 'quit' 或 'exit' 退出聊天。")
print("-" * 50)

# ==========================================
# 4. 核心循环：开始聊天
# ==========================================
cur_mes_num = 0
while True:
    # 接收你的输入
    user_input = input("\n👤 我: ").strip()
    # 记录本次对话的条数
    cur_mes_num += 1
    if not user_input:
        continue
    if user_input.lower() in ["quit", "exit"]:
        print("\n👋 正在保存聊天记录并退出...")
        break
    # 一个判断，如果用户输入<<nosave<< 就不保存这场对话
    if user_input.lower() == "<<nosave<<" :
        print(f"收到，本次的{cur_mes_num-1}条对话不会记入历史对话中，马上我就忘了你了😭")
        print("程序结束。")
        save = 0
        break

    # 将你的话塞进对话历史列表里（字典结构）
    messages.append({"role": "user", "content": user_input})

    # 异常处理：防止因为网络抖动、断网导致程序直接崩溃
    try:
        print("🤖 思考中...") 
        
        # 呼叫大模型
        response = client.chat.completions.create(
            model="deepseek-chat",  # 对应 DeepSeek-V3 模型
            messages=messages,      # 把整个聊天历史发给它，它才知道上下文
            temperature=0.2,    # 控制创造力的超参数（0.0最严谨，1.0最有想象力）
            stream=True  
            # stop=["itisover"]       
        )

        # 基础拼图：嵌套字典解析
        # 模型的返回是一个复杂的对象，我们通过一层层剥开，拿到真正的回复文字
        # 此时的reply就是一个纯文本的形式了，相当于 "hfufsefh"
        reply = response.choices[0].message.content
        
        print(f"\n🤖 导师: {reply}")

        # 把大模型的回复也存入历史，大模型就知道它自己上一句说了什么
        messages.append({"role": "assistant", "content": reply})

    except Exception as e:
        print(f"\n❌ 请求失败，可能网络有波动，错误信息: {e}")
        # 如果失败了，把刚才用户发的那句话从历史里删掉，保证数据一致性
        messages.pop()

# ==========================================
# 5. 基础拼图：文件处理（退出前保存记录）
# ==========================================

if save == 1:
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            # 将列表/字典结构的数据，漂亮的格式化（indent=4）后写入文件
            json.dump(messages, f, ensure_ascii=False, indent=4)
        print(" 聊天记录已安全保存至 chat_history.json")
    except Exception as e:
        print(f"❌ 保存聊天记录失败: {e}")

    print("程序结束。")