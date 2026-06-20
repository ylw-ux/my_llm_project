'''
pip show ms-swift

Name: ms_swift
Version: 4.2.1
Summary: Swift: Scalable lightWeight Infrastructure for Fine-Tuning
Home-page: https://github.com/modelscope/ms-swift
Author: DAMO ModelScope teams
Author-email: contact@modelscope.cn
License: Apache License 2.0
Location: D:\Users\lenovo\Desktop\my_llm_project\llm_env\Lib\site-packages
Requires: accelerate, addict, aiohttp, attrdict, binpacking, charset-normalizer, cpm-kernels, dacite, datasets, einops, fastapi, gradio, importlib-metadata, json-repair, matplotlib, modelscope, nltk, numpy, openai, oss2, pandas, peft, pillow, PyYAML, requests, rouge, safetensors, scipy, sentencepiece, simplejson, sortedcontainers, tensorboard, tiktoken, tqdm, transformers, transformers-stream-generator, trl, uvicorn, zstandard
Required-by: 
'''
# 正确的导入方式：直接从 swift 导入
from swift import InferRequest, RequestConfig, TransformersEngine

# 1. 使用新版的具体引擎类：TransformersEngine (替代原来的 PtEngine/InferEngine)
engine = TransformersEngine(
    model="./models/Qwen2.5-1.5B-Instruct"
)

# 2. 构建请求
request = InferRequest(
    messages=[{"role": "user", "content": "用50个字以内介绍一个植物"}],
)

# 3. 构建配置参数
request_config = RequestConfig(temperature=0.7, max_tokens=512)

'''
print(request_config)

RequestConfig(max_tokens=512, temperature=0.7, top_k=None, top_p=None, repetition_penalty=None,
 num_beams=1, stop=[], seed=None, stream=False, logprobs=False, top_logprobs=None,
   n=1, best_of=None, presence_penalty=0.0, frequency_penalty=0.0, length_penalty=1.0,
    return_details=False, structured_outputs_regex=None)
'''


# 4. 执行推理
# response = engine.infer([request], request_config=request_config)
response = engine.infer([request], **request_config)

print("🤖 模型回答:", response[0].choices[0].message.content)

# 总之SWIFT只要三步
# 1，加载你的本地大模型 TransformersEngine model写上文件位置
# 2，不用去做复杂的分词en decode 截取回复token  直接发送推理请求InferRequest 在请求里把我们的上下文messages传进去
# RequestConfig方法 传递实参来控制超参数 通过赋值来后续给engine.infer使用
# 3，engine.infer 方法 你不再需要手动截取 token，框架返回的直接是干净的回答
# 这一步融合了发送messages 和调整超参数 和返回回复文本
# 4，print回复