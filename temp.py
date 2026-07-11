import os
from openai import OpenAI

try:
    client = OpenAI(
        # 若没有配置环境变量，请用阿里云百炼API Key将下行替换为: api_key="sk-xxx",
        api_key="sk-ws-H.RYHPHHR.myrS.MEYCIQC20M54ZPqedfJBczX6ibFWreicSJY8ZBBTgWNRHSilFAIhAJYxzFAhk4_SymVaiIUVYpSFB8uB8EaW5AFoNqKX6sKw",
        # 以下为华北2（北京）地域的URL，各地域的URL不同。调用时请将{WorkspaceId}替换为真实的业务空间ID。
        base_url="https://llm-w9d5heik4oxcaj3y.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
    )

    completion = client.chat.completions.create(
        model="qwen-plus",  # 模型列表: https://help.aliyun.com/model-studio/getting-started/models
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': '你是谁？'}
        ]
    )
    print(completion.choices[0].message.content)
except Exception as e:
    print(f"错误信息：{e}")
    print("请参考文档：https://help.aliyun.com/model-studio/developer-reference/error-code")