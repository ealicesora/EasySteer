# demo_sdk.py
import litellm
from LiteLLM_kidou import vllm_local

# 关键：把你的 provider 挂到 LiteLLM 的映射里
litellm.custom_provider_map = [
    {"provider": "vllm-python", "custom_handler": vllm_local}
]

resp = litellm.completion(
    model="vllm-python/glm4",              # 任意前缀名；用于路由到你的 handler
    messages=[{"role": "user", "content": "给我一首 4 句短诗"}],
    temperature=0.2,
    max_tokens=128,
)
print(resp.model, resp.usage, resp.choices[0].message["content"])
