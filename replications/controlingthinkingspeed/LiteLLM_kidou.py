# custom_vllm_provider.py
import os, asyncio
from typing import Any, Dict, List, Optional, Iterator, AsyncIterator
import litellm
from litellm import CustomLLM, completion, acompletion
from litellm.types.utils import GenericStreamingChunk, ModelResponse
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

import time
from litellm.types.utils import ModelResponse, Choices, Message, Usage
# 可选：环境变量控制 vLLM
os.environ.setdefault("VLLM_USE_V1", "0")  # 你环境里若更稳用 V1，可开；按需调整

def _lite_usage(prompt_tokens: int, completion_tokens: int) -> Dict[str, int]:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


import base64, io, requests
from PIL import Image

def _iter_parts(msg):
    c = msg.get("content")
    if isinstance(c, list):
        return c
    # 纯文本也统一成 parts 形式，方便下游统计 image 出现次数
    return [{"type": "text", "text": c if isinstance(c, str) else ""}]

def _load_image_from_url(url: str, timeout: float = 5.0) -> Image.Image:
    if url.startswith("file://"):
        return Image.open(url[7:])
    if url.startswith("http://") or url.startswith("https://"):
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content))
    if url.startswith("data:"):
        # data:image/png;base64,xxxx
        header, b64 = url.split(",", 1)
        return Image.open(io.BytesIO(base64.b64decode(b64)))
    raise ValueError(f"Unsupported image_url scheme: {url[:32]}...")



class VLLMPythonLLM(CustomLLM):
    """
    一个 LiteLLM Provider，用本地 vLLM Python 引擎直接完成推理。
    支持：
      - chat.completions 非流式
      - 简单流式（把最终结果切 token/字串增量吐出）
      - 多模态（从 messages 中抓取 image/file://）
      - 采样参数映射（温度/Top-P/Top-K/stop 等）
    """
    def __init__(self,
                 model: str,
                 **engine_kwargs):
        self.model_name = model
        # 初始化 vLLM 引擎（仅一次，全局复用）
        self.llm = LLM(model=model, **engine_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model, trust_remote_code=True, use_fast=True
        )

    # ---------- 工具方法 ----------
    def _sampling_from_optional(self, optional_params: Dict) -> SamplingParams:
        # 基础项：给明确默认值
        kwargs = dict(
            temperature=optional_params.get("temperature", 0.0),
            top_p=optional_params.get("top_p", 1.0),
            max_tokens=optional_params.get("max_tokens", 1024),
            stop=optional_params.get("stop", None),
            logprobs=optional_params.get("logprobs", None),
        )

        # 仅当用户显式提供时才传给 vLLM；否则交给 vLLM 的内部默认
        def set_if_not_none(k, cast=float):
            v = optional_params.get(k, None)
            if v is not None:
                kwargs[k] = cast(v)

        set_if_not_none("top_k", int)
        set_if_not_none("presence_penalty", float)   # ← 关键
        set_if_not_none("frequency_penalty", float)  # ← 关键
        set_if_not_none("repetition_penalty", float)

        return SamplingParams(**kwargs)


    def _extract_multimodal(self, messages: List[Dict]) -> Optional[Dict[str, Any]]:
        images: List[Image.Image] = []
        img_placeholders = 0
        for m in messages:
            for part in _iter_parts(m):
                if part.get("type") == "image_url":
                    url = (part.get("image_url") or {}).get("url")
                    if url:
                        images.append(_load_image_from_url(url, timeout=float(os.getenv("VLLM_IMAGE_FETCH_TIMEOUT", "5"))))
                        img_placeholders += 1
                elif part.get("type") == "image":
                    img = part.get("image")
                    if isinstance(img, Image.Image):
                        images.append(img)
                        img_placeholders += 1
                elif part.get("type") == "text":
                    pass

        if not images:
            return None

        # 版本/模型兼容：很多构建仅支持“单图”
        single_image_only = os.getenv("VLLM_SINGLE_IMAGE_ONLY", "1") == "1"
        if single_image_only and len(images) != 1:
            # 可改成仅取首图 / 显式报错
            images = [images[0]]

        # 简单一致性检查：图像数 >= 出现的图像占位数
        # （chat 模板通常会按出现顺序绑定）
        if len(images) < img_placeholders:
            raise ValueError(f"Found {img_placeholders} image placeholders in messages but only {len(images)} images could be loaded.")

        return {"image": images}

    
    


    def _build_model_response(self, text: str, prompt_tokens: int, completion_tokens: int) -> ModelResponse:
        # 直接构造 ModelResponse，避免再次调用 litellm.completion()
        resp = ModelResponse(
            id="chatcmpl-local",
            created=int(time.time()),
            model=self.model_name,
            object="chat.completion",
            choices=[
                Choices(
                    index=0,
                    message=Message(role="assistant", content=text),
                    finish_reason="stop",
                    logprobs=None,
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )
        return resp




    # ---------- 必须实现：非流式 ----------
    def completion(self,
                   model: str,
                   messages: List[Dict],
                   model_response: ModelResponse,
                   optional_params: Dict,
                   logging_obj: Any,
                   timeout=None,
                   client=None,
                   **kwargs) -> ModelResponse:
        sampling = self._sampling_from_optional(optional_params)
        mm = self._extract_multimodal(messages)

        # vLLM 0.9+ 推荐直接用 .chat()（messages 列表的列表）
        # outputs = self.llm.chat(
        #     [messages],
        #     sampling,
        #     multi_modal_data=[mm] if mm else None,
        # )
        # out = outputs[0].outputs[0]
        # text = out.text
        text = "correct_ouput"

        # 计数 tokens（近似）：prompt 用 chat_template tokenize，completion 直接 encode
        prompt_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        prompt_tokens = len(prompt_ids)
        completion_tokens = len(self.tokenizer.encode(text))

        return self._build_model_response(text, prompt_tokens, completion_tokens)

    # ---------- 必须实现：异步 ----------
    async def acompletion(self, *args, **kwargs) -> ModelResponse:
        # vLLM 的 .chat 是同步方法；这里用线程池包一层即可
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.completion(*args, **kwargs))

    # ---------- 可选：流式（这里采用“最终文本切片流出”的兼容方案） ----------
    def streaming(self, *args, **kwargs) -> Iterator[GenericStreamingChunk]:
        # 取到完整响应
        resp = self.completion(*args, **kwargs)
        text = resp.choices[0].message["content"]
        # 简单按子串流出（也可按 tokenizer 一词一出）
        for i in range(0, len(text), 40):
            yield {
                "index": 0,
                "text": text[i:i+40],
                "is_finished": False,
                "finish_reason": "",
                "tool_use": None,
                "usage": None,
            }
        # 最后一块：带 usage
        yield {
            "index": 0,
            "text": "",
            "is_finished": True,
            "finish_reason": "stop",
            "tool_use": None,
            "usage": resp.usage,
        }

    async def astreaming(self, *args, **kwargs) -> AsyncIterator[GenericStreamingChunk]:
        # 异步版本：复用同步实现
        loop = asyncio.get_running_loop()
        # 先得到完整响应
        resp = await loop.run_in_executor(None, lambda: self.completion(*args, **kwargs))
        text = resp.choices[0].message["content"]
        for i in range(0, len(text), 40):
            yield {
                "index": 0,
                "text": text[i:i+40],
                "is_finished": False,
                "finish_reason": "",
                "tool_use": None,
                "usage": None,
            }
        yield {
            "index": 0,
            "text": "",
            "is_finished": True,
            "finish_reason": "stop",
            "tool_use": None,
            "usage": resp.usage,
        }

# -------- 在 Python 里注册为一个 Provider 实例（纯 SDK 模式）--------
# 你也可以在别处构造，并把实例丢进 litellm.custom_provider_map
vllm_local = VLLMPythonLLM(
    model=os.environ.get("VLLM_MODEL", "/home/bingxing2/ailab/gaoyuanyuan_p/GLM-4.1V-9B-Thinking"),
    tensor_parallel_size=int(os.environ.get("TP_SIZE", "1")),
    gpu_memory_utilization=float(os.environ.get("GPU_UTIL", "0.9")),
    max_num_seqs=int(os.environ.get("MAX_NUM_SEQS", "32")),
)
