import os
import vllm
import json
from easysteer.hidden_states import get_all_hidden_states
from vllm import LLM,SamplingParams

from easysteer.steer import extract_pca_control_vector,StatisticalControlVector
from transformers import AutoTokenizer
import torch, gc

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["VLLM_USE_V1"] = "0"

# file_path = "../../temp/test.jsonl" #MATH-500
file_path = "/home/bingxing2/ailab/gaoyuanyuan_p/yuning/temp/test.jsonl" #GSM8K

model_path = "/home/bingxing2/ailab/gaoyuanyuan_p/GLM-4.1V-9B-Thinking"

# vector_path = "vectors/thinking_switch_pca_MATH-500.gguf" #MATH-500
vector_path = "GLM_MATH500_20.gguf"

num_question = 20

problem_list = []

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        line_count = 0
        for line in f:
            if line_count >= num_question:
                break
            stripped_line = line.strip()
            if not stripped_line:
                continue
                
            try:
                data = json.loads(stripped_line)
                if "problem" in data:
                    problem_list.append(data["problem"])
                    line_count += 1
                elif "question" in data:
                    problem_list.append(data["question"])
                    line_count += 1
            except json.JSONDecodeError:
                print(f"skip: {line[:50]}...")
                continue

except Exception as e:
    print(f"{str(e)}")
# problem_list = ["Find the roots of $(x - 3)^3 + (x -7)^3 = (2x - 10)^3.$","A regular hexagon can be divided into six equilateral triangles. If the perimeter of one of the triangles is 21 inches, what is the perimeter, in inches, of the regular hexagon?",""]

# generate texts

tokenizer = AutoTokenizer.from_pretrained(
    model_path, trust_remote_code=True, use_fast=True
)

# 假设已有 tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)

def build_glm_prompt(q: str,
                     mode: str = "base",
                     sys_msg: str | None = None,
                     add_think_tag: bool = True) -> str:
    """
    mode: "base" | "fast" | "slow"
      - fast  : assistant 起始处补 "<think>\\nTo"
      - slow  : assistant 起始处补 "<think>\\nAlright"
      - base  : 不做额外偏置（也可加 think，见 add_think_tag）
    """
    user = f"Return your final response within \\boxed{{}}. {q}"

    msgs = []
    if sys_msg:
        msgs.append({"role": "system", "content": sys_msg})
    msgs.append({"role": "user", "content": user})

    # 优先用 GLM 的 chat_template
    try:
        prompt = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # 回退到手搓的小写 GLM 风格（注意是小写）
        prompt = f"<|user|>\n{user}\n<|assistant|>\n"

    # ——在 assistant 段刚开始处，按模式插入 think+seed（与老模板对齐）——
    if add_think_tag:
        if mode == "fast":
            seed = "To "
        elif mode == "slow":
            seed = "Alright, "
        else:
            seed = ""  # base 模式不强行加种子词
        prompt += "<think>" + seed

    return prompt


texts_fast = [build_glm_prompt(q, "fast") for q in problem_list]
texts_slow = [build_glm_prompt(q, "slow") for q in problem_list]
print('---test----prompt_build')
s = texts_fast[0]
print(repr(s))


llm_gen = LLM(
    model=model_path,
    task="generate",
    tensor_parallel_size=1,
    # 40G: 建议别开 eager，内存更吃紧；按需调高/调低利用率
    enforce_eager=True,
    gpu_memory_utilization=0.90,
    trust_remote_code=True,
    max_model_len=8192
)
gen_params = SamplingParams(
    temperature=0.0,
    max_tokens=1024,           # 不要 4096，40G 压力很大；按实际需要增减
    skip_special_tokens=False, # 需要和 hidden_states 对齐，建议先保留特殊符号
)

answers_fast = llm_gen.generate(
    texts_fast,
    gen_params
)




answers_slow = llm_gen.generate(
    texts_slow,
    gen_params
)



# new tiqu code
ans_fast = [o.outputs[0].text for o in answers_fast]
qa_fast  = [texts_fast[i] + ans_fast[i] for i in range(len(texts_fast))]

ans_slow = [o.outputs[0].text for o in answers_slow]
qa_slow  = [texts_slow[i] + ans_slow[i] for i in range(len(texts_slow))]




del llm_gen, answers_fast, answers_slow
gc.collect()
torch.cuda.empty_cache()







def first_answer_token_idx(prompt_str: str, answer_str: str, tokenizer) -> int:
    # 答案在整段里的字符边界：恰好是 prompt 的长度
    boundary = len(prompt_str)

    # 在 "prompt + answer" 一次性分词，保证与实际喂给 vLLM 的序列一致
    qa = prompt_str + answer_str
    enc = tokenizer(qa, add_special_tokens=False, return_offsets_mapping=True)

    # 找到覆盖边界字符的 token
    for t, (s, e) in enumerate(enc["offset_mapping"]):
        if s <= boundary < e:
            return t
        # 边界刚好落在上一个 token 的结尾时，取下一个 token
        if boundary == e and t + 1 < len(enc["offset_mapping"]):
            return t + 1

    # 兜底：取最后一个 token（理论上不会走到）
    return len(enc["input_ids"]) - 1


# not second

# fast_pos = [first_answer_token_idx(texts_fast[i], ans_fast[i], tokenizer)
#             for i in range(len(qa_fast))]
# slow_pos = [first_answer_token_idx(texts_slow[i], ans_slow[i], tokenizer)
#             for i in range(len(qa_slow))]






def second_paragraph_break_token_pos(text: str):
    # 找第2个 "\n\n" 的字符区间
    k = 0
    start = -1
    for i in range(len(text)-1):
        if text[i:i+2] == "\n\n":
            k += 1
            if k == 2:
                start = i
                break
    if start < 0:
        return None

    enc = tokenizer(
        text, add_special_tokens=True,
        return_offsets_mapping=True
    )
    # 找到“覆盖 start 字符”的 token 索引
    for idx, (s, e) in enumerate(enc["offset_mapping"]):
        if s <= start < e:
            return idx
    # 找不到就返回最后一个 token
    return len(enc["input_ids"]) - 1

# 逐样本计算 fast/slow 的锚点，并做边界保护
def pick_pos_list(text_list):
    pos_list = []
    for t in text_list:
        pos = second_paragraph_break_token_pos(t)
        pos_list.append(pos)
    return pos_list



fast_pos = pick_pos_list(qa_fast)
slow_pos = []
for i, t in enumerate(qa_slow):
    # 用与 fast 最接近长度的段落换行（如果 fast 没有，直接用 slow 的第二个）
    p = second_paragraph_break_token_pos(t)
    if p is None:
        slow_pos.append(p)
    else:
        slow_pos.append(p)





import numpy as np, json


def _to_numpy_f32(x):
    # 把各种可能的类型都稳妥转成 numpy.float32 的一维向量
    if isinstance(x, torch.Tensor):
        # 放到 CPU，并转成 float32，再转 numpy
        return x.detach().to(dtype=torch.float32, device="cpu").contiguous().view(-1).numpy()
    x = np.asarray(x)
    if x.dtype != np.float32:
        x = x.astype(np.float32, copy=False)
    return x.reshape(-1)

def safe_pick(hs_layers, token_idx):
    """hs_layers: List[layer]，每个元素是 List[seq_len] 的向量(通常是 torch.Tensor)
       返回形状：List[layer][1][H]，且最里层是 numpy.float32
    """
    seq_len = len(hs_layers[0])
    if token_idx is None:
        token_idx = seq_len - 1
    token_idx = max(0, min(token_idx, seq_len - 1))

    out = []
    for L in range(len(hs_layers)):
        v = hs_layers[L][token_idx]         # 可能是 torch.bfloat16 GPU Tensor
        v = _to_numpy_f32(v)                # 转成 numpy.float32 (H,)
        out.append([v])                     # 变成 [1, H]
    return out


llm_hs = LLM(model=model_path,task="reward",tensor_parallel_size=1,trust_remote_code=True,enforce_eager=True,    gpu_memory_utilization=0.90,max_model_len=8192)

hidden_fast, _= get_all_hidden_states(llm_hs, qa_fast)
hidden_slow, _= get_all_hidden_states(llm_hs, qa_slow)

# finishing getting hidden state

n_layers = len(hidden_fast[0])
assert n_layers > 0, "Failed to get hidden states."


all_hidden_states = []
for i in range(len(qa_fast)):
    all_hidden_states.append(safe_pick(hidden_fast[i], fast_pos[i]))
for i in range(len(qa_slow)):
    all_hidden_states.append(safe_pick(hidden_slow[i], slow_pos[i]))

N = len(qa_fast)


positive_indices = list(range(0, N))     # fast
negative_indices = list(range(N, 2*N))   # slow



control_vector = extract_pca_control_vector(
    all_hidden_states=all_hidden_states,
    positive_indices=positive_indices,
    negative_indices=negative_indices,
    model_type="glm",     # <--- 不要再用 "qwen2.5"
    method="center",
    token_pos=-1,         # 我们已经在 all_hidden_states 里把 seq 砍到 1 了，这里不会再用到
    normalize=False
)


# from easysteer.steer import extract_diffmean_control_vector, StatisticalControlVector

# control_vector = extract_diffmean_control_vector(
#     all_hidden_states=all_hidden_states,
#     positive_indices=positive_indices,   # slow
#     negative_indices=negative_indices,   # fast
#     model_type="glm",     # 见第4条
#     token_pos=-1,
#     normalize=False
# )

control_vector.export_gguf(vector_path)
control_vector = StatisticalControlVector.import_gguf(vector_path)
print(control_vector)




# from transformers import AutoTokenizer

# # Initialize tokenizer
# tokenizer = AutoTokenizer.from_pretrained(
#     model_path, trust_remote_code=True, use_fast=True
# )
# # tokenizer = AutoTokenizer.from_pretrained("/home/bingxing2/ailab/gaoyuanyuan_p/GLM-4.1V-9B-Thinking")

# # The newline token suffix in tokenizer vocabulary
# target_suffix = "ĊĊ"  # "\n\n" is tokenized as "ĊĊ"

# all_hidden_states=[]
# # Process each QA pair to find newline positions
# fast_token = []
# fast_positions = []
# for i,answer in enumerate(qa_pairs_fast):
#     # Tokenize the QA pair
#     tokens = tokenizer.tokenize(answer, add_special_tokens=True)
#     fast_token.append(tokens)
    
#     # Find all positions of "ĊĊ" in the tokens
#     # These represent potential paragraph breaks in the text
#     positions = [
#         i for i, token in enumerate(tokens) 
#         if isinstance(token, str) and token.endswith(target_suffix)
#     ]
#     second_position = positions[1] if len(positions) > 1 else None
#     fast_positions.append(second_position)
#     print(len(hidden_states_fast[i][0]))
    
#     all_hidden_states.append([
#         [hidden_states_fast[i][layer][second_position]]  # 注意这里多加了一层方括号
#         for layer in range(28)
#     ])
#     # all_hidden_states.append([
#     #     hidden_states_slow[i][layer][second_position]
#     #     for layer in range(19, 28)
#     # ])

# # Process slow-thinking responses to find truncation positions
# slow_tokens = []
# slow_truncation_positions = []
# for i, answer in enumerate(qa_pairs_slow):
#     # Tokenize the response
#     tokens = tokenizer.tokenize(answer, add_special_tokens=True)
#     slow_tokens.append(tokens)
    
#     # Get the corresponding slow-thinking second position
#     fast_second_pos = fast_positions[i]
    
#     # if fast_second_pos is None:
#     #     slow_truncation_positions.append(None)
#     #     continue
    
#     # Find all positions of "ĊĊ" in slow-thinking response
#     slow_positions = [
#         j for j, token in enumerate(tokens) 
#         if isinstance(token, str) and token.endswith(target_suffix)
#     ]
    
#     # if not slow_positions:
#     #     slow_truncation_positions.append(None)
#     #     continue
    
#     # Find the position with nearest length to fast-thinking response
#     closest_position = min(slow_positions, key=lambda x: abs(x - fast_second_pos))
#     slow_truncation_positions.append(closest_position)
    
#     all_hidden_states.append([
#         [hidden_states_slow[i][layer][closest_position]]  # 注意这里多加了一层方括号
#         for layer in range(28)
#     ])



# control_vector = extract_pca_control_vector(
#     all_hidden_states=all_hidden_states,
#     positive_indices=list(range(num_question)), 
#     negative_indices=list(range(num_question,2*num_question)),
#     model_type="qwen2.5",
#     method="center",
#     token_pos=-1,
#     normalize=False
# )

# control_vector.export_gguf(vector_path)
# control_vector = StatisticalControlVector.import_gguf(vector_path)
# print(control_vector)