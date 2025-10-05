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

num_question = 32

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


# ===  统计 fast / slow 的生成 token 数 ===
import numpy as np

def count_tokens_list(texts, tokenizer, exclude_special=False):
    """统计每条文本的 token 数；
       exclude_special=True 时会尽量去掉 special token（若有 special_tokens_mask）"""
    counts = []
    for t in texts:
        enc = tokenizer(t, add_special_tokens=False, return_special_tokens_mask=True)
        ids = enc["input_ids"]
        if exclude_special and "special_tokens_mask" in enc:
            ids = [tid for tid, m in zip(ids, enc["special_tokens_mask"]) if m == 0]
        counts.append(len(ids))
    return np.array(counts, dtype=np.int32)

fast_tok = count_tokens_list(ans_fast, tokenizer, exclude_special=False)
slow_tok = count_tokens_list(ans_slow, tokenizer, exclude_special=False)

def _summary(arr):
    return (f"mean={arr.mean():.1f}, median={np.median(arr):.0f}, "
            f"std={arr.std():.1f}, min={arr.min()}, max={arr.max()}, n={len(arr)}")

print("\n[Token stats]")
print("FAST :", _summary(fast_tok))
print("SLOW :", _summary(slow_tok))
diff = slow_tok - fast_tok
print("Δ(SLOW-FAST):", _summary(diff))




del llm_gen, answers_fast, answers_slow
gc.collect()
torch.cuda.empty_cache()




# === 2) “阶段”预览：找到答案内的第2个 "\n\n"，展示位置并打印截断前/后的文本 ===
def second_break_char_and_token(answer: str):
    """返回：(char_idx, token_idx_in_answer)；找不到则 (None, None)"""
    first = answer.find("\n\n")
    if first == -1:
        return None, None
    second = answer.find("\n\n", first + 2)
    if second == -1:
        return None, None

    # 将字符位置映射到“答案自身”的 token 索引
    enc = tokenizer(answer, add_special_tokens=False, return_offsets_mapping=True)
    tok_idx = None
    for t, (s, e) in enumerate(enc["offset_mapping"]):
        if s <= second < e:
            tok_idx = t
            break
        if second == e and t + 1 < len(enc["offset_mapping"]):
            tok_idx = t + 1
            break
    return second, tok_idx

def preview_stage(idx: int, which: str = "fast"):
    """打印第 idx 个样本（fast/slow）的完整回答与“阶段”回答"""
    prompt = texts_fast[idx] if which == "fast" else texts_slow[idx]
    answer = ans_fast[idx] if which == "fast" else ans_slow[idx]
    char_idx, tok_idx = second_break_char_and_token(answer)
    print(f"\n=== [{which.upper()}] sample #{idx} ===")
    print("Question:", problem_list[idx])
    if char_idx is None:
        print("No second paragraph break found; showing full answer.\n")
        print(answer)
        return
    print(f"Second break -> char={char_idx}, token={tok_idx}")
    print("\n--- FULL ANSWER ---")
    print(answer)
    print("\n--- TRUNCATED (until 2nd \\n\\n) ---")
    print(answer[:char_idx])

# # Demo：查看第 0 个样本的 fast / slow
# preview_stage(0, "fast")
# preview_stage(0, "slow")





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

# （可选）看看有多少样本找到了“第二个段落分隔”的锚点
print(f"Anchor coverage - FAST: {sum(p is not None for p in fast_pos)}/{len(fast_pos)}")
print(f"Anchor coverage - SLOW: {sum(p is not None for p in slow_pos)}/{len(slow_pos)}")




# ===== INSERT: 抽隐层专用短版 QA + 锚点 =====
MAX_TOK_HS = 4096   # 抽隐层的 token 上限（可改 2048/3072）
SAFETY     = 64     # 预留安全余量，避免刚好顶到上限

def _second_break_char(answer: str):
    first = answer.find("\n\n")
    if first == -1:
        return None
    second = answer.find("\n\n", first + 2)
    return None if second == -1 else second

def _boundary_token_idx(qa: str, boundary_char: int, tokenizer) -> int:
    """
    在单次分词(qa)上，把字符边界 boundary_char 映射到 token 索引。
    与你 first_answer_token_idx 的写法一致，但作用在任意边界。
    """
    enc = tokenizer(qa, add_special_tokens=False, return_offsets_mapping=True)
    # 保护：空序列~
    if len(enc["input_ids"]) == 0:
        return 0
    for t, (s, e) in enumerate(enc["offset_mapping"]):
        if s <= boundary_char < e:
            return t
        if boundary_char == e and t + 1 < len(enc["offset_mapping"]):
            return t + 1
    return len(enc["input_ids"]) - 1

def _truncate_qa_and_anchor(prompt: str, answer: str, tokenizer,
                            prefer_second_break: bool = True,
                            max_tokens: int = MAX_TOK_HS, safety: int = SAFETY):
    """
    返回：(qa_hs, anchor_tok_idx)
      - 先按“第二段分隔”截答案（若存在）
      - 然后拼 prompt 检查 token 上限；若超上限则二次截断到上限
      - 锚点 token：在“截断边界”处（段落截则是段落边界；上限截则是序列末端）
    """
    # 1) 先在答案里找“第二段落分隔”
    ans_part = answer
    second_ci = _second_break_char(answer) if prefer_second_break else None
    if second_ci is not None:
        ans_part = answer[:second_ci]

    # 2) 组成短版 qa，并检查 token 上限
    qa = prompt + ans_part
    enc = tokenizer(qa, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    cap = max_tokens - safety

    # 锚点字符边界（尚未考虑上限二次截断）
    boundary_char = len(prompt) + len(ans_part)

    if cap > 0 and len(ids) > cap:
        # 发生了“上限截断”，锚点就是最后一个 token
        ids = ids[:cap]
        qa = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        anchor_tok_idx = len(ids) - 1
    else:
        # 没到上限：锚点 = 段落边界处对应的 token
        anchor_tok_idx = _boundary_token_idx(qa, boundary_char, tokenizer)

    return qa, anchor_tok_idx

def build_hs_pack(text_prompts, answers, tokenizer,
                  prefer_second_break=True, max_tokens=MAX_TOK_HS, safety=SAFETY):
    qa_hs = []
    pos_hs = []
    for i in range(len(text_prompts)):
        qa_i, pos_i = _truncate_qa_and_anchor(text_prompts[i], answers[i], tokenizer,
                                              prefer_second_break, max_tokens, safety)
        qa_hs.append(qa_i)
        pos_hs.append(pos_i)
    return qa_hs, pos_hs



# # —— 用在 fast/slow 上 —— #
# qa_fast, fast_pos = build_hs_pack(texts_fast, ans_fast, tokenizer)
# qa_slow, slow_pos = build_hs_pack(texts_slow, ans_slow, tokenizer)

# # 可选：看长度分布
# def _toklen(t): 
#     return len(tokenizer(t, add_special_tokens=False)["input_ids"])
# fast_hs_len = np.array([_toklen(t) for t in qa_fast], dtype=np.int32)
# slow_hs_len = np.array([_toklen(t) for t in qa_slow], dtype=np.int32)
# print(f"[HS-short] fast mean={fast_hs_len.mean():.1f}, max={fast_hs_len.max()}, n={len(fast_hs_len)}")
# print(f"[HS-short] slow mean={slow_hs_len.mean():.1f}, max={slow_hs_len.max()}, n={len(slow_hs_len)}")

# del fast_hs_len,slow_hs_len





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
    # print('picking'+str(len(hs_layers[0]))+' '+str(token_idx))
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

directcollect = False
all_hidden_states = []
if directcollect:
    hidden_fast, _= get_all_hidden_states(llm_hs, qa_fast)
    hidden_slow, _= get_all_hidden_states(llm_hs, qa_slow)

    # finishing getting hidden state

    n_layers = len(hidden_fast[0])
    assert n_layers > 0, "Failed to get hidden states."


    
    for i in range(len(qa_fast)):
        all_hidden_states.append(safe_pick(hidden_fast[i], fast_pos[i]))
    for i in range(len(qa_slow)):
        all_hidden_states.append(safe_pick(hidden_slow[i], slow_pos[i]))

    
else:
    BATCH = 8  # 或 4/16，视内存而定

    fast_idx_list, slow_idx_list = [], []
    all_hidden_states = []

    def collect(which, hs_list, pos_list):
        # which: "fast" or "slow"
        for i in range(len(hs_list)):
            all_hidden_states.append(safe_pick(hs_list[i], pos_list[i]))
            if which == "fast":
                fast_idx_list.append(len(all_hidden_states) - 1)
            else:
                slow_idx_list.append(len(all_hidden_states) - 1)
        del hs_list[:]
        gc.collect(); torch.cuda.empty_cache()

    # --- fast ---
    for s in range(0, len(qa_fast), BATCH):
        chunk = qa_fast[s:s+BATCH]
        hs_chunk, _ = get_all_hidden_states(llm_hs, chunk, split_by_samples=True)
        for i in range(30):
            if len(chunk) != len(hs_chunk):
                print(f"[warn] retry {i} times for batch {s}-{s+len(chunk)}")
                hs_chunk, _ = get_all_hidden_states(llm_hs, chunk, split_by_samples=True)
        print(f"[diag] fast batch {s}-{s+len(chunk)}: want={len(chunk)} got={len(hs_chunk)}")
        collect("fast",hs_chunk, fast_pos[s:s+BATCH])

    # --- slow ---
    for s in range(0, len(qa_slow), BATCH):
        chunk = qa_slow[s:s+BATCH]
        hs_chunk, _ = get_all_hidden_states(llm_hs, chunk, split_by_samples=True)
        print(f"[diag] fast batch {s}-{s+len(chunk)}: want={len(chunk)} got={len(hs_chunk)}")
        collect("slow",hs_chunk, slow_pos[s:s+BATCH])

N = len(qa_fast)

positive_indices = list(range(N, 2*N))    # fast
negative_indices = list(range(0, N))   # slow

print(f"[diag] problems={len(problem_list)} "
      f"qa_fast={len(qa_fast)} qa_slow={len(qa_slow)}")


# 收集完：
print(f"[diag] collected fast={len(fast_idx_list)} slow={len(slow_idx_list)} "
      f"total={len(all_hidden_states)}")



control_vector = extract_pca_control_vector(
    all_hidden_states=all_hidden_states,
    positive_indices=positive_indices,
    negative_indices=negative_indices,
    model_type="glm",     # <--- 不要再用 "qwen2.5"
    method="diff_bid", # center
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


