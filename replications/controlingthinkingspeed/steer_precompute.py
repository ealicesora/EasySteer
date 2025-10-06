import os
import vllm
import json
import argparse
from easysteer.hidden_states import get_all_hidden_states,get_all_hidden_states_capture,get_all_hidden_states_by_capture
from vllm import LLM,SamplingParams

from easysteer.steer import extract_pca_control_vector,StatisticalControlVector
from transformers import AutoTokenizer
import torch, gc

os.environ["VLLM_USE_V1"] = "0"


def load_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_config(config, save_dir):
    """Save configuration to the output directory."""
    os.makedirs(save_dir, exist_ok=True)
    config_save_path = os.path.join(save_dir, 'precompute_config.json')
    with open(config_save_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"Configuration saved to: {config_save_path}")


def main(config_path):
    # Load configuration
    config = load_config(config_path)

    # Extract configuration parameters
    base_path = config['base_path']
    os.makedirs(base_path, exist_ok=True)
    model_path = config['model_path']
    vector_path = base_path + config['vector_path']
    data_file = config.get('data_file', '/home/bingxing2/ailab/gaoyuanyuan_p/yuning/temp/test.jsonl')
    slow_prefix = config['slow_prefix'] if 'slow_prefix' in config else "Alright"
    print('using' + data_file)
    num_question = config.get('num_question', 256)
    pca_method = config.get('pca_method', 'center')  # center, pca, etc.

    # Save config to output directory
    # save_config(config, save_dir)

    problem_list = []

    try:
        with open(data_file, 'r', encoding='utf-8') as f:
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
    # print(problem_list)
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
                seed = "To"
            elif mode == "slow":
                seed = ""
                seed = slow_prefix
            else:
                seed = ""  # base 模式不强行加种子词
            prompt += "<think>\n" + seed

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
            enc = tokenizer(t, add_special_tokens=True, return_special_tokens_mask=True)
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
        enc = tokenizer(answer, add_special_tokens=True, return_offsets_mapping=True)
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
    preview_stage(0, "fast")
    preview_stage(0, "slow")
    
    
    
    
    
    
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
            text, add_special_tokens=False,
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
    
    
    
    fast_pos_preview = pick_pos_list(qa_fast)
    slow_pos_preview = []
    for i, t in enumerate(qa_slow):
        # 用与 fast 最接近长度的段落换行（如果 fast 没有，直接用 slow 的第二个）
        p = second_paragraph_break_token_pos(t)
        if p is None:
            slow_pos_preview.append(p)
        else:
            slow_pos_preview.append(p)
    
    # （可选）看看有多少样本找到了“第二个段落分隔”的锚点
    print(f"Anchor coverage - FAST: {sum(p is not None for p in fast_pos)}/{len(fast_pos_preview)}")
    print(f"Anchor coverage - SLOW: {sum(p is not None for p in slow_pos)}/{len(slow_pos_preview)}")
    
    
    
    # ===== 抽隐层专用短版 QA + 锚点（与论文对齐：slow 取与 fast 初始段最接近长度） =====
    MAX_TOK_HS = 4096   # 抽隐层 token 上限（含 prompt+初始段）
    SAFETY     = 64     # 预留余量，避免刚好顶到上限导致偏差或引擎侧 pack 差异

    def _step_break_positions(answer: str):
        """返回所有 '\n\n' 的起始字符下标列表，用于按“步”切分。"""
        pos = []
        start = 0
        while True:
            i = answer.find("\n\n", start)
            if i == -1:
                break
            pos.append(i)
            start = i + 2
        return pos

    def _answer_prefix_by_k_steps(answer: str, k: int) -> str:
        """保留前 k 步（k 次 '\n\n' 之前的内容）。若不足 k 步，则返回全文。"""
        if k <= 0:
            return ""
        brks = _step_break_positions(answer)
        if len(brks) >= k:
            return answer[:brks[k-1]]
        return answer

    def _boundary_token_idx(qa: str, boundary_char: int, tokenizer) -> int:
        """
        在单次分词(qa)上，把字符边界 boundary_char 映射到 token 索引。
        """
        enc = tokenizer(qa, add_special_tokens=True, return_offsets_mapping=True)
        if len(enc["input_ids"]) == 0:
            return 0
        for t, (s, e) in enumerate(enc["offset_mapping"]):
            if s <= boundary_char < e:
                return t
            if boundary_char == e and t + 1 < len(enc["offset_mapping"]):
                return t + 1
        return len(enc["input_ids"]) - 1


    def _last_token_before_char(qa: str, boundary_char: int, tokenizer) -> int:
        """
        返回“边界前最后一个 token”的索引（严格落在初始段内）。
        boundary_char = len(prompt) + len(ans_part)
        """
        enc = tokenizer(qa, add_special_tokens=True, return_offsets_mapping=True)
        ids = enc["input_ids"]; offs = enc["offset_mapping"]
        if not ids: return 0
        target = max(0, boundary_char - 1)  # 边界前一个字符
        # 优先找覆盖 target 的 token
        for t, (s, e) in enumerate(offs):
            if s <= target < e:
                return t
        # 兜底：找 e <= target 的最后一个 token
        for t in range(len(offs)-1, -1, -1):
            if offs[t][1] <= target:
                return t
        return 0

    def _token_len_of_qa(prompt: str, ans_part: str, tokenizer) -> int:
        """返回 tokenize(prompt + ans_part, add_special_tokens=True) 的 token 数。"""
        return len(tokenizer(prompt + ans_part, add_special_tokens=True)["input_ids"])

    def _truncate_qa_and_anchor_from_ans_part(prompt: str, ans_part: str, tokenizer,
                                            max_tokens: int = MAX_TOK_HS, safety: int = SAFETY):
        """
        给定已选好的 ans_part（初始段），拼 qa 并计算锚点：
        - 若未触顶：锚点=初始段字符边界映射到的 token
        - 若触上限(cap=max_tokens-safety)：截到 cap，锚点=末 token
        返回：(qa_hs, anchor_tok_idx)
        """
        qa = prompt + ans_part
        enc = tokenizer(qa, add_special_tokens=True, return_offsets_mapping=True)
        ids = enc["input_ids"]
        cap = max(0, (max_tokens or 0) - (safety or 0))

        boundary_char = len(prompt) + len(ans_part)

        if cap > 0 and len(ids) > cap:
            ids = ids[:cap]
            qa = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            anchor_tok_idx = len(ids) - 1
        else:
            anchor_tok_idx = _last_token_before_char(qa, boundary_char, tokenizer)

        return qa, anchor_tok_idx

    def build_hs_pack_pairwise(texts_fast, answers_fast, texts_slow, answers_slow,
                            tokenizer, max_tokens: int = MAX_TOK_HS, safety: int = SAFETY):
        """
        与论文对齐的成对构造：
        1) fast：固定“前 2 步”为初始段
        2) slow：在 {前1步, 前2步, …} 里选一个，使 len(prompt_slow+初始段) 最接近
                len(prompt_fast+前2步)（目标长度），以抵消长度/位置差异
        3) 锚点均取“初始段最后一个 token”（若触顶则末 token）
        """
        assert len(texts_fast) == len(texts_slow) == len(answers_fast) == len(answers_slow)
        N = len(texts_fast)

        qa_fast, pos_fast = [], []
        qa_slow, pos_slow = [], []

        for i in range(N):
            prompt_f, ans_f = texts_fast[i], answers_fast[i]
            prompt_s, ans_s = texts_slow[i], answers_slow[i]

            # --- fast: 保留前 2 步 ---
            ans_f_init = _answer_prefix_by_k_steps(ans_f, 2)  # 不足 2 步则用全文
            # 目标长度用“未截 cap 的理论长度”，更贴近论文的“长度匹配”定义
            target_len = _token_len_of_qa(prompt_f, ans_f_init, tokenizer)

            qa_f_i, pos_f_i = _truncate_qa_and_anchor_from_ans_part(
                prompt_f, ans_f_init, tokenizer, max_tokens, safety
            )
            qa_fast.append(qa_f_i)
            pos_fast.append(pos_f_i)

            # --- slow: 在 1..K 步中选最接近 target_len 的 k ---
            brks = _step_break_positions(ans_s)
            K = max(1, len(brks))  # 至少考虑 1 步；如果一处 '\n\n' 都没有，K=1 等价于全文
            best_k, best_gap, best_len = 1, float("inf"), None

            for k in range(1, K + 1):
                cand_init = _answer_prefix_by_k_steps(ans_s, k)
                cand_len  = _token_len_of_qa(prompt_s, cand_init, tokenizer)
                gap = abs(cand_len - target_len)
                # 规则：gap 更小优先；gap 相等取更小的 k（更短的初始段）
                if gap < best_gap or (gap == best_gap and k < best_k):
                    best_k, best_gap, best_len = k, gap, cand_len
                    best_init = cand_init

            qa_s_i, pos_s_i = _truncate_qa_and_anchor_from_ans_part(
                prompt_s, best_init, tokenizer, max_tokens, safety
            )
            qa_slow.append(qa_s_i)
            pos_slow.append(pos_s_i)

        return qa_fast, pos_fast, qa_slow, pos_slow

    # —— 用在 fast/slow 上 —— #
    qa_fast, fast_pos, qa_slow, slow_pos = build_hs_pack_pairwise(
        texts_fast, ans_fast, texts_slow, ans_slow, tokenizer,
        max_tokens=MAX_TOK_HS, safety=SAFETY
    )


    
    # 可选：看长度分布
    def _toklen(t): 
        return len(tokenizer(t, add_special_tokens=True)["input_ids"])
    fast_hs_len = np.array([_toklen(t) for t in qa_fast], dtype=np.int32)
    slow_hs_len = np.array([_toklen(t) for t in qa_slow], dtype=np.int32)
    print(f"[HS-short] fast mean={fast_hs_len.mean():.1f}, max={fast_hs_len.max()}, n={len(fast_hs_len)}")
    print(f"[HS-short] slow mean={slow_hs_len.mean():.1f}, max={slow_hs_len.max()}, n={len(slow_hs_len)}")
    
    del fast_hs_len,slow_hs_len
    
    
    
    

    
    
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
    
    capture = get_all_hidden_states_capture(llm_hs)
    
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
        BATCH = 256  # 或 4/16，视内存而定
    
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
            hs_chunk, _ = get_all_hidden_states_by_capture(llm_hs, capture, chunk, split_by_samples=True)
            for i in range(30):
                if len(chunk) != len(hs_chunk):
                    print(f"[warn] retry {i} times for batch {s}-{s+len(chunk)}")
                    hs_chunk, _ = get_all_hidden_states_by_capture(llm_hs, capture,chunk, split_by_samples=True)
            print(f"[diag] fast batch {s}-{s+len(chunk)}: want={len(chunk)} got={len(hs_chunk)}")
            collect("fast",hs_chunk, fast_pos[s:s+BATCH])
    
        # --- slow ---
        for s in range(0, len(qa_slow), BATCH):
            chunk = qa_slow[s:s+BATCH]
            hs_chunk, _ = get_all_hidden_states_by_capture(llm_hs,capture, chunk, split_by_samples=True)
            for i in range(30):
                if len(chunk) != len(hs_chunk):
                    print(f"[warn] retry {i} times for batch {s}-{s+len(chunk)}")
                    hs_chunk, _ = get_all_hidden_states_by_capture(llm_hs,capture, chunk, split_by_samples=True)
            print(f"[diag] fast batch {s}-{s+len(chunk)}: want={len(chunk)} got={len(hs_chunk)}")
            collect("slow",hs_chunk, slow_pos[s:s+BATCH])
    
    N = len(qa_fast)
    
    negative_indices = list(range(N, 2*N))    # slow
    positive_indices = list(range(0, N))   # fast
    
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
        method=pca_method,    # Use method from config
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Precompute steering vectors from dataset')
    parser.add_argument('--config', type=str, required=True, help='Path to the JSON configuration file')
    args = parser.parse_args()

    main(args.config)


