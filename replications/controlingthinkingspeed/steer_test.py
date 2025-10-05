import os
import datetime

from vllm.steer_vectors.request import SteerVectorRequest
from vllm import LLM, SamplingParams

model_path = "/home/bingxing2/ailab/gaoyuanyuan_p/GLM-4.1V-9B-Thinking"

# vector_path = "vectors/thinking_switch_pca_MATH-500.gguf" #MATH-500
vector_path = "./GLM_MATH500_20.gguf"

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_path, trust_remote_code=True, use_fast=True
)



from easysteer.steer import StatisticalControlVector
control_vector = StatisticalControlVector.import_gguf(vector_path)
# print(control_vector)


steer_vector_request_pos = SteerVectorRequest(
            steer_vector_name="fast",
            steer_vector_id=1,
            steer_vector_local_path=vector_path,
            scale=4.0,
            target_layers=list(range(19,38)),
            # prefill_trigger_tokens=[-1],
            # prefill_trigger_positions=[-1],
            generate_trigger_tokens=[-1],
            debug=False,
            algorithm='direct'
        )
steer_vector_request_neg = SteerVectorRequest(
            steer_vector_name="slow",
            steer_vector_id=2,
            steer_vector_local_path=vector_path,
            scale=-4.0,
            target_layers=list(range(19,38)),
            # prefill_trigger_tokens=[-1],
            # prefill_trigger_positions=[-1],
            generate_trigger_tokens=[-1],
            debug=False,
            algorithm='direct'
        )

sampling_params = SamplingParams(temperature=0.0,max_tokens=2048)
# prompt_template = "<|User|>Return your final response within \\boxed{}.\n%s<|Assistant|><think>\n"
# # prompt = "Find the constant term in the expansion of $$\\left(10x^3-\\frac{1}{2x^2}\\right)^{5}$$"

def build_glm_prompt(q: str) -> str:
    user = f"Return your final response within \\boxed{{}}. {q}"
    msgs = [{"role": "user", "content": user}]
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )


prompt = "Darrell and Allen's ages are in the ratio of 7:11. If their total age now is 162, calculate Allen's age 10 years from now."

text = build_glm_prompt(prompt)

# （可选）看一下最终提示长什么样，确认确实走了模板
print("PROMPT PREVIEW:\n", text[:300])



llm = LLM(model=model_path, enable_steer_vector=True, tensor_parallel_size=1,enforce_eager=True,    gpu_memory_utilization=0.90,
    trust_remote_code=True)

output_base = llm.generate(
    text,
    sampling_params
)
generated_text_base = output_base[0].outputs[0].text
print("Base: ", generated_text_base)
output_pos = llm.generate(
                text,
                sampling_params,
                steer_vector_request=steer_vector_request_pos
            )
generated_text_pos = output_pos[0].outputs[0].text
print("Positive: ", generated_text_pos)
output_neg = llm.generate(
                text,
                sampling_params,
                steer_vector_request=steer_vector_request_neg
            )
generated_text_neg = output_neg[0].outputs[0].text
print("Negative: ", generated_text_neg)





timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename=f'fractreason_GSM8K_{timestamp}.txt'


# filename=f'../../temp/fractreason_GSM8K_{timestamp}.txt'
with open(filename, 'w', encoding='utf-8') as f:
    f.write("=== Base ===\n")
    f.write(generated_text_base + "\n\n")
    f.write("=== Positive: Fast thinking ===\n")
    f.write(generated_text_pos + "\n\n")
    f.write("=== Negative: Slow thinking ===\n")
    f.write(generated_text_neg + "\n")
    print("cts tokens: ", len(tokenizer.tokenize(output_base[0].outputs[0].text, add_special_tokens=True)))
    print("cts tokens: ", len(tokenizer.tokenize(output_pos[0].outputs[0].text, add_special_tokens=True)))
    print("cts tokens: ", len(tokenizer.tokenize(output_neg[0].outputs[0].text, add_special_tokens=True)))