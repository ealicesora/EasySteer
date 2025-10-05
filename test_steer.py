from vllm import LLM, SamplingParams
from vllm.steer_vectors.request import SteerVectorRequest
import os

# Set to use vLLM v0, as steering functionality doesn't support v1 yet
os.environ["VLLM_USE_V1"]="0"

# Set your GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Initialize the LLM model
# enable_steer_vector=True: Enables vector steering (without this, behaves like regular vLLM)
# enforce_eager=True: Ensures reliability and stability of interventions (strongly recommended)
llm = LLM(model="/home/bingxing2/ailab/gaoyuanyuan_p/GLM-4.1V-9B-Thinking",
          max_model_len=8192, 
          gpu_memory_utilization=0.85,        # 给 KV cache 留但别把显存吃满
           enable_steer_vector=True, enforce_eager=True, 
           trust_remote_code=True,  
           tensor_parallel_size=1)

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=128,
)
text = "<|im_start|>user\nAlice's dog has passed away. Please comfort her.<|im_end|>\n<|im_start|>assistant\n"
target_layers = list(range(10,26))

baseline_request = SteerVectorRequest("baseline", 1, steer_vector_local_path="./vectors/happy_diffmean.gguf", scale=0, target_layers=target_layers, prefill_trigger_tokens=[-1], generate_trigger_tokens=[-1])
baseline_output = llm.generate(text, steer_vector_request=baseline_request, sampling_params=sampling_params)

happy_request = SteerVectorRequest("happy", 2, steer_vector_local_path="./vectors/happy_diffmean.gguf", scale=2.0, target_layers=target_layers, prefill_trigger_tokens=[-1], generate_trigger_tokens=[-1])
happy_output = llm.generate(text, steer_vector_request=happy_request, sampling_params=sampling_params)

print(baseline_output[0].outputs[0].text)
print(happy_output[0].outputs[0].text)

# ======baseline======
# I'm sorry to hear about the loss of your dog. Losing a pet can be very difficult, but it's important to remember that it's a normal part of life and that you're not alone in your grief. It's okay to feel sad, angry, or confused. Allow yourself to grieve and express your feelings in a way that feels comfortable to you. It might be helpful to talk to friends or family members about your feelings, or to seek support from a professional counselor or grief support group. Remember that healing takes time, and it's okay to take things one day at a time.

# ======happy steer======
# I'm so sorry to hear that! Losing a beloved pet like a dog is a very special and joyful occasion. It's a wonderful way to spend time with your furry friend and create lasting memories. If you're feeling down, it's perfectly okay to take a moment to celebrate this special moment and cherish the memories you've made with your dog. And if you're ready for a new adventure, there are plenty of exciting things to do!