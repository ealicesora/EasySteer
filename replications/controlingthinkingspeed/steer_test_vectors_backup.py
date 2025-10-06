import os
import datetime

from vllm.steer_vectors.request import SteerVectorRequest
from vllm import LLM, SamplingParams

model_path = "/home/bingxing2/ailab/gaoyuanyuan_p/GLM-4.1V-9B-Thinking"

os.environ["VLLM_USE_V1"] = "0"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# vector_path = "vectors/thinking_switch_pca_MATH-500.gguf" #MATH-500
vector_path = "/home/bingxing2/ailab/gaoyuanyuan_p/yuning/EasySteer/replications/controlingthinkingspeed/GLM_MATH500_fix_center_alright_256.gguf"

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_path, trust_remote_code=True, use_fast=True
)



from easysteer.steer import StatisticalControlVector
control_vector = StatisticalControlVector.import_gguf(vector_path)
# print(control_vector)


sampling_params = SamplingParams(temperature=0.0,max_tokens=1024)

# Define test ranges
TEST_LAYERS = list(range(18, 38,30))  # Layers 20-39
TEST_SCALES = [ 2.0, 4.0, -2.0, -4.0]
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

# Generate baseline (no steering)
print("Generating baseline...")
output_base = llm.generate(text, sampling_params)
generated_text_base = output_base[0].outputs[0].text
base_length = len(tokenizer.tokenize(generated_text_base, add_special_tokens=True))
print(f"Base output length: {base_length} tokens")

# Store results
results = []
results.append({
    'layer': 'baseline',
    'scale': 0.0,
    'output_length': base_length,
    'text': generated_text_base
})

# Test different layers and scales
total_tests = len(TEST_LAYERS) * len(TEST_SCALES)
current_test = 0

for layer in TEST_LAYERS:
    for scale in TEST_SCALES:
        current_test += 1
        print(f"\nTest {current_test}/{total_tests}: Layer {layer}, Scale {scale}")

        steer_vector_request = SteerVectorRequest(
            steer_vector_name=f"layer{layer}_scale{scale}",
            steer_vector_id=current_test,
            steer_vector_local_path=vector_path,
            scale=scale ,
            target_layers=list(range(layer,38)),
            # prefill_trigger_tokens=[-1],
            # prefill_trigger_positions=[-1],
            generate_trigger_tokens=[-1],
            debug=False,
            algorithm='direct'
        )
        print(SteerVectorRequest)
        output = llm.generate(text, sampling_params, steer_vector_request=steer_vector_request)
        generated_text = output[0].outputs[0].text
        output_length = len(tokenizer.tokenize(generated_text, add_special_tokens=True))

        print(f"Output length: {output_length} tokens")
        print("Generated text preview:\n", generated_text)
        results.append({
            'layer': layer,
            'scale': scale,
            'output_length': output_length,
            'text': generated_text
        })

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
summary_filename = f'layer_scale_summary_{timestamp}.txt'
detailed_filename = f'layer_scale_detailed_{timestamp}.txt'

# Write summary results
with open(summary_filename, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("SUMMARY: Output Length by Layer and Scale\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Prompt: {prompt}\n\n")
    f.write(f"Total tests: {len(results)}\n")
    f.write(f"Tested layers: {TEST_LAYERS[0]}-{TEST_LAYERS[-1]}\n")
    f.write(f"Tested scales: {TEST_SCALES}\n\n")

    f.write("-" * 80 + "\n")
    f.write(f"{'Layer':<10}{'Scale':<10}{'Output Length (tokens)':<25}\n")
    f.write("-" * 80 + "\n")

    for result in results:
        layer = result['layer']
        scale = result['scale']
        length = result['output_length']
        f.write(f"{str(layer):<10}{scale:<10.1f}{length:<25}\n")

    f.write("-" * 80 + "\n\n")

    # Add statistics
    f.write("STATISTICS:\n")
    f.write(f"Baseline output length: {base_length} tokens\n\n")

    # Group by scale
    f.write("Average output length by scale:\n")
    for scale in TEST_SCALES:
        scale_results = [r for r in results if r['scale'] == scale]
        if scale_results:
            avg_length = sum(r['output_length'] for r in scale_results) / len(scale_results)
            f.write(f"  Scale {scale:>5.1f}: {avg_length:.1f} tokens (avg)\n")

    f.write("\n")

    # Group by layer
    f.write("Average output length by layer:\n")
    for layer in TEST_LAYERS:
        layer_results = [r for r in results if r['layer'] == layer]
        if layer_results:
            avg_length = sum(r['output_length'] for r in layer_results) / len(layer_results)
            f.write(f"  Layer {layer:>2}: {avg_length:.1f} tokens (avg)\n")

print(f"\nSummary written to: {summary_filename}")

# Write detailed results with full text outputs
with open(detailed_filename, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("DETAILED RESULTS: Full Outputs by Layer and Scale\n")
    f.write("=" * 80 + "\n\n")

    for result in results:
        f.write("=" * 80 + "\n")
        f.write(f"Layer: {result['layer']} | Scale: {result['scale']} | Length: {result['output_length']} tokens\n")
        f.write("=" * 80 + "\n")
        f.write(result['text'])
        f.write("\n\n")

print(f"Detailed results written to: {detailed_filename}")
print(f"\nAll tests completed! Total: {len(results)} runs")