import os
import datetime
import json
import numpy as np

from vllm.steer_vectors.request import SteerVectorRequest
from vllm import LLM, SamplingParams

model_path = "/home/bingxing2/ailab/gaoyuanyuan_p/GLM-4.1V-9B-Thinking"

os.environ["VLLM_USE_V1"] = "0"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# vector_path = "vectors/thinking_switch_pca_MATH-500.gguf" #MATH-500
vector_path = "/home/bingxing2/ailab/gaoyuanyuan_p/yuning/EasySteer/replications/controlingthinkingspeed/GLM_MATH500_fix_center_with_alright_256_fixed/used.gguf"

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_path, trust_remote_code=True, use_fast=True
)



from easysteer.steer import StatisticalControlVector
control_vector = StatisticalControlVector.import_gguf(vector_path)
# print(control_vector)


sampling_params = SamplingParams(temperature=0.0,max_tokens=4096)

# Define test ranges
TEST_LAYERS = list(range(18, 38,10))  # Layers 20-39
TEST_SCALES = [ 2.0, 4.0,  -2.0, -4.0, ]

# Configuration
NUM_QUESTIONS = 1  # Number of questions to test
JSONL_FILE_PATH = "/home/bingxing2/ailab/gaoyuanyuan_p/yuning/temp/test.jsonl"

def build_glm_prompt(q: str) -> str:
    user = f"Return your final response within \\boxed{{}}. {q}"
    msgs = [{"role": "user", "content": user}]
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )


# Load questions from JSONL file
def load_questions(file_path, num_questions):
    questions = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_questions:
                break
            data = json.loads(line)
            questions.append(data['question'])
    return questions

print(f"Loading {NUM_QUESTIONS} questions from {JSONL_FILE_PATH}...")
questions = load_questions(JSONL_FILE_PATH, NUM_QUESTIONS)
print(f"Loaded {len(questions)} questions\n")



llm = LLM(model=model_path, enable_steer_vector=True, tensor_parallel_size=1,enforce_eager=False,    gpu_memory_utilization=0.90,
    trust_remote_code=True)

# Store results: {(layer, scale): [lengths for each question]}
results_by_config = {}

# Generate baseline for all questions
print("=" * 80)
print("GENERATING BASELINE (no steering)...")
print("=" * 80)
baseline_lengths = []

# for i, question in enumerate(questions):
#     text = build_glm_prompt(question)
#     output = llm.generate(text, sampling_params)
#     generated_text = output[0].outputs[0].text
#     length = len(tokenizer.tokenize(generated_text, add_special_tokens=True))
#     baseline_lengths.append(length)
#     print(f"Question {i+1}/{len(questions)}: {length} tokens")

results_by_config[('baseline', 0.0)] = baseline_lengths
print(f"\nBaseline average: {np.mean(baseline_lengths):.1f} ± {np.std(baseline_lengths):.1f} tokens\n")

# Test different layers and scales
total_configs = len(TEST_LAYERS) * len(TEST_SCALES)
current_config = 0

for layer in TEST_LAYERS:
    for scale in TEST_SCALES:
        current_config += 1
        config_key = (layer, scale)
        config_lengths = []

        print("=" * 80)
        print(f"Config {current_config}/{total_configs}: Layer {layer}, Scale {scale}")
        print("=" * 80)

        steer_vector_request = SteerVectorRequest(
            steer_vector_name=f"layer{layer}_scale{scale}",
            steer_vector_id=current_config,
            steer_vector_local_path=vector_path,
            scale=scale,
            target_layers=list(range(layer,38)),
            generate_trigger_tokens=[-1],
            debug=False,
            algorithm='direct'
        )

        for i, question in enumerate(questions):
            text = build_glm_prompt(question)
            output = llm.generate(text, sampling_params, steer_vector_request=steer_vector_request)
            generated_text = output[0].outputs[0].text
            length = len(tokenizer.tokenize(generated_text, add_special_tokens=True))
            config_lengths.append(length)
            print(f"Question {i+1}/{len(questions)}: {length} tokens")

        results_by_config[config_key] = config_lengths
        avg_length = np.mean(config_lengths)
        std_length = np.std(config_lengths)
        print(f"\nAverage for this config: {avg_length:.1f} ± {std_length:.1f} tokens\n")

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
summary_filename = f'multi_question_summary_{timestamp}.txt'

# Write summary results (averages only, no text content)
print("=" * 80)
print("Writing summary file...")
print("=" * 80)

with open(summary_filename, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("MULTI-QUESTION SUMMARY: Average Output Lengths by Layer and Scale\n")
    f.write("=" * 80 + "\n\n")

    f.write(f"Number of questions tested: {NUM_QUESTIONS}\n")
    f.write(f"Source file: {JSONL_FILE_PATH}\n")
    f.write(f"Tested layers: {TEST_LAYERS}\n")
    f.write(f"Tested scales: {TEST_SCALES}\n\n")

    # Baseline statistics
    baseline_avg = np.mean(baseline_lengths)
    baseline_std = np.std(baseline_lengths)
    baseline_min = np.min(baseline_lengths)
    baseline_max = np.max(baseline_lengths)

    f.write("BASELINE (no steering):\n")
    f.write(f"  Average: {baseline_avg:.1f} tokens\n")
    f.write(f"  Std Dev: {baseline_std:.1f} tokens\n")
    f.write(f"  Min: {baseline_min} tokens\n")
    f.write(f"  Max: {baseline_max} tokens\n")
    f.write(f"  Individual lengths: {baseline_lengths}\n\n")

    # Main results table
    f.write("=" * 80 + "\n")
    f.write(f"{'Layer':<10}{'Scale':<10}{'Avg Length':<15}{'Std Dev':<15}{'Min':<10}{'Max':<10}\n")
    f.write("=" * 80 + "\n")

    # Sort results by layer then scale for better readability
    sorted_configs = sorted([(k, v) for k, v in results_by_config.items() if k[0] != 'baseline'],
                           key=lambda x: (x[0][0], x[0][1]))

    for (layer, scale), lengths in sorted_configs:
        avg = np.mean(lengths)
        std = np.std(lengths)
        min_len = np.min(lengths)
        max_len = np.max(lengths)
        f.write(f"{str(layer):<10}{scale:<10.1f}{avg:<15.1f}{std:<15.1f}{min_len:<10}{max_len:<10}\n")

    f.write("=" * 80 + "\n\n")

    # Statistics by scale
    f.write("AVERAGE OUTPUT LENGTH BY SCALE (across all layers and questions):\n")
    f.write("-" * 80 + "\n")
    for scale in sorted(set(TEST_SCALES)):
        scale_lengths = []
        for (layer, s), lengths in results_by_config.items():
            if s == scale and layer != 'baseline':
                scale_lengths.extend(lengths)
        if scale_lengths:
            avg = np.mean(scale_lengths)
            std = np.std(scale_lengths)
            f.write(f"  Scale {scale:>6.1f}: {avg:>7.1f} ± {std:<6.1f} tokens (n={len(scale_lengths)} samples)\n")

    f.write("\n")

    # Statistics by layer
    f.write("AVERAGE OUTPUT LENGTH BY LAYER (across all scales and questions):\n")
    f.write("-" * 80 + "\n")
    for layer in sorted(TEST_LAYERS):
        layer_lengths = []
        for (l, scale), lengths in results_by_config.items():
            if l == layer:
                layer_lengths.extend(lengths)
        if layer_lengths:
            avg = np.mean(layer_lengths)
            std = np.std(layer_lengths)
            f.write(f"  Layer {layer:>2}: {avg:>7.1f} ± {std:<6.1f} tokens (n={len(layer_lengths)} samples)\n")

    f.write("\n")

    # Comparison to baseline
    f.write("RELATIVE CHANGE FROM BASELINE:\n")
    f.write("-" * 80 + "\n")
    f.write(f"{'Layer':<10}{'Scale':<10}{'Avg Change':<20}{'% Change':<15}\n")
    f.write("-" * 80 + "\n")

    for (layer, scale), lengths in sorted_configs:
        avg = np.mean(lengths)
        change = avg - baseline_avg
        pct_change = (change / baseline_avg) * 100
        f.write(f"{str(layer):<10}{scale:<10.1f}{change:+.1f} tokens{' '*8}{pct_change:+.1f}%{' '*5}\n")

print(f"\nSummary written to: {summary_filename}")
print(f"Total configurations tested: {len(results_by_config)}")
print(f"Total inference runs: {len(results_by_config) * NUM_QUESTIONS}")
print("\n✓ All tests completed!")