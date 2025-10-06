import os
import datetime
import json
import argparse
from pathlib import Path

from vllm.steer_vectors.request import SteerVectorRequest
from vllm import LLM, SamplingParams

os.environ["VLLM_USE_V1"] = "0"


def load_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_config(config, save_dir):
    """Save configuration to the output directory."""
    config_save_path = os.path.join(save_dir, 'config.json')
    with open(config_save_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"Configuration saved to: {config_save_path}")


def main(config_path):
    # Load configuration
    config = load_config(config_path)

    # Extract configuration parameters
    base_path = config['base_path']
    model_path = config['model_path']
    vector_path = base_path + config['vector_path']
    save_dir =  base_path + config['save_dir']
    TEST_LAYERS = config['test_layers']
    TEST_SCALES = config['test_scales']
    use_prefill_trigger = config.get('use_prefill_trigger', False)
    prompt = config.get('prompt', "Darrell and Allen's ages are in the ratio of 7:11. If their total age now is 162, calculate Allen's age 10 years from now.")

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Save config to output directory
    save_config(config, save_dir)

    # Initialize tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, use_fast=True
    )

    # Load control vector
    from easysteer.steer import StatisticalControlVector
    control_vector = StatisticalControlVector.import_gguf(vector_path)

    # Sampling parameters
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1024)

    def build_glm_prompt(q: str) -> str:
        user = f"Return your final response within \\boxed{{}}. {q}"
        msgs = [{"role": "user", "content": user}]
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )

    text = build_glm_prompt(prompt)

    # Print prompt preview
    print("PROMPT PREVIEW:\n", text[:300])

    # Initialize LLM
    llm = LLM(model=model_path, enable_steer_vector=True, tensor_parallel_size=1,
              enforce_eager=True, gpu_memory_utilization=0.90, trust_remote_code=True)

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

            # Build SteerVectorRequest parameters
            steer_params = {
                'steer_vector_name': f"layer{layer}_scale{scale}",
                'steer_vector_id': current_test,
                'steer_vector_local_path': vector_path,
                'scale': scale,
                'target_layers': list(range(layer, 38)),
                'generate_trigger_tokens': [-1],
                'debug': False,
                'algorithm': 'direct'
            }

            # Add prefill trigger if configured
            if use_prefill_trigger:
                steer_params['prefill_trigger_tokens'] = [-1]
                steer_params['prefill_trigger_positions'] = [-1]

            steer_vector_request = SteerVectorRequest(**steer_params)

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

    # Generate filenames with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_filename = os.path.join(save_dir, f'layer_scale_summary_{timestamp}.txt')
    detailed_filename = os.path.join(save_dir, f'layer_scale_detailed_{timestamp}.txt')

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run steering vector tests with configurable parameters')
    parser.add_argument('--config', type=str, required=True, help='Path to the JSON configuration file')
    args = parser.parse_args()

    main(args.config)