# SPDX-License-Identifier: Apache-2.0
# Simple Qwen3-0.6B inference script using local vLLM code
#
# Run with: PYTHONPATH=. python test_qwen3_inference.py

from vllm import LLM, SamplingParams


def main():
    # Load Qwen3-0.6B model
    llm = LLM(model="Qwen/Qwen3-0.6B")

    # Sampling params (recommended for Qwen3 non-thinking mode)
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.8,
        max_tokens=256
    )

    # Simple prompt
    prompt = "Hello, my name is"

    # Generate
    outputs = llm.generate([prompt], sampling_params)

    # Print results
    print("\n" + "=" * 60)
    print("Generated Output:")
    print("=" * 60)
    for output in outputs:
        print(f"Prompt: {output.prompt!r}")
        print(f"Generated: {output.outputs[0].text!r}")
    print("=" * 60)


if __name__ == "__main__":
    main()

