# GenAI Optimizations

This module provides experimental optimizations for GenAI models in PyTorch. The goal is to improve efficiency and performance for generative AI tasks while minimizing accuracy loss. This is PoC code and is intended to be compatible with OpenVINO GenAI.

## Supported Generative AI Scenarios

- Text Generation Using LLMs
- Visual language text generation

## Supported Generative AI Optimization Methods

- [**Visual Token Pruning**](./visual_token_pruning.py):
  Designed to accelerate inference in VLMs, where the number of input visual tokens is often significantly larger than that of textual tokens. Pruning these tokens reduces first-token latency and overall FLOPs while preserving accuracy. In this repository, we implement a visual token pruning method called [CDPruner](https://arxiv.org/pdf/2506.10967), which maximizes the conditional diversity of retained tokens. It can reduce FLOPs by 95% and CUDA latency by 78%, while maintaining 94% of the original accuracy.

- [**Sparse Attention**](./sparse_attention.py):
  Designed to accelerate the prefill stage in LLMs and MMLLMs with long prompts, high-resolution images, or videos by attending only to the most relevant query-key blocks. This block-wise attention mechanism reduces memory usage and FLOPs while preserving model accuracy. Supported modes:
  - **Tri-Shape Mode** – A static block-sparse attention pattern that preserves the initial tokens, local windows, and the final segment of the query, forming a triangular structure to capture critical tokens while maintaining instruction-following performance in both turn-0 and multi-request scenarios. Paper: https://arxiv.org/pdf/2412.10319
  - **XAttention Mode** – A dynamic block-sparse attention mechanism that accelerates inference by focusing computation on the most important regions of the attention matrix using antidiagonal block scoring, reducing FLOPs and memory usage without significant loss of accuracy. Paper: https://arxiv.org/pdf/2503.16428

## Supported and tested models

Large Language Models:

- [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
- [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
- [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- [Qwen/Qwen2-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct)
- [mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)

Multimodal Large Language Models:

- [llava-hf/llava-1.5-7b-hf](https://huggingface.co/llava-hf/llava-1.5-7b-hf)
- [llava-hf/llava-v1.6-mistral-7b-hf](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf)
- [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
- [Qwen/Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)

## Prerequisites

Before running algorithms, ensure you have **Python 3.10+** installed and set up your environment.

### 1. Create and activate a virtual environment

```bash
python3 -m venv env
source env/bin/activate      # On Windows: env\Scripts\activate.bat
```

### 2. Installation

You can install the package directly from the repository. To avoid running out of memory during the build, you can limit threads with `MAX_JOBS=4`:

```bash
pip install git+https://github.com/openvinotoolkit/openvino_contrib.git#egg=genai_opt&subdirectory=modules/genai_optimizations
```

Or install it locally with extra dependencies for benchmarks support:

```bash
pip install .[benchmarks]
```
