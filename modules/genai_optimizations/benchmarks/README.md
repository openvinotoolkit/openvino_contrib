# Generative AI Models Optimization Examples

This folder provides examples for evaluating and optimizing Generative AI models across different scenarios.


<details>
<summary><b>Multimodal Large Language Models Optimization Example: MME Benchmark</b></summary>

This [example](./mmebench.py) demonstrates how to evaluate and optimize MLLMs using the [MME benchmark](https://arxiv.org/pdf/2306.13394), which measures both perception and cognition abilities across 14 subtasks. Its concise instruction design enables fair comparison of MLLMs without the need for extensive prompt engineering.

Visual token pruning enables significant acceleration of inference in VLMs, where the number of input visual tokens is often much larger than the number of textual tokens. By pruning these tokens, we reduce first-token latency and overall FLOPs while preserving accuracy.

### Run Example

```bash
python mmebench.py \
    --subset artwork \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --enable_visual_pruning \
    --num_keep_tokens 128 \
    --theta 0.5
```
This will automatically:

- Download the selected model and dataset
- Apply the visual token pruning algorithm
- Evaluate the model and report the score

</details>

<details>
<summary><b>Multimodal Large Language Models Optimization Example: MileBench</b></summary>

This [example](./milebench.py) demonstrates how to optimize MLLMs using an experimental visual token pruning algorithm. The example leverages [MileBench](https://arxiv.org/pdf/2404.18532), a pioneering benchmark designed to rigorously evaluate the multimodal long-context capabilities of MLLMs. MileBench encompasses diverse tasks requiring both comprehension and generation, and introduces two distinct evaluation sets— diagnostic and realistic — that systematically assess models’ capacity for long-context adaptation and effective task completion.


### Run Example

```bash
python milebench.py \
    --subset WikiVQA \
    --model Qwen/Qwen2-VL-2B-Instruct \
    --enable_visual_pruning \
    --num_keep_tokens 64 \
    --theta 0.5
```

This will automatically:

- Download the selected model and dataset
- Apply the visual token pruning algorithm
- Evaluate the model and report the score

</details>
