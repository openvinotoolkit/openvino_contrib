# OpenVINO™ LangChain.js adapter

This package contains the LangChain.js integrations for OpenVINO™

> **Disclaimer**
> It's preview version, do not use it on production!

## Introduction

OpenVINO is an open-source toolkit for deploying performant AI solutions. Convert, optimize, and run inference on local hardware utilizing the full potential of Intel® hardware.

## Installation and Setup

See [this section](https://js.langchain.com/docs/how_to/installation#installing-integration-packages) for general instructions on installing integration packages.

```bash
npm install openvino-langchain
```

### Export your model to the OpenVINO™ IR

In order to use OpenVINO, you need to convert and compress the text generation model into the [OpenVINO IR format](https://docs.openvino.ai/2025/documentation/openvino-ir-format.html).

The following models are tested:
  * Embeddings models:
    - BAAI/bge-small-en-v1.5
    - intfloat/multilingual-e5-large
    - sentence-transformers/all-MiniLM-L12-v2
    - sentence-transformers/all-mpnet-base-v2
  * Large language models:  
    - openlm-research/open_llama_7b_v2
    - meta-llama/Llama-2-13b-chat-hf
    - microsoft/Phi-3.5-mini-instruct
    - deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

#### Use HuggingFace Hub

Pre-converted and pre-optimized models are available under the [LLM collections](https://huggingface.co/collections/OpenVINO/llm-6687aaa2abca3bbcec71a9bd) in the [OpenVINO Toolkit](https://huggingface.co/OpenVINO) organization.

To export another [model](https://huggingface.co/docs/optimum/main/en/intel/openvino/models) hosted on the [HuggingFace Hub](https://huggingface.co/models) you can use [OpenVINO space](https://huggingface.co/spaces/echarlaix/openvino-export). After conversion, a repository will be pushed under your namespace, this repository can be either public or private.

#### Use the Optimum Intel

[Optimum Intel](https://github.com/huggingface/optimum-intel) provides a simple interface to optimize your Transformers and Diffusers models and convert them to the OpenVINO Intermediate Representation (IR) format.

Firstly install Optimum Intel for OpenVINO:

```bash
pip install --upgrade --upgrade-strategy eager "optimum[openvino]"
```

Then you download and convert a model to OpenVINO:

```bash
optimum-cli export openvino --model <model_id> --trust-remote-code <exported_model_name>
```
> **Note:** Any model_id, for example "TinyLlama/TinyLlama-1.1B-Chat-v1.0", or the path to a local model file can be used.

Optimum-Intel API also provides out-of-the-box model optimization through weight compression using NNCF which substantially reduces the model footprint and inference latency:

```bash
optimum-cli export openvino --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --weight-format int4 --trust-remote-code "TinyLlama-1.1B-Chat-v1.0"
```

## LLM

This package contains the `GenAI` class, which is the recommended way to interact with models optimized for the OpenVINO toolkit.

**GenAI Parameters**

| Name  | Type | Required | Description |
| ----- | ---- |--------- | ----------- |
| modelPath | string | ✅ | Path to the directory containing model xml/bin files and tokenizer |
| device | string | ❌ | Device to run the model on (e.g., CPU, GPU). |
| generationConfig | [GenerationConfig](https://github.com/openvinotoolkit/openvino.genai/blob/master/src/js/lib/utils.ts#L107-L110) | ❌ | Structure to keep generation config parameters. |

```typescript
import { GenAI } from "openvino-langchain";

const model = new GenAI({
    modelPath: "path-to-model",
    device: "CPU",
    generationConfig: {
        "max_new_tokens": 100,
    },
  });
const response = await model.invoke("Hello, world!");
```

## Text Embedding Model

This package also adds support for OpenVINO's embeddings model.

| Name  | Type | Required | Description |
| ----- | ---- |--------- | ----------- |
| modelPath | string | ✅ | Path to the directory containing embeddings model |
| device | string | ❌ | Device to run the embeddings model on (e.g., CPU, GPU). |

```typescript
import { OvEmbeddings } from "openvino-langchain";

const embeddings = new OvEmbeddings({
    modelPath: "path-to-model",
    device: "CPU",
});
const res = await embeddings.embedQuery("Hello world");
```
