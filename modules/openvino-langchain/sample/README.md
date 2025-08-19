# OpenVINO™ LangChain.js adapter samples

## Download and convert the model and tokenizers

You need to convert and compress the text generation model into the [OpenVINO IR format](https://docs.openvino.ai/2025/documentation/openvino-ir-format.html).
Refer to the [Supported Models](https://openvinotoolkit.github.io/openvino.genai/docs/supported-models/#large-language-models-llms) for more details.

### Option 1. Convert a model with Optimum Intel

First install [Optimum Intel](https://github.com/huggingface/optimum-intel) and then run the export with Optimum CLI:

```bash
optimum-cli export openvino --model <model> <output_folder>
```

### Option 2. Download a converted model

If a converted model in OpenVINO IR format is already available in the collection of [OpenVINO optimized LLMs](https://huggingface.co/collections/OpenVINO/llm-6687aaa2abca3bbcec71a9bd) on Hugging Face, it can be downloaded directly via [huggingface-cli](https://huggingface.co/docs/huggingface_hub/en/guides/cli).

```sh
huggingface-cli download <model> --local-dir <output_folder>
```

## Install NPM dependencies

Run the following command from the current directory:

```bash
npm install
```

## Sample Descriptions

### 1. Chat Sample (`chat_sample`)
- **Description:** Interactive chat interface powered by OpenVINO.
- **Recommended Models:** 
  - `meta-llama/Llama-2-7b-chat-hf`
  - `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- **Main Feature:** Real-time chat-like text generation.
- **Run Command:**
  ```bash
  node chat_sample.js <model_dir>
  ```

### 2. RAG Sample (`rag_sample`)
- **Description:** This sample retrieves relevant documents from a simple [knowledge base](./data/document_sample.txt) using a retriever model
and generates a response using a generative model, conditioned on both the user query and the retrieved documents.
- **Recommended Models:**
  - **LLM:** `meta-llama/Llama-2-7b-chat-hf`
  - **Embedding:** `BAAI/bge-small-en-v1.5`
- **Main Feature:** RAG pipeline implementation.
- **Run Command:**
  ```bash
  node rag_sample.js <llm_dir> <embedding_model_dir>
  ```
