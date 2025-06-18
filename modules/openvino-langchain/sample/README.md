## How to run sample?

First download a sample model. You can use Optimum Intel [tool](https://github.com/huggingface/optimum-intel):

```bash
optimum-cli export openvino --trust-remote-code --model microsoft/Phi-3.5-mini-instruct Phi-3.5-mini-instruct
```

Alternatively, you can clone the repository:

```bash
git clone https://huggingface.co/OpenVINO/Phi-3.5-mini-instruct-fp16-ov
```

Then navigate to the `openvino-langchain/sample` directory and run the sample:

```bash
cd sample/
npm install
node index.js *path_to_llm_model_dir* *path_to_embeddings_model_dir*
```

The sample model `Phi-3.5-mini-instruct` should include both LLM and embeddings models.
