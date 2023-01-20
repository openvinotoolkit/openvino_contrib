# Question Answering evaluation using Hugging Face pipelines

This [notebook](question-answering.ipynb) provides instructions on running question-answering task evaluation for the 🤗 Transformers models which are converted using the OpenVINO™ Integration with Optimum module. <br>
There are two parts in this evaluation: 
* Evaluating the model performance based on metrics such as F-Score and Exact match. We use 🤗 [Pipelines](https://huggingface.co/docs/transformers/v4.19.2/en/main_classes/pipelines#pipelines) to evaluate the model.
* Evaluating the runtime performance (in terms of throughput and latency) using the benchmark application provided by the Intel® Distribution of OpenVINO™ Toolkit.
