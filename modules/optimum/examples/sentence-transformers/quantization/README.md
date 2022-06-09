# Accuracy-aware 🤗 Transformers models training using the NNCF toolkit

We provide here an example of [Sentence-transformers](https://huggingface.co/sentence-transformers) accuracy-aware finetuning with the [Neural Network Compression Framework (NNCF)](https://github.com/openvinotoolkit/nncf) for the  STS benchmark data. The finetuned models can the converted using the OpenVINO™ Integration with Optimum module to use the optimizations provided by the Intel® Distribution of OpenVINO™ Toolkit. <br>

NNCF provides various neural networks compression optimizations such as quantization, filter pruning, sparsity and binarization. Additionally, the framework supports an [accuracy-aware finetuning](https://github.com/openvinotoolkit/nncf/blob/develop/docs/Usage.md#accuracy-aware-model-training) process to meet the accuracy constraints set by the user.

To run the accuracy aware finetuning on 🤗 Transformers models:
* Install the required dependencies:
> pip install -r requirements.txt

* Install OpenVINO™ Integration with Optimum module (with or without NNCF support):
> pip install openvino-optimum[nncf]

* Run the provided example on a device, model and parameters of choice, for example:
> python run_nncf_senttransformer_stsb.py --model_name_or_path sentence-transformers/roberta-base-nli-stsb-mean-tokens --task_name stsb --max_length 512 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --learning_rate 2e-5 --num_train_epochs 1 --output_dir stsb_out_dir_senttransformer_nncf --labels_name 'label' --nncf_config nncf_configs/nncf_roberta_config.json --device 'cpu' --acc_metric 'spearmanr'

OR

> python run_nncf_senttransformer_stsb.py --model_name_or_path sentence-transformers/roberta-base-nli-stsb-mean-tokens --task_name stsb --max_length 512 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --learning_rate 2e-5 --num_train_epochs 1 --output_dir stsb_out_dir_senttransformer_nncf --labels_name 'label' --nncf_config nncf_configs/nncf_roberta_config.json --device 'cuda' --acc_metric 'spearmanr'

The NNCF config example is provided in [nncf_configs](nncf_configs).
The optimized model is generated under `NNCF_optimized_model` by default.

The example included here is adapted from 🤗 Transformers [examples](https://github.com/huggingface/transformers/tree/main/examples) which is licensed under: <br>
**Apache License 2.0** <br>
A permissive license whose main conditions require preservation of copyright and license notices. Contributors provide an express grant of patent rights. Licensed works, modifications, and larger works may be distributed under different terms and without source code.