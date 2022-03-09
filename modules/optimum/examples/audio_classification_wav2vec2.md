# Quantize Wav2Vec2

This guide demonstrates how to quantize a pre-trained `Wav2Vec2` model for audio classification. We also published ready to use quantized models:

| Dataset | Pretrained Model | # transformer layers | Accuracy on eval (baseline) | Accuracy on eval (quantized) | Download |
|---------|------------------|----------------------|-----------------------------|----------------------------------------|----------|
| Keyword Spotting | [facebook/wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base) | 12 | 0.9828 | 0.9553 (-0.0274) | [here](https://huggingface.co/dkurt/wav2vec2-base-ft-keyword-spotting-int8) |


1. Download the example source code from https://github.com/huggingface/transformers/tree/v4.15.0/examples/pytorch/audio-classification. Install necessary requirements.

2. [Install](../README.md#NNCF) Optimum OpenVINO and apply the following patch to enable NNCF compression:

```bash
patch -p1 < run_audio_classification.patch
```

```patch
--- a/examples/pytorch/audio-classification/run_audio_classification.py
+++ b/examples/pytorch/audio-classification/run_audio_classification.py
@@ -25,6 +25,8 @@ import datasets
 import numpy as np
 from datasets import DatasetDict, load_dataset

+from optimum.intel.nncf import NNCFAutoConfig
+
 import transformers
 from transformers import (
     AutoConfig,
@@ -52,7 +54,7 @@ def random_subsample(wav: np.ndarray, max_length: float, sample_rate: int = 1600
     """Randomly sample chunks of `max_length` seconds from the input audio"""
     sample_length = int(round(sample_rate * max_length))
     if len(wav) <= sample_length:
-        return wav
+        return np.pad(wav, (0, sample_length - len(wav)))
     random_offset = randint(0, len(wav) - sample_length - 1)
     return wav[random_offset : random_offset + sample_length]

@@ -321,6 +323,8 @@ def main():
         # Set the validation transforms
         raw_datasets["eval"].set_transform(val_transforms, output_all_columns=False)

+    nncf_config = NNCFAutoConfig.from_json(training_args.nncf_config)
+
     # Initialize our trainer
     trainer = Trainer(
         model=model,
@@ -329,6 +333,7 @@ def main():
         eval_dataset=raw_datasets["eval"] if training_args.do_eval else None,
         compute_metrics=compute_metrics,
         tokenizer=feature_extractor,
+        nncf_config=nncf_config,
     )
```

3. Run training process from pre-trained model and [NNCF config](../optimum/intel/nncf/configs/nncf_wav2vec2_config.json):

```bash
python run_audio_classification.py \
    --model_name_or_path anton-l/wav2vec2-base-ft-keyword-spotting \
    --dataset_name superb \
    --dataset_config_name ks \
    --output_dir wav2vec2-base-ft-keyword-spotting \
    --overwrite_output_dir \
    --remove_unused_columns False \
    --do_eval \
    --do_train \
    --nncf_config optimum/intel/nncf/configs/nncf_wav2vec2_config.json \
    --learning_rate 3e-5 \
    --max_length_seconds 1 \
    --attention_mask False \
    --warmup_ratio 0.1 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 4 \
    --per_device_eval_batch_size 32 \
    --dataloader_num_workers 4 \
    --logging_strategy steps \
    --logging_steps 10 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --metric_for_best_model accuracy \
    --save_total_limit 3 \
    --seed 0
```
