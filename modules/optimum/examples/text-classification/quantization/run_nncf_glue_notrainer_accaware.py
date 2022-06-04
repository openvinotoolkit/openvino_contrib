#!/usr/bin/env python
# coding=utf-8
"""
Fine-tuning with Intel NNCF toolkit for the question answering task using 🤗 Accelerate.
Uses HuggingFace accelarate - https://github.com/huggingface/accelerate
References: https://github.com/huggingface/transformers/blob/main/examples/pytorch/question-answering/run_qa_beam_search_no_trainer.py
"""
import argparse
import json
import logging
import math
import os
import random
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader

import transformers
from transformers import (
    AutoModelForSequenceClassification,
    PretrainedConfig,
    DataCollatorWithPadding,
    SchedulerType,
    AutoConfig,
    AutoTokenizer,
    default_data_collator,
    get_scheduler,
    AdamW,
)
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

import datasets
from datasets import load_dataset, load_metric

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from nncf import NNCFConfig
from nncf.torch import create_compressed_model, register_default_init_args
from nncf.common.accuracy_aware_training import create_accuracy_aware_training_loop

from glue_nncf_utils import NNCFGLUEDataLoader

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
#check_min_version("4.20.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r requirements.txt")

logger = get_logger(__name__)

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a Question Answering task")
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="The name of the glue task to train on.",
        choices=list(task_to_keys.keys()),
    )
    parser.add_argument(
        "--do_train", type=bool, default=True, help="Whether to run training."
    )
    parser.add_argument(
        "--do_eval", type=bool, default=True, help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--test_file", type=str, default=None, help="A csv or a json file containing the Prediction data."
    )
    parser.add_argument(
        "--preprocessing_num_workers", type=int, default=4, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_seq_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cuda',
        help="Device to use for the finetuning",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--optimized_model_dir", type=str, default='NNCF_optimized_model', help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--doc_stride",
        type=int,
        default=128,
        help="When splitting up a long document into chunks how much stride to take between chunks.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--acc_metric",
        type=str,
        default='glue',
        help="The evaluation metric to use for the NNCF finetuning",
    )
    parser.add_argument(
        "--overwrite_cache", type=bool, default=False, help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        help="Whether or not to enable to load a pretrained model whose head dimensions are different.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to load in all available experiment trackers from the environment and use them for logging.",
    )
    parser.add_argument(
        "--nncf_config",
        type=str,
        default='nncf_configs/nncf_glue_config.json',
        help="A configuration .json file used for NNCF enabled compression optimized training.",
    )
    parser.add_argument(
        "--opset_version",
        type=int,
        default=11,
        help=(
            "The opset version used to export the NNCF compressed model to ONNX format"
        ),
    )
    args = parser.parse_args()

    # Sanity checks
    if (
        args.task_name is None 
        and args.train_file is None
        and args.validation_file is None
        and args.test_file is None
    ):
        raise ValueError("Need either a dataset name or a training/validation/test file.")

    return args

def load_raw_dataset(args):
    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset('glue', args.task_name)
    else:
        data_files = {}
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
            data_files["validation"] = args.validation_file
        extension = args.train_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files, field="data")
    
    # Labels
    label_list = []
    if args.task_name is not None:
        is_regression = args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)
    return raw_datasets, label_list, num_labels, is_regression

def main():
    args = parse_args()

    # Intialize the accelarator 
    accelerator = Accelerator(log_with="all", logging_dir=args.output_dir) if args.with_tracking else Accelerator()
    
    # Logging for debugging purposes 
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    accelerator.wait_for_everyone()

    # Load the pretrained model, tokenizer as well as the model config 
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        ignore_mismatched_sizes=args.ignore_mismatched_sizes,
    )


    # Load the datasets: Either from HuggingFace datasets 
    # or from the provided train and test csv files and
    # validate that the extension of these files 
    raw_datasets, label_list, num_labels, is_regression  = load_raw_dataset(args)

    # Load the metric for evaluation 
    # Get the metric function
    if args.task_name is not None:
        metric = load_metric("glue", args.task_name)
    else:
        metric = load_metric("accuracy")

    # Preprocessing the datasets.
    # Preprocessing is slighlty different for training and evaluation.
    
    if "train" not in raw_datasets:
        raise ValueError("--do_train requires a train dataset")
    if "validation" not in raw_datasets:
        if "validation_matched" in raw_datasets:
            pass
        else:
            raise ValueError("--do_eval requires a validation dataset")
    
    # Preprocessing the datasets
    if args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            logger.info(
                f"The configuration of the model provided the following label correspondence: {label_name_to_id}. "
                "Using it!"
            )
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True)

        if "label" in examples:
            if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples["label"]
        return result
       
    # Create train feature from dataset
    with accelerator.main_process_first():
        preprocessed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=raw_datasets["train"].column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on train dataset",
        )

    train_dataset = preprocessed_datasets["train"]
    eval_dataset = preprocessed_datasets["validation_matched" if args.task_name == "mnli" else "validation"]
    
    
    if args.max_train_samples is not None:
        # We will select sample from whole data
        train_dataset = train_dataset.select(range(args.max_train_samples))
    
    if args.max_eval_samples is not None:
        # We will select sample from whole data
        eval_examples = eval_examples.select(range(args.max_eval_samples))

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )

    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # NNCF suitable train dataloader
    train_dataloader = NNCFGLUEDataLoader(train_dataloader)

    # NNCF suitable eval dataloader
    eval_dataloader = NNCFGLUEDataLoader(eval_dataloader)
    
    nncf_config = NNCFConfig.from_json(args.nncf_config) 

    optimized_model_dir = args.optimized_model_dir
    if optimized_model_dir is None:
        optimized_model_dir = 'NNCF_compressed_model'
        if os.path.isdir(optimized_model_dir):
            logger.warning(
                f" The existing NNCF compression optimized models will be replaced"
            )
        else:
            os.mkdir(optimized_model_dir)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch =  args.max_train_steps if args.max_train_steps else math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    def configure_optimizers_fn():
        # Optimizer
        # Split weights in two groups, one with weight decay and the other not.
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=args.max_train_steps,
        )
        return optimizer, lr_scheduler

    optimizer, lr_scheduler = configure_optimizers_fn() 

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    def train_epoch_fn(compression_ctrl, compressed_model, epoch=None, optimizer=optimizer, lr_scheduler=lr_scheduler):
        total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

        # Figure out how many steps we should save the Accelerator states
        if hasattr(args.checkpointing_steps, "isdigit"):
            checkpointing_steps = args.checkpointing_steps
            if args.checkpointing_steps.isdigit():
                checkpointing_steps = int(args.checkpointing_steps)
        else:
            checkpointing_steps = None

        # We need to initialize the trackers we use, and also store our configuration
        if args.with_tracking:
            experiment_config = vars(args)
            # TensorBoard cannot log Enums, need the raw value
            experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
            accelerator.init_trackers("qa_beam_search_no_trainer", experiment_config)

        logger.info("\n***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataloader)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")

        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
        completed_steps = 0
        starting_epoch = 0

        # Potentially load in the weights and states from a previous save
        if args.resume_from_checkpoint:
            if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
                accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
                accelerator.load_state(args.resume_from_checkpoint)
                path = os.path.basename(args.resume_from_checkpoint)
            else:
                # Get the most recent checkpoint
                dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
                dirs.sort(key=os.path.getctime)
                path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            # Extract `epoch_{i}` or `step_{i}`
            training_difference = os.path.splitext(path)[0]

            if "epoch" in training_difference:
                starting_epoch = int(training_difference.replace("epoch_", "")) + 1
                resume_step = None
            else:
                resume_step = int(training_difference.replace("step_", ""))
                starting_epoch = resume_step // len(train_dataloader)
                resume_step -= starting_epoch * len(train_dataloader)

        # We need to track the total loss for NNCF finetuning
        # We also do a single epoch training since we're running it
        # under NNCF training loop
        total_loss = 0.0
        compressed_model.train()
        if args.with_tracking:
            total_loss = 0
        for step, batch in enumerate(train_dataloader):
            # We need to skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == starting_epoch:
                if resume_step is not None and step < resume_step:
                    completed_steps += 1
                    continue
            compression_ctrl.scheduler.step()
            batch = batch.to(args.device)
            outputs = compressed_model(**batch)
            loss = outputs.loss
            compression_loss = compression_ctrl.loss()
            loss += compression_loss
            # We keep track of the loss at each epoch
            if args.with_tracking:
                total_loss += loss.detach().float()
            total_loss += loss.detach().float()
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    accelerator.save_state(f"step_{completed_steps}")

            if completed_steps >= args.max_train_steps:
                break
        train_metric = {}
        acc_metric = "accuracy" if args.task_name is None else args.acc_metric
        train_metric[acc_metric] = validate_fn(compressed_model)

        logger.info(f"Total loss: {total_loss}, Eval metric: {train_metric}")
        return total_loss, train_metric


    def validate_fn(model, epoch=None):
        logger.info("\n***** Running Evaluation *****")
        logger.info(f"  Num examples = {len(eval_dataloader)}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_eval_batch_size}")

        model.eval()

        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                batch = batch.to(args.device)
                outputs = model(**batch)
                predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
                predictions, references = accelerator.gather((predictions, batch["labels"]))
                # If we are in a multiprocess environment, the last batch has duplicates
            if accelerator.num_processes > 1:
                if step == len(eval_dataloader) - 1:
                    predictions = predictions[: len(eval_dataloader.dataset) - samples_seen]
                    references = references[: len(eval_dataloader.dataset) - samples_seen]
                else:
                    samples_seen += references.shape[0]
            metric.add_batch(
                predictions=predictions,
                references=references,
            )

        eval_metric = metric.compute()
        acc_metric = "accuracy" if args.task_name is None else args.acc_metric
        logger.info(f"Evaluation metrics: {eval_metric}")
        return eval_metric[acc_metric]

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    nncf_config = register_default_init_args(nncf_config, train_dataloader, model_eval_fn=validate_fn)
    logger.info(f" NNCF config: {nncf_config}")

    # Apply the NNCF specified compression algorithms to the model
    compression_ctrl, compressed_model = create_compressed_model(model, nncf_config)

    compressed_model = accelerator.prepare(compressed_model)

    ## NNCF accuracy aware training loop
    acc_aware_training_loop = create_accuracy_aware_training_loop(nncf_config, compression_ctrl)
    optimized_model = acc_aware_training_loop.run(compressed_model,
                                        train_epoch_fn=train_epoch_fn,
                                        validate_fn=validate_fn,
                                        configure_optimizers_fn=configure_optimizers_fn,
                                        log_dir=optimized_model_dir)


    ## Save compressed model when done fine-tuning
    logger.info("\n***** Saving the compressed model *****")
    input_names = tokenizer.model_input_names
    if input_names[0] == "input_ids":
        if 'distil' not in args.model_name_or_path:
            input_names = ["input_ids", "attention_mask", "token_type_ids"]
        else:
            input_names = ["input_ids", "attention_mask"]
    
    path_to_onnx = os.path.join(optimized_model_dir, "ov_model.onnx")

    save_format = 'onnx_{}'.format(args.opset_version)

    # Export to ONNX 
    compression_ctrl.export_model(path_to_onnx, input_names=input_names, save_format=save_format)

    import subprocess

    subprocess.run(
        ["mo", "--input_model", path_to_onnx, "--output_dir", optimized_model_dir], check=True
    )

if __name__ == "__main__":
    main()
