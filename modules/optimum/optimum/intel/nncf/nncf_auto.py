import os
from packaging import version
from copy import deepcopy

from nncf import NNCFConfig
from nncf.config.structures import BNAdaptationInitArgs
from nncf.config.structures import QuantizationRangeInitArgs
from nncf.torch.initialization import PTInitializingDataLoader

from typing import List


def get_train_dataloader_for_init(args, train_dataset, data_collator=None):
    from torch.utils.data import RandomSampler
    from torch.utils.data import DistributedSampler

    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)

    if data_collator is None:
        from transformers.data.data_collator import default_data_collator

        data_collator = default_data_collator

    from torch.utils.data import DataLoader

    data_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        sampler=train_sampler,
        collate_fn=data_collator,
        drop_last=args.dataloader_drop_last,
    )
    return data_loader


def filter_columns(dataset, keep_columns: List[str], remove_columns: List[str]):
    import datasets

    if version.parse(datasets.__version__) < version.parse("1.4.0"):
        dataset.set_format(
            type=dataset.format["type"], columns=keep_columns, format_kwargs=dataset.format["format_kwargs"]
        )
        return dataset
    else:
        return dataset.remove_columns(remove_columns)


def get_data_loader_cls(args, train_dataset):
    dataset_name = train_dataset.info.builder_name
    task_name = train_dataset.info.config_name
    if dataset_name == "squad":

        class SquadInitializingDataloader(PTInitializingDataLoader):
            def get_inputs(self, dataloader_output):
                return (), dataloader_output

        return SquadInitializingDataloader
    elif dataset_name == "glue":
        if task_name == "sst2":

            class SST2InitializingDataLoader(PTInitializingDataLoader):
                def get_inputs(self, dataloader_output):
                    return (
                        (),
                        {
                            "labels": dataloader_output["labels"],
                            "attention_mask": dataloader_output["attention_mask"],
                            "input_ids": dataloader_output["input_ids"],
                        },
                    )

            return SST2InitializingDataLoader

        if task_name == "mrpc":

            class MRPCInitializingDataLoader(PTInitializingDataLoader):
                def get_inputs(self, dataloader_output):
                    return (
                        (),
                        {
                            "labels": dataloader_output["labels"],
                            "attention_mask": dataloader_output["attention_mask"],
                            "input_ids": dataloader_output["input_ids"],
                            "token_type_ids": dataloader_output["token_type_ids"],
                        },
                    )

            return MRPCInitializingDataLoader

        if task_name == "mnli":

            class MNLIInitializingDataLoader(PTInitializingDataLoader):
                def get_inputs(self, dataloader_output):
                    return (
                        (),
                        {
                            "labels": dataloader_output["labels"],
                            "attention_mask": dataloader_output["attention_mask"],
                            "input_ids": dataloader_output["input_ids"],
                        },
                    )

            return MNLIInitializingDataLoader
    elif dataset_name == "xnli":

        class KwargBasedInitializingDataloader(PTInitializingDataLoader):
            def get_inputs(self, dataloader_output):
                return (), dataloader_output

        return KwargBasedInitializingDataloader
    elif dataset_name == "conll2003":

        class ConllInitializingDataloader(PTInitializingDataLoader):
            def get_inputs(self, dataloader_output):
                return (
                    (),
                    {
                        "input_ids": dataloader_output["input_ids"],
                        "attention_mask": dataloader_output["attention_mask"],
                        "token_type_ids": dataloader_output["token_type_ids"],
                    },
                )

        return ConllInitializingDataloader
    else:

        class DefaultDataLoader(PTInitializingDataLoader):
            def get_inputs(self, dataloader_output):
                return (), dataloader_output

        return DefaultDataLoader


class NNCFAutoConfig(NNCFConfig):
    """Class providing automatic NNCF config setup and dataset loader adoptation."""

    def auto_register_extra_structs(self, args, train_dataset, data_collator):
        if self.get("log_dir") is None:
            self["log_dir"] = args.output_dir
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(self["log_dir"])
        if args.do_train:
            if train_dataset.info.builder_name == "conll2003":
                train_dataset = deepcopy(train_dataset)
                train_dataset = filter_columns(
                    train_dataset,
                    keep_columns=["labels", "input_ids", "attention_mask", "token_type_ids"],
                    remove_columns=["ner_tags", "pos_tags", "tokens", "id", "chunk_tags"],
                )

            train_dataloader = get_train_dataloader_for_init(args, train_dataset, data_collator)

            initializing_data_loader_cls = get_data_loader_cls(args, train_dataset)

            self.register_extra_structs(
                [
                    QuantizationRangeInitArgs(initializing_data_loader_cls(train_dataloader)),
                    BNAdaptationInitArgs(initializing_data_loader_cls(train_dataloader)),
                ]
            )
