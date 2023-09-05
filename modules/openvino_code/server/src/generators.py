from functools import lru_cache
from pathlib import Path
from threading import Thread
from typing import Any, Callable, Container, Dict, List, Optional, Type, Union

import torch
from huggingface_hub.utils import EntryNotFoundError
from optimum.intel import OVModelForCausalLM, OVModelForSeq2SeqLM
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)

from src.utils import get_logger


logger = get_logger(__name__)

OVModel = Union[OVModelForSeq2SeqLM, OVModelForCausalLM]

model_dir = Path("models")
model_dir.mkdir(exist_ok=True)

INSTRUCTION_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
)
SUMMARIZE_INSTRUCTION = "The function description in numpy style"
SUMMARIZE_STOP_TOKENS = ("\n", ".\n")


def get_model_class(checkpoint: Union[str, Path]) -> Type[OVModel]:
    config = AutoConfig.from_pretrained(checkpoint)
    architecture: str = config.architectures[0]
    if architecture.endswith("ConditionalGeneration") or architecture.endswith("Seq2SeqLM"):
        return OVModelForSeq2SeqLM

    return OVModelForCausalLM


def get_model(checkpoint: str, device: str = "CPU") -> OVModel:
    ov_config = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1"}
    model_path = model_dir / Path(checkpoint)
    if model_path.exists():
        model_class = get_model_class(model_path)
        model = model_class.from_pretrained(model_path, ov_config=ov_config, compile=False, device=device)
    else:
        model_class = get_model_class(checkpoint)
        try:
            model = model_class.from_pretrained(checkpoint, ov_config=ov_config, compile=False, device=device, trust_remote_code=True)
        except EntryNotFoundError:
            model = model_class.from_pretrained(
                checkpoint, ov_config=ov_config, export=True, compile=False, device=device, trust_remote_code=True
            )
        model.save_pretrained(model_path)
    model.compile()
    return model


class GeneratorFunctor:
    def __call__(self, input_text: str, parameters: Dict[str, Any]) -> str:
        raise NotImplementedError

    async def generate_stream(self, input_text: str, parameters: Dict[str, Any]):
        raise NotImplementedError


class OVGenerator(GeneratorFunctor):
    def __init__(
        self,
        checkpoint: str,
        device: str = "CPU",
        tokenizer_checkpoint: Optional[str] = None,
        assistant_checkpoint: Optional[str] = None,
        summarize_stop_tokens: Optional[Container[str]] = SUMMARIZE_STOP_TOKENS,
    ) -> None:
        self.device = device
        self.model = get_model(checkpoint, device)
        # self.model = AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=True)

        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
            tokenizer_checkpoint if tokenizer_checkpoint is not None else checkpoint,
            trust_remote_code=True,
        )
        self.tokenizer.truncation_side = "left"
        self.tokenizer.truncation = True

        self.generation_config = GenerationConfig.from_model_config(self.model.config)
        self.generation_config.pad_token_id = self.tokenizer.eos_token_id

        self.assistant_model_config = {}
        if assistant_checkpoint is not None:
            self.assistant_model = get_model(assistant_checkpoint, device)
            # self.assistant_model = AutoModelForSeq2SeqLM.from_pretrained(assistant_checkpoint)
            self.assistant_model_config["assistant_model"] = self.assistant_model

        self.summarize_stopping_criteria = None
        if summarize_stop_tokens:
            stop_tokens_ids = self.tokenizer(summarize_stop_tokens).input_ids
            self.summarize_stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_tokens_ids)])

    def __call__(
        self, input_text: str, parameters: Dict[str, Any], stopping_criteria: Optional[StoppingCriteriaList] = None
    ) -> str:
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        prompt_len = input_ids.shape[-1]
        config = GenerationConfig.from_dict({**self.generation_config.to_dict(), **parameters})
        output_ids = self.model.generate(
            input_ids, generation_config=config, stopping_criteria=stopping_criteria, **self.assistant_model_config
        )[0][prompt_len:]
        logger.info(f"Number of input tokens: {prompt_len}; generated {len(output_ids)} tokens")
        return self.tokenizer.decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    async def generate_stream(
        self, input_text: str, parameters: Dict[str, Any], stopping_criteria: Optional[StoppingCriteriaList] = None
    ):
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        parameters["streamer"] = streamer
        config = GenerationConfig.from_dict({**self.generation_config.to_dict(), **parameters})
        generation_kwargs = dict(
            input_ids=input_ids,
            streamer=streamer,
            stopping_criteria=stopping_criteria,
            **config.to_dict(),
        )
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        for token in streamer:
            yield token

    def summarization_input(self, input_text: str) -> str:
        return INSTRUCTION_WITH_INPUT.format(
            instruction=SUMMARIZE_INSTRUCTION,
            input=input_text,
        )

    def summarize(self, input_text: str, parameters: Dict[str, Any]) -> str:
        return self(
            self.summarization_input(input_text), parameters, stopping_criteria=self.summarize_stopping_criteria
        )

    async def summarize_stream(self, input_text: str, parameters: Dict[str, Any]):
        async for token in self.generate_stream(
            self.summarization_input(input_text), parameters, stopping_criteria=self.summarize_stopping_criteria
        ):
            yield token


def get_generator_dependency(
    checkpoint: str,
    device: str = "CPU",
    tokenizer_checkpoint: Optional[str] = None,
    assistant: Optional[str] = None,
) -> Callable[[], GeneratorFunctor]:
    generator = OVGenerator(checkpoint, device, tokenizer_checkpoint, assistant)

    @lru_cache(1)
    def inner() -> GeneratorFunctor:
        return generator

    return inner


class StopOnTokens(StoppingCriteria):
    def __init__(self, token_ids: List[List[int]]) -> None:
        self.token_ids = [torch.tensor(ids, requires_grad=False) for ids in token_ids]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return any(
            torch.all(input_ids[0][-len(stopping_token_ids) :] == stopping_token_ids)
            for stopping_token_ids in self.token_ids
        )
