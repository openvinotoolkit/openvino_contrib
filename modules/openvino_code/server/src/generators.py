import asyncio
import re
from functools import lru_cache
from io import StringIO
from pathlib import Path
from threading import Thread
from time import time
from typing import (
    Any,
    Callable,
    Container,
    Dict,
    Generator,
    List,
    Optional,
    Type,
    Union,
)

import torch
from fastapi import Request
from huggingface_hub.utils import EntryNotFoundError
from optimum.intel import OVModelForCausalLM, OVModelForSeq2SeqLM
from transformers import (
    AutoConfig,
    AutoTokenizer,
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

SUMMARIZE_INSTRUCTION = "{function}\n\n# The function with {style} style docstring\n\n{signature}\n"
SUMMARIZE_STOP_TOKENS = ("\r\n", "\n")


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
            model = model_class.from_pretrained(
                checkpoint,
                ov_config=ov_config,
                compile=False,
                device=device,
                trust_remote_code=True,
            )
        except EntryNotFoundError:
            model = model_class.from_pretrained(
                checkpoint,
                ov_config=ov_config,
                export=True,
                compile=False,
                device=device,
                trust_remote_code=True,
            )
        model.save_pretrained(model_path)
    model.compile()
    return model


# TODO: generator needs running flag or cancellation on new generation request
# generator cannot handle concurrent requests - fails and stalls process
# RuntimeError: Exception from src/inference/src/infer_request.cpp:189:
# [ REQUEST_BUSY ]
class GeneratorFunctor:
    def __call__(self, input_text: str, parameters: Dict[str, Any]) -> str:
        raise NotImplementedError

    async def generate_stream(self, input_text: str, parameters: Dict[str, Any], request: Request):
        raise NotImplementedError

    def summarize(
        self,
        input_text: str,
        template: str,
        signature: str,
        style: str,
        parameters: Dict[str, Any],
    ):
        raise NotImplementedError

    def summarize_stream(
        self,
        input_text: str,
        template: str,
        signature: str,
        style: str,
        parameters: Dict[str, Any],
    ):
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
            self.assistant_model_config["assistant_model"] = self.assistant_model

        self.summarize_stopping_criteria = None
        if summarize_stop_tokens:
            stop_tokens = []
            for token_id in self.tokenizer.vocab.values():
                if any(stop_word in self.tokenizer.decode(token_id) for stop_word in summarize_stop_tokens):
                    stop_tokens.append(token_id)
            self.summarize_stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_tokens)])

    def __call__(self, input_text: str, parameters: Dict[str, Any]) -> str:
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")

        stopping_criteria = None
        if (timeout := parameters.pop("timeout", None)) is not None:
            stop_on_time = StopOnTime(timeout)
            stopping_criteria = StoppingCriteriaList([stop_on_time])

        prompt_len = input_ids.shape[-1]
        config = GenerationConfig.from_dict({**self.generation_config.to_dict(), **parameters})
        output_ids = self.model.generate(
            input_ids,
            generation_config=config,
            stopping_criteria=stopping_criteria,
            **self.assistant_model_config,
        )[0][prompt_len:]
        logger.info(f"Number of input tokens: {prompt_len}; generated {len(output_ids)} tokens")
        return self.tokenizer.decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    async def generate_stream(
        self,
        input_text: str,
        parameters: Dict[str, Any],
        request: Optional[Request] = None,
    ) -> Generator[str, None, None]:
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        parameters["streamer"] = streamer
        config = GenerationConfig.from_dict({**self.generation_config.to_dict(), **parameters})

        stop_on_tokens = StopOnTokens([])

        generation_kwargs = dict(
            input_ids=input_ids,
            streamer=streamer,
            stopping_criteria=StoppingCriteriaList([stop_on_tokens]),
            **config.to_dict(),
        )

        # listen disconnect event so generation can be stopped
        def listen_for_disconnect():
            async def listen():
                message = await request.receive()
                if message.get("type") == "http.disconnect":
                    stop_on_tokens.cancelled = True

            asyncio.create_task(listen())

        listen_thread = Thread(target=listen_for_disconnect)
        # thread.run doesn't actually start a new thread
        # it runs the thread function in current thread context
        # thread.start() doesn't work here
        listen_thread.run()

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for token in streamer:
            await asyncio.sleep(0.01)
            yield token

        thread.join()

    def generate_between(
        self,
        input_parts: List[str],
        parameters: Dict[str, Any],
        stopping_criteria: Optional[StoppingCriteriaList] = None,
    ) -> str:
        config = GenerationConfig.from_dict({**self.generation_config.to_dict(), **parameters})

        prompt = torch.tensor([[]], dtype=torch.int64)
        buffer = StringIO()
        for text_input in input_parts[:-1]:
            buffer.write(text_input)

            tokenized_input = self.tokenizer.encode(text_input, return_tensors="pt")
            prompt = torch.concat((prompt, tokenized_input), dim=1)
            prev_len = prompt.shape[-1]

            prompt = self.model.generate(
                prompt,
                generation_config=config,
                stopping_criteria=stopping_criteria,
                **self.assistant_model_config,
            )[
                :, :-1
            ]  # skip the last token - stop token

            decoded = self.tokenizer.decode(prompt[0, prev_len:], skip_special_tokens=True)
            buffer.write(decoded.lstrip(" "))  # hack to delete leadding spaces if there are any
        buffer.write(input_parts[-1])
        return buffer.getvalue()

    async def generate_between_stream(
        self,
        input_parts: List[str],
        parameters: Dict[str, Any],
        stopping_criteria: Optional[StoppingCriteriaList] = None,
    ) -> Generator[str, None, None]:
        config = GenerationConfig.from_dict({**self.generation_config.to_dict(), **parameters})

        prompt = self.tokenizer.encode(input_parts[0], return_tensors="pt")
        for text_input in input_parts[1:-1]:
            yield text_input

            tokenized_input = self.tokenizer.encode(text_input, return_tensors="pt")
            prompt = torch.concat((prompt, tokenized_input), dim=1)
            prev_len = prompt.shape[-1]

            prompt = self.model.generate(
                prompt,
                generation_config=config,
                stopping_criteria=stopping_criteria,
                **self.assistant_model_config,
            )[
                :, :-1
            ]  # skip the last token - stop token

            decoded = self.tokenizer.decode(prompt[0, prev_len:], skip_special_tokens=True)
            yield decoded.lstrip(" ")  # hack to delete leadding spaces if there are any

        yield input_parts[-1]

    @staticmethod
    def summarization_input(function: str, signature: str, style: str) -> str:
        return SUMMARIZE_INSTRUCTION.format(
            function=function,
            style=style,
            signature=signature,
        )

    def summarize(
        self,
        input_text: str,
        template: str,
        signature: str,
        style: str,
        parameters: Dict[str, Any],
    ) -> str:
        prompt = self.summarization_input(input_text, signature, style)
        splited_template = re.split(r"\$\{.*\}", template)
        splited_template[0] = prompt + splited_template[0]

        return self.generate_between(
            splited_template,
            parameters,
            stopping_criteria=self.summarize_stopping_criteria,
        )[len(prompt) :]

    async def summarize_stream(
        self,
        input_text: str,
        template: str,
        signature: str,
        style: str,
        parameters: Dict[str, Any],
    ):
        prompt = self.summarization_input(input_text, signature, style)
        splited_template = re.split(r"\$\{.*\}", template)
        splited_template = [prompt] + splited_template

        async for token in self.generate_between_stream(
            splited_template,
            parameters,
            stopping_criteria=self.summarize_stopping_criteria,
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
    def __init__(self, token_ids: List[int]) -> None:
        self.cancelled = False
        self.token_ids = torch.tensor(token_ids, requires_grad=False)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.cancelled:
            return True
        return torch.any(torch.eq(input_ids[0, -1], self.token_ids)).item()


class StopOnTime(StoppingCriteria):
    def __init__(self, timeout: float, budget_reduction: float = 0.99) -> None:
        self.time = time()
        self.stop_until = self.time + timeout * budget_reduction
        self.time_for_prev_token = 0.0
        self.grow_factor = 0.0

    def __call__(self, *args, **kwargs) -> bool:
        current_time = time()
        if current_time > self.stop_until:
            return True

        elapsed = current_time - self.time
        if self.time_for_prev_token > 0:
            self.grow_factor = elapsed / self.time_for_prev_token

        self.time_for_prev_token = elapsed
        self.time = current_time

        return self.stop_until < current_time + self.time_for_prev_token * self.grow_factor
