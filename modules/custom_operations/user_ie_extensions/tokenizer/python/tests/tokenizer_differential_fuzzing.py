import sys
from functools import lru_cache

import atheris
import numpy as np
from openvino import compile_model
from transformers import AutoTokenizer


with atheris.instrument_imports():
    from openvino_tokenizers import convert_tokenizer


@lru_cache()
def get_tokenizers(hub_id):
    hf_tokenizer = AutoTokenizer.from_pretrained(hub_id, trust_remote_code=True)
    ov_tokenizer = compile_model(convert_tokenizer(hf_tokenizer, with_detokenizer=False))

    return (
        hf_tokenizer,
        ov_tokenizer,
    )


@atheris.instrument_func
def TestOneInput(input_bytes):
    fdp = atheris.FuzzedDataProvider(input_bytes)
    input_text = fdp.ConsumeUnicodeNoSurrogates(sys.maxsize)

    hf, ovt = get_tokenizers("Qwen/Qwen-14B-Chat")

    hf_tokenized = hf([input_text], return_tensors="np")
    ov_tokenized = ovt([input_text])

    if not np.all(ov_tokenized["input_ids"] == hf_tokenized["input_ids"]):
        raise RuntimeError(
            f"Test failed! Test string: `{input_text}`, {input_text.encode()}\n"
            f"{ov_tokenized['input_ids'], hf_tokenized['input_ids']}"
        )


def main():
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
