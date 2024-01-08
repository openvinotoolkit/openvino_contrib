import sys
from functools import lru_cache

import atheris
import numpy as np
from openvino import compile_model
from transformers import AutoTokenizer
import unicodedata


with atheris.instrument_imports():
    from openvino_tokenizers import convert_tokenizer


def remove_control_characters(s):
    return "".join(ch for ch in s if not unicodedata.category(ch).startswith("C"))


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
    input_text = remove_control_characters(fdp.ConsumeUnicodeNoSurrogates(sys.maxsize))

    if not input_text:
        return

    hf, ovt = get_tokenizers("Qwen/Qwen-14B-Chat")

    hf_tokenized = hf([input_text], return_tensors="np").input_ids
    ov_tokenized = ovt([input_text])["input_ids"]

    if not np.all(ov_tokenized == hf_tokenized):
        raise RuntimeError(
            f"Test failed! Test string: `{input_text}`, {input_text.encode()}\n"
            f"{ov_tokenized, hf_tokenized}\n"
            # f"{hf.decode(ov_tokenized), hf.decode(hf_tokenized)}"
            f"{type(ov_tokenized), type(hf_tokenized)}"
        )


def main():
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
