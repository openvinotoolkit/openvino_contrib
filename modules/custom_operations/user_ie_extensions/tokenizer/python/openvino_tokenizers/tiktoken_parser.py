import logging
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from tiktoken import Encoding


# from transformers.models.gpt2.tokenization_gpt2
@lru_cache()
def bytes_to_unicode() -> Dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = (chr(n) for n in cs)
    return dict(zip(bs, cs))


#  https://gist.github.com/xenova/a452a6474428de0182b17605a98631ee
def token_bytes_to_string(b: bytes) -> str:
    byte_encoder = bytes_to_unicode()
    return "".join(byte_encoder[ord(char)] for char in b.decode("latin-1"))


def bpe(mergeable_ranks: Dict[bytes, int], token: bytes, max_rank: Optional[int] = None) -> List[bytes]:
    parts = [bytes([b]) for b in token]
    while True:
        min_idx = None
        min_rank = None
        for i, pair in enumerate(zip(parts[:-1], parts[1:])):
            rank = mergeable_ranks.get(pair[0] + pair[1])
            if rank is not None and (min_rank is None or rank < min_rank):
                min_idx = i
                min_rank = rank
        if min_rank is None or (max_rank is not None and min_rank >= max_rank):
            break
        if min_idx is None:
            raise ValueError(f"Tiktoken conversion error: cannot determine bpe for token {token}.")
        parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2 :]
    return parts


def generate_vocab_and_merges(encoding: Encoding) -> Tuple[Dict[str, int], List[str]]:
    mergeable_ranks = encoding._mergeable_ranks

    merges = []
    vocab = {}
    for token, rank in mergeable_ranks.items():
        vocab[token_bytes_to_string(token)] = rank

        if len(token) == 1:
            continue
        merged = tuple(bpe(mergeable_ranks, token, max_rank=rank))

        #  if special tokens added to the tokenizer and the bpe split might produce more than 2 tokens
        #  if there are "\t" in the vocab and special token "\t\t\t" was added before "\t\t" it will
        #  be tokenized into 3 tokens: bpe("\t\t\t") -> ["\t", "\t", "\t"] which is cannot be included
        #  in merges
        if len(merged) == 2:
            merges.append(" ".join(map(token_bytes_to_string, merged)))
        else:
            logging.warning("Skip merges for added tokens. Tokenization results might be different.")

    # Also add special tokens
    vocab.update(encoding._special_tokens)

    return vocab, merges
