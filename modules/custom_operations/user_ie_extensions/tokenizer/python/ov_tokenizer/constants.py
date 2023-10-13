from enum import StrEnum, auto


class DecodingType(StrEnum):
    greedy = auto()


ATTENTION_MASK_INPUT_NAME = "attention_mask"
TOKEN_IDS_INPUT_NAME = "input_ids"
TOKEN_TYPE_IDS_INPUT_NAME = "token_type_ids"

LOGITS_OUTPUT_NAME = "logits"
TOKEN_IDS_OUTPUT_NAME = "token_ids"
STRING_OUTPUT_NAME = "string_output"

GREEDY_DECODER_NAME = "greedy_decoder"

TOKENIZER_ENCODER_NAME = "tokenizer_encoder"
TOKENIZER_DECODER_NAME = "tokenizer_decoder"
