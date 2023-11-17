# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from io import BytesIO
from typing import List

import numpy as np
from numpy.typing import NDArray
from openvino.runtime.exceptions import UserInputError


def to_bytes(number: int) -> bytes:
    return number.to_bytes(4, "little")


def pack_string(string: str) -> NDArray:
    return np.frombuffer(bytes(string, "utf-8"), dtype=np.uint8)


def pack_strings(strings: List[str]) -> NDArray:
    """
    Convert any list of string to U8/1D numpy array compatible with converted OV model input
    """
    if not isinstance(strings, list):
        raise UserInputError("")

    batch_size = len(strings)
    if batch_size == 0:
        return to_bytes(0)

    buffer = BytesIO()
    buffer.write(to_bytes(batch_size))
    symbols = BytesIO()
    offset = 0
    buffer.write(to_bytes(offset))
    for string in strings:
        byte_string = string.encode("utf-8")
        offset += len(byte_string)

        buffer.write(to_bytes(offset))
        symbols.write(byte_string)

    buffer.write(symbols.getvalue())
    return np.frombuffer(buffer.getvalue(), np.uint8)


# TODO: handle possible sighed values in batch size and offsets
def unpack_strings(u8_tensor: NDArray, decoding_errors: str = "replace") -> List[str]:
    """
    Convert an array of uint8 elements to a list of strings; reverse to pack_strings
    """

    def from_bytes(offset: int, size: int) -> int:
        return int.from_bytes(u8_tensor[offset : offset + size], "little")

    batch_size = from_bytes(0, 4)
    strings = []
    for i in range(batch_size):
        begin = from_bytes(4 + i * 4, 4)
        end = from_bytes(4 + (i + 1) * 4, 4)
        length = end - begin
        begin += 4 * (batch_size + 2)
        strings.append(bytes(u8_tensor[begin : begin + length]).decode("utf-8", errors=decoding_errors))
    return strings
