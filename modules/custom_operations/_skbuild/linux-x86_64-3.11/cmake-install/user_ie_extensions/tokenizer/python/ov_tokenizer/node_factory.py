# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
from typing import Union

from openvino.runtime.utils.node_factory import NodeFactory


factory = NodeFactory()


def init_extension(extension_path: Union[str, Path]) -> None:
    """
    Initialize factory with compiled tokenizer extension.

    :param extension_path: path to prebuilt C++ tokenizer library.
    """
    factory.add_extension(extension_path)


if _extension_path := os.environ.get("OV_TOKENIZER_PREBUILD_EXTENSION_PATH"):
    init_extension(_extension_path)
