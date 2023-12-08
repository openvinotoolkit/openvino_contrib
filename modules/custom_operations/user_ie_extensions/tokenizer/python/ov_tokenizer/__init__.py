# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import functools
import os
import sys
import sysconfig
from pathlib import Path

import openvino
from openvino.runtime.utils.node_factory import NodeFactory

from .convert_tokenizer import convert_tokenizer
from .str_pack import pack_strings, unpack_strings
from .utils import add_greedy_decoding, connect_models


_extension_path = os.environ.get("OV_TOKENIZER_PREBUILD_EXTENSION_PATH")
_ext_name = "user_ov_extensions"
if _extension_path:
    # when the path to extension set manually
    _ext_libs_path = Path(_extension_path).parent
else:
    # python installation case
    _ext_libs_path = Path(sysconfig.get_paths()["purelib"]) / __name__ / "libs"

if sys.platform == "win32":
    _ext_path = _ext_libs_path / f"{_ext_name}.dll"
    if _ext_libs_path.is_dir():
        # On Windows, with Python >= 3.8, DLLs are no longer imported from the PATH.
        os.add_dll_directory(str(_ext_libs_path.absolute()))
    else:
        sys.exit(f"Error: extention libriary path {_ext_libs_path} not found")
elif sys.platform == "darwin":
    _ext_path = _ext_libs_path / f"lib{_ext_name}.dylib"
elif sys.platform == "linux":
    _ext_path = _ext_libs_path / f"lib{_ext_name}.so"
else:
    sys.exit(f"Error: extension does not support platform {sys.platform}")

# patching openvino
old_core_init = openvino.runtime.Core.__init__


@functools.wraps(old_core_init)
def new_core_init(self, *args, **kwargs):
    old_core_init(self, *args, **kwargs)
    self.add_extension(str(_ext_path))  # Core.add_extension doesn't support Path object


openvino.runtime.Core.__init__ = new_core_init

_factory = NodeFactory()
_factory.add_extension(_ext_path)
