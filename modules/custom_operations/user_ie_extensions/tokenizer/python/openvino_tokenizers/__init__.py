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
if _extension_path:
    # when the path to the extension set manually
    _ext_libs_path = Path(_extension_path).parent
else:
    # python installation case
    _ext_libs_path = Path(sysconfig.get_paths()["purelib"]) / __name__ / "lib"

_ext_name = "user_ov_extensions"
if sys.platform == "win32":
    _ext_name = f"{_ext_name}.dll"
elif sys.platform == "darwin":
    _ext_name = f"lib{_ext_name}.dylib"
elif sys.platform == "linux":
    _ext_name = f"lib{_ext_name}.so"
else:
    sys.exit(f"Error: extension does not support the platform {sys.platform}")

_ext_path = _ext_libs_path / _ext_name
if not _ext_path.is_file():
    # Case when the library can be found in the PATH/LD_LIBRAY_PATH
    _ext_path = _ext_name

del _ext_name
del _ext_libs_path
del _extension_path

# patching openvino
old_core_init = openvino.runtime.Core.__init__

@functools.wraps(old_core_init)
def new_core_init(self, *args, **kwargs):
    old_core_init(self, *args, **kwargs)
    self.add_extension(str(_ext_path))  # Core.add_extension doesn't support Path object

openvino.runtime.Core.__init__ = new_core_init

_factory = NodeFactory()
_factory.add_extension(_ext_path)
