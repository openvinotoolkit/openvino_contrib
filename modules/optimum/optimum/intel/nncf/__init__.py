import os
import inspect
import textwrap
from packaging import version

from .nncf_auto import NNCFAutoConfig

import transformers
from transformers import trainer, trainer_callback, training_args, modeling_utils

__all__ = ["NNCFAutoConfig"]


# This code patches Transformers methods for NNCF. Source:
# https://github.com/openvinotoolkit/nncf/blob/develop/third_party_integration/huggingface_transformers/0001-Modifications-for-NNCF-usage.patch
def replace_code_of_module(module, new_source):
    import ast

    code = compile(ast.parse(new_source), "<string>", "exec")
    exec(code, module.__dict__)  # nosec


"""
This method behaves like "git apply" applying patch to Transformers source code.

The changes are done in runtime replacing Python module source code.

Parameters:
    * module - Imported Python module
    * patch_file - A name of patch file
"""


def patch(module, patch_file):
    def insert(dst, src, idx):
        return dst[:idx] + src + dst[idx:]

    # Get module source code
    lines = inspect.getsourcelines(module)
    module_lines = [line.rstrip("\n") for line in lines[0]]

    # Read patch file
    path = os.path.join(os.path.dirname(__file__), "patches", patch_file)
    with open(path, "rt") as f:
        patch_data = f.readlines()
        patch_data = [line.rstrip("\n") for line in patch_data]

    i = 0
    while i < len(patch_data):
        changes = []
        start_idx = i
        end_idx = i + 1
        patch_line = patch_data[i]

        # Collect a block of new lines
        while patch_line.startswith(("+", "-")) and not patch_line.startswith(("+++", "---")):
            # Check for boundary between lines remove and addition
            if changes and patch_data[start_idx][0] != patch_line[0]:
                break

            changes.append(patch_line[1:])
            i += 1
            end_idx = i
            patch_line = patch_data[i]

        if not changes:
            i += 1
            continue

        add_lines = patch_data[start_idx].startswith("+")
        # Find a unique line before or after the newly added block
        idx = [i for i, line in enumerate(module_lines) if line == patch_data[start_idx - 1][1:]]
        if len(idx) == 1:
            if add_lines:
                module_lines = insert(module_lines, changes, idx[0] + 1)
            else:
                i -= end_idx - start_idx
                del patch_data[start_idx:end_idx]
                del module_lines[idx[0] + 1 : idx[0] + 1 + len(changes)]
        else:
            idx = [i for i, line in enumerate(module_lines) if line == patch_data[end_idx][1:]]
            if len(idx) == 1:
                if add_lines:
                    module_lines = insert(module_lines, changes, idx[0])
                else:
                    i -= end_idx - start_idx
                    del patch_data[start_idx:end_idx]
                    del module_lines[idx[0] - len(changes) : idx[0]]
            else:
                changes = [patch_data[start_idx - 1]] + patch_data[start_idx:end_idx] + [patch_data[end_idx]]
                msg = "\n".join(changes)
                raise Exception(f"Failed to apply patch:\n{msg}")

    # Restore newline characters
    code = "".join([line + "\n" for line in module_lines])

    code = textwrap.dedent(code)
    replace_code_of_module(module, code)


if version.parse(transformers.__version__) >= version.parse("4.11.0"):
    patch(trainer, "trainer_4.11.0.patch")
else:
    patch(trainer, "trainer_4.9.1.patch")
patch(training_args, "training_args.patch")
patch(trainer_callback, "trainer_callback.patch")
patch(modeling_utils, "modeling_utils.patch")
