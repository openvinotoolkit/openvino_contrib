#!/usr/bin/env python3
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Strip an mmdetection3d checkpoint down to a plain ``state_dict``.

The published BEVFusion checkpoints pickle mmengine objects into their ``meta``
field (the training config), so ``torch.load`` raises ``ModuleNotFoundError: No
module named 'mmengine'`` unless the whole mmengine/mmcv stack is installed.
``export.py`` only ever reads ``ckpt['state_dict']``, so we unpickle with stub
modules standing in for the mm* packages and re-save just the weights.

Usage:
    python sanitize_checkpoint.py in.pth out.pth
"""
from __future__ import annotations

import argparse
import importlib.abc
import importlib.machinery
import sys
import types

import torch

# mm* classes appearing in the pickle are all dict-like config containers, so a
# dict subclass reconstructs them faithfully enough to reach the state_dict.
STUB_PACKAGES = ('mmengine', 'mmcv', 'mmdet', 'mmdet3d')


class _StubConfig(dict):
    """Stand-in for mmengine.config.ConfigDict and friends."""


class _StubModule(types.ModuleType):
    """Package-like module whose every attribute resolves to _StubConfig."""

    __path__: list = []

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        # A dotted lookup may be either a submodule or a class; submodules are
        # already in sys.modules by the time pickle asks for them.
        return sys.modules.get(f'{self.__name__}.{name}', _StubConfig)


class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Import hook that fabricates a _StubModule for any mm* module path."""

    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split('.')[0]
        if root not in STUB_PACKAGES:
            return None
        return importlib.machinery.ModuleSpec(fullname, self, is_package=True)

    def create_module(self, spec):
        return _StubModule(spec.name)

    def exec_module(self, module):
        pass


def _install_stubs() -> None:
    """Make the mm* packages importable as no-op stubs."""
    sys.meta_path.insert(0, _StubFinder())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', help='Checkpoint to read')
    parser.add_argument('output', help='Where to write the weights-only checkpoint')
    args = parser.parse_args()

    _install_stubs()

    ckpt = torch.load(args.input, map_location='cpu', weights_only=False)
    state = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
    if not isinstance(state, dict) or not state:
        raise SystemExit(f'No state_dict found in {args.input}')

    tensors = {k: v for k, v in state.items() if torch.is_tensor(v)}
    dropped = len(state) - len(tensors)
    torch.save({'state_dict': tensors}, args.output)

    print(f'{args.input} -> {args.output}')
    print(f'  {len(tensors)} weight tensors kept'
          + (f', {dropped} non-tensor entries dropped' if dropped else ''))
    prefixes = sorted({k.split(".")[0] for k in tensors})
    print(f'  top-level prefixes: {prefixes}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
