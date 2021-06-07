#!/usr/bin/env python
import os
from setuptools import setup

if not 'EXT_LIB' in os.environ:
    raise Exception('Specify <EXT_LIB> environment variable with a path to extensions library')

setup(name='openvino-extensions',
      version='0.0.0',
      author='Intel Corporation',
      author_email='openvino_pushbot@intel.com',
      url='https://github.com/openvinotoolkit/openvino',
      packages=['openvino_extensions'],
      data_files=[('../../openvino_extensions', [os.environ['EXT_LIB']])],
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
      ],
)
