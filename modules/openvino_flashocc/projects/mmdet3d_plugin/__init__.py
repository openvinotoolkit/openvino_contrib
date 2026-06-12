import os

# Conversion-safe mode avoids importing dataset/core modules that pull optional
# CUDA/C++ ops (mmcv._ext) unavailable in CPU-only conversion environments.
if os.getenv("FLASHOCC_CONVERSION_SAFE_IMPORTS", "0") == "1":
	from .models import *
else:
	from .datasets import *
	from .core import *
	from .models import *
