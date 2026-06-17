import os

# Default to OV-only safe imports. Legacy full imports can be enabled explicitly.
from .models import *

if os.getenv("FLASHOCC_ENABLE_LEGACY_IMPORTS", "0") == "1":
	from .core import *
	try:
		from .datasets import *
	except Exception:
		pass
