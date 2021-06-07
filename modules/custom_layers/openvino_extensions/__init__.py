import os
import sys

def get_extensions_path():
    lib_name = 'user_cpu_extension'
    if sys.platform == 'win32':
        lib_name += '.dll'
    elif sys.platform == 'linux':
        lib_name = 'lib' + lib_name + '.so'
    else:
        lib_name = 'lib' + lib_name + '.dylib'
    return os.path.join(os.path.dirname(__file__), lib_name)
