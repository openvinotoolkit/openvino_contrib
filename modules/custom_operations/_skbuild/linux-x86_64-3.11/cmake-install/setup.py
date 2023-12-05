from skbuild import setup
from skbuild import constants

setup(
    packages=["ov_tokenizer"],
    package_dir={"": "user_ie_extensions/tokenizer/python"},
    cmake_install_dir="user_ie_extensions/tokenizer/python/ov_tokenizer/libs",
    cmake_args=['-DCUSTOM_OPERATIONS:STRING=tokenizer',
                '-DBUILD_FAST_TOKENIZERS=OFF'],
    python_requires='>=3'
)

# When building extension modules `cmake_install_dir` should always be set to the
# location of the package you are building extension modules for.
# Specifying the installation directory in the CMakeLists subtley breaks the relative
# paths in the helloTargets.cmake file to all of the library components.