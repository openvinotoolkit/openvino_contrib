from setuptools import find_packages, setup

EXTRAS_REQUIRE = {
    "tests": ["onnx", "onnxruntime", "accelerate", "diffusers", "openvino", "optimum", "optimum-intel", "timm"],
}

setup(
    name="tomeov",
    version="0.1.0",
    author="Alexander Kozlov",
    url="https://github.com/openvinotoolkit/openvino_contrib/tree/master/modules/token_merging",
    description="Token Merging for OpenVINO",
    install_requires=["torch==1.13.1+cpu", "torchvision==0.14.1+cpu"],
    dependency_links=["https://download.pytorch.org/whl/cpu"],
    extras_require=EXTRAS_REQUIRE,
    packages=find_packages(exclude=("examples", "build")),
    license = 'Apache 2.0',
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
)