from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="ov_training_kit",
    version="0.1.1",
    description="Wrappers for scikit-learn and PyTorch models with OpenVINO optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/openvinotoolkit/openvino_contrib",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
        "scikit-learn>=1.2.0",
        "scikit-learn-intelex>=2023.1.0",  
        "torch>=1.12.0",
        "openvino>=2023.0",
        "nncf>=2.7.0",
        "intel_extension_for_pytorch>=2.1.0",
        "joblib>=1.2.0",
        "numpy>=1.21.0",
        "psutil>=5.9.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "flake8>=6.0.0",
            "black>=23.0.0",
            "isort>=5.10.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx_rtd_theme>=1.0.0",
        ],
    },
    python_requires=">=3.8",
    license="Apache-2.0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="openvino scikit-learn pytorch machine-learning edge-ai",
)