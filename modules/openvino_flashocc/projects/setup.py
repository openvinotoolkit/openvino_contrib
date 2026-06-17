from setuptools import find_packages, setup


if __name__ == '__main__':
    setup(
        name='flashocc_plugin',
        description='FlashOCC plugin package for OpenVINO workflow',
        packages=find_packages(),
        ext_modules=[],
        zip_safe=False,
    )
