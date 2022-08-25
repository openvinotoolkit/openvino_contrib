import os
import xml.etree.ElementTree as ET
import platform


def _get_lib_file_extension() -> str:
    platform_name = platform.system()
    if platform_name == 'Linux':
        return "so"
    elif platform_name == 'Windows':
        return "dll"
    elif platform_name == 'Darwin':
        return "dylib"


def _register_cuda_plugin():
    import openvino
    openvino_package_dir = os.path.dirname(os.path.abspath(openvino.__file__))
    openvino_package_libs_dir = os.path.join(openvino_package_dir, "libs")
    openvino_cuda_package_dir = os.path.dirname(os.path.abspath(__file__))
    openvino_cuda_library = os.path.join(openvino_cuda_package_dir, f"../libCUDAPlugin.{_get_lib_file_extension()}")

    xml_file = os.path.join(openvino_package_libs_dir, "plugins.xml")
    tree = ET.parse(xml_file)
    plugins = tree.find("plugins")
    if all(plugin.get('name') != 'CUDA' for plugin in plugins.iter('plugin')):
        plugins.append(ET.Element('plugin', {'name': 'CUDA', 'location': openvino_cuda_library}))
        tree.write(xml_file)


_register_cuda_plugin()

__version__ = "2022.1.0"
