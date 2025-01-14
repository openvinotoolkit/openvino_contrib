import os
import defusedxml.ElementTree as ET
from defusedxml import defuse_stdlib
import platform

# defuse_stdlib provide patched version of xml.etree.ElementTree which allows to use objects from xml.etree.ElementTree
# in a safe manner without including unsafe xml.etree.ElementTree
ET_defused = defuse_stdlib()[ET]
Element = ET_defused.Element

def _get_lib_file_extension() -> str:
    platform_name = platform.system()
    if platform_name == 'Linux':
        return "so"
    elif platform_name == 'Windows':
        return "dll"
    elif platform_name == 'Darwin':
        return "dylib"


def _register_nvidia_plugin():
    import openvino
    openvino_package_dir = os.path.dirname(os.path.abspath(openvino.__file__))
    openvino_package_libs_dir = os.path.join(openvino_package_dir, "libs")
    openvino_nvidia_gpu_package_dir = os.path.dirname(os.path.abspath(__file__))
    openvino_nvidia_gpu_library = os.path.join(openvino_nvidia_gpu_package_dir, f"../libopenvino_nvidia_gpu_plugin.{_get_lib_file_extension()}")

    xml_file = os.path.join(openvino_package_libs_dir, "plugins.xml")
    tree = ET.parse(xml_file).getroot()
    plugins = tree.find("plugins")
    if all(plugin.get('name') != 'NVIDIA' for plugin in plugins.iter('plugin')):
        plugins.append(Element('plugin', {'name': 'NVIDIA', 'location': openvino_nvidia_gpu_library}))
        with open(xml_file, "w") as f:
            f.write(ET.tostring(tree).decode('utf8'))


_register_nvidia_plugin()

__version__ = "2024.1.0"
