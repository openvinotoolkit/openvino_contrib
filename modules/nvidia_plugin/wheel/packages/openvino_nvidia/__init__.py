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

def get_gpu_llm_inference_config() -> dict:
    """
    Returns a dictionary of OpenVINO properties optimized for LLM inference on NVIDIA GPUs.
    This configuration aims to improve performance and stability for stateful models
    by leveraging FP16 precision and disabling profiling.
    """
    # Import openvino here to ensure it's available when this function is called.
    # Using 'ov' as an alias for consistency with typical OpenVINO usage.
    import openvino as ov 
    return {
        ov.properties.hint.performance_mode: ov.properties.hint.PerformanceMode.LATENCY,
        # Hint for FP16 precision to utilize NVIDIA Tensor Cores for LLM inference
        ov.properties.hint.inference_precision: ov.properties.Type.f16,
        # Disable profiling to reduce overhead and potentially prevent interference with stateful models
        ov.properties.enable_profiling: False,
    }


def _register_nvidia_plugin(force=False):
    import openvino
    openvino_package_dir = os.path.dirname(os.path.abspath(openvino.__file__))
    openvino_package_libs_dir = os.path.join(openvino_package_dir, "libs")
    openvino_nvidia_gpu_package_dir = os.path.dirname(os.path.abspath(__file__))
    openvino_nvidia_gpu_library = os.path.join(openvino_nvidia_gpu_package_dir, f"./lib/libopenvino_nvidia_gpu_plugin.{_get_lib_file_extension()}")

    xml_file = os.path.join(openvino_package_libs_dir, "plugins.xml")
    xml_file_updated = False
    tree = ET.parse(xml_file).getroot()
    plugins = tree.find("plugins")
    
    if force:
        for plugin in plugins:
            if plugin.tag == "plugin" and plugin.get("name") == "NVIDIA":
                plugins.remove(plugin)
                plugins.append(Element('plugin', {'name': 'NVIDIA', 'location': openvino_nvidia_gpu_library}))
                xml_file_updated = True
    else:
        if all(plugin.get('name') != 'NVIDIA' for plugin in plugins.iter('plugin')):
            plugins.append(Element('plugin', {'name': 'NVIDIA', 'location': openvino_nvidia_gpu_library}))
            xml_file_updated = True
    
    if xml_file_updated:
        with open(xml_file, "w") as f:
            f.write(ET.tostring(tree).decode('utf8'))


def __getattr__(name):
    if name == "install":
        _register_nvidia_plugin()
    elif name == "force_install":
        _register_nvidia_plugin(True)
    else:
        raise AttributeError(f"module {__name__} has no attribute {name}")


_register_nvidia_plugin()


__version__ = "2025.3.0"