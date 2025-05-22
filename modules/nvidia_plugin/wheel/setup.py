# -*- coding: utf-8 -*-
# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os.path
import pathlib
import sys
import abc
import subprocess # nosec - disable B404:import-subprocess check
import platform
import shutil
import glob
import errno
import multiprocessing
import typing
import sysconfig
import defusedxml.ElementTree as ET
from defusedxml import defuse_stdlib

from textwrap import dedent
from pathlib import Path
from setuptools import setup
from setuptools.command.build_clib import build_clib
from setuptools.command.install_lib import install_lib
from decouple import config
from distutils import log
from typing import Optional

import wheel.vendored.packaging.tags as tags
import wheel.bdist_wheel
import wheel.vendored.packaging.tags

# defuse_stdlib provide patched version of xml.etree.ElementTree which allows to use objects from xml.etree.ElementTree
# in a safe manner without including unsafe xml.etree.ElementTree
ET_defused = defuse_stdlib()[ET]
Element = ET_defused.Element

platforms = ['linux', 'win32', 'darwin']
if not any(pl in sys.platform for pl in platforms):
    sys.exit(f'Unsupported platform: {sys.platform}, expected: linux, win32, darwin')

PACKAGE_NAME = config('WHEEL_PACKAGE_NAME', 'openvino-nvidia')
OPENVINO_REPO_URI = config('OPENVINO_REPO_DOWNLOAD_URL', 'https://github.com/openvinotoolkit/openvino.git')
WHEEL_VERSION = config('WHEEL_VERSION', "2024.1.0")
OPENVINO_REPO_TAG = config('OPENVINO_REPO_TAG', WHEEL_VERSION)
NVIDIA_PLUGIN_CMAKE_TARGET_NAME = 'openvino_nvidia_gpu_plugin'
LIBS_RPATH = '$ORIGIN' if sys.platform == 'linux' else '@loader_path'
OPENVINO_INSTALL_BUILD_DEPS_SCRIPT = "install_build_dependencies.sh"

openvino_src_dir: Optional[str] = None
build_configuration_name: Optional[str] = None
package_data: typing.Dict[str, list] = {}


def is_tool(name):
    """
    Check if the command-line tool is available
    :param name: Name of command-line tool to check
    :return: True if @name is command-line tool, False - otherwise
    """
    try:
        devnull = subprocess.DEVNULL
        subprocess.Popen([name], stdout=devnull, stderr=devnull).communicate()  # nosec
    except OSError as error:
        if error.errno == errno.ENOENT:
            return False
    return True


def remove_rpath(file_path: pathlib.Path):
    """
    Remove rpath from binaries
    :param file_path: binary path
    """
    if sys.platform == 'darwin':
        cmd = (
            f'otool -l {file_path} '  # noqa: P103
            f'| grep LC_RPATH -A3 '
            f'| grep -o "path.*" '
            f'| cut -d " " -f2 '
            f'| xargs -I{{}} install_name_tool -delete_rpath {{}} {file_path}'
        )
        if os.WEXITSTATUS(os.system(cmd)) != 0:  # nosec
            sys.exit(f'Could not remove rpath for {file_path}')
    else:
        sys.exit(f'Unsupported platform: {sys.platform}')


def set_rpath(rpath, executable):
    """
    Set rpath for linux and macOS libraries
    :param rpath:
    :param executable:
    """
    print(f'Setting rpath {rpath} for {executable}')  # noqa: T001
    cmd = []
    rpath_tool = ''

    if sys.platform == 'linux':
        with open(os.path.realpath(executable), 'rb') as file:
            if file.read(1) != b'\x7f':
                log.warn(f'WARNING: {executable}: missed ELF header')
                return
        rpath_tool = 'patchelf'
        cmd = [rpath_tool, '--set-rpath', rpath, executable]
    elif sys.platform == 'darwin':
        rpath_tool = 'install_name_tool'
        cmd = [rpath_tool, '-add_rpath', rpath, executable]
    else:
        sys.exit(f'Unsupported platform: {sys.platform}')

    if is_tool(rpath_tool):
        if sys.platform == 'darwin':
            remove_rpath(executable)
        ret_info = subprocess.run(cmd, check=True, shell=False)  # nosec
        if ret_info.returncode != 0:
            sys.exit(f'Could not set rpath: {rpath} for {executable}')
    else:
        sys.exit(f'Could not found {rpath_tool} on the system, ' f'please make sure that this tool is installed')


def get_description(desc_file_path):
    """Read description from README.md"""
    with open(desc_file_path, 'r', encoding='utf-8') as fstream:
        description = fstream.read()
    return description


def get_dependencies(requirements_file_path):
    """Read dependencies from requirements.txt"""
    if not os.path.exists(requirements_file_path):
        return ""
    with open(requirements_file_path, 'r', encoding='utf-8') as fstream:
        dependencies = fstream.read()
    return dependencies


def create_setup_py_command(setup_py_path, *options):
    if '--user' in sys.argv:
        return [sys.executable, setup_py_path, 'install', '--user', *options]
    else:
        return [sys.executable, setup_py_path, 'install', *options]


def create_pip_command(*options):
    if '--user' in sys.argv:
        return [sys.executable, '-m', 'pip', 'install', '--user', *options]
    else:
        return [sys.executable, '-m', 'pip', 'install', *options]


def run_command(command, cwd: str = None, on_fail_msg: str = '', env=None):
    try:
        subprocess.check_call(command, cwd=cwd, env=env) # nosec - disable B603:subprocess_without_shell_equals_true check
    except subprocess.CalledProcessError as e:
        raise RuntimeError(on_fail_msg) from e

def get_command_output(command, cwd: str = None, on_fail_msg: str = '', env=None):
    try:
        return subprocess.check_output(command, cwd=cwd, env=env).decode() # nosec - disable B603:subprocess_without_shell_equals_true check
    except subprocess.CalledProcessError as e:
        raise RuntimeError(on_fail_msg) from e

class AbsPlatformSpecific(abc.ABC):
    @abc.abstractmethod
    def get_lib_file_extension(self) -> str:
        """Get library file extension for a given platform"""

    @abc.abstractmethod
    def get_env_lib_path_variable(self) -> str:
        """Get library file extension for a given platform"""


class LinuxSpecific(AbsPlatformSpecific):
    def get_lib_file_extension(self) -> str:
        return "so"

    def get_env_lib_path_variable(self) -> str:
        return "LD_LIBRARY_PATH"


class DarwinSpecific(AbsPlatformSpecific):
    def get_lib_file_extension(self) -> str:
        return "dylib"

    def get_env_lib_path_variable(self) -> str:
        return "LD_LIBRARY_PATH"


class WindowsSpecific(AbsPlatformSpecific):
    def get_lib_file_extension(self) -> str:
        return "dll"

    def get_env_lib_path_variable(self) -> str:
        return "PATH"


platform_specifics: AbsPlatformSpecific = {
    "Windows": WindowsSpecific,
    "Darwin": DarwinSpecific,
    "Linux": LinuxSpecific
}[platform.system()]()


class BuildCMakeLib(build_clib):
    def finalize_options(self):
        """Set final values for all the options that this command supports"""
        super().finalize_options()

        global openvino_src_dir
        global build_configuration_name
        
        self.git_exec = shutil.which("git")
        self.cmake_exec = shutil.which("cmake")
        build_configuration_name = 'Debug' if self.debug else 'Release'
        self.nvidia_plugin_src_dir = os.path.abspath(os.path.dirname(__file__))
        self.build_lib_dir = os.path.join(self.nvidia_plugin_src_dir, "../build/lib")
        self.openvino_contrib_src_dir = os.path.normpath(os.path.join(self.nvidia_plugin_src_dir, "../../.."))
        self.deps_dir = os.path.abspath(os.path.join(self.build_temp, "deps"))
        self.openvino_src_dir = os.path.join(self.deps_dir, "openvino")
        openvino_src_dir = self.openvino_src_dir
        self.openvino_build_dir = os.path.join(self.openvino_src_dir, "build")
        self.openvino_dist_dir = os.path.join(self.deps_dir, "openvino_dist")
        for cmd in ['install', 'build']:
            self.set_undefined_options(cmd, ('force', 'force'))

    def run(self):
        if not self.libraries:
            return
        if self.git_exec is None:
            raise FileNotFoundError("git path not located on path")
        if self.cmake_exec is None:
            raise FileNotFoundError("cmake path not located on path")

        self.clone_openvino_src()
        # TODO: Uncomment when issue with conan dependecies will be resolved.
        #       When uncomment this line, got the following error during
        #       cmake configuration step:
        #           CMake Error at build/protobuf-Target-release.cmake:74 (set_property):
        #               set_property can not be used on an ALIAS target.
        #               ...
        # self.openvino_conan_install()
        self.configure_openvino_cmake()
        if self.force:
            self.build_openvino()
        self.build_nvidia_plugin()
        self.locate_built_lib()

    def clone_openvino_src(self):
        self.mkpath(self.deps_dir)

        if os.path.isdir(self.openvino_src_dir):
            return
        self.announce("Cloning the OpenVINO sources", level=3)
        run_command([self.git_exec, 'clone', '--recursive', '--single-branch', '--branch', OPENVINO_REPO_TAG,
                     OPENVINO_REPO_URI],
                    cwd=self.deps_dir,
                    on_fail_msg='Failed to clone the OpenVINO from git repository')
        run_command([self.git_exec, 'submodule', 'update', '--init', '--recursive'],
                    cwd=self.openvino_src_dir,
                    on_fail_msg='Failed to update the OpenVINO git submodules')

    def openvino_conan_install(self):
        if not os.path.isdir(self.openvino_build_dir):
            self.mkpath(self.openvino_build_dir)

        run_command(["conan",
                     "install",
                     f'-of={self.openvino_build_dir}',
                     '--build=missing',
                     self.openvino_src_dir],
                     cwd=self.openvino_build_dir,
                     on_fail_msg='Failed to install conan dependecies for OpenVINO CMake Project')

    def configure_openvino_cmake(self):
        if not os.path.isdir(self.openvino_build_dir):
            self.mkpath(self.openvino_build_dir)

        python_include_dir = sysconfig.get_path("include")
        configure_command = [self.cmake_exec,
                             '-G', 'Unix Makefiles',
                             f'-S{self.openvino_src_dir}',
                             f'-B{self.openvino_build_dir}',
                             '-DENABLE_PLUGINS_XML=ON',
                             '-DCMAKE_VERBOSE_MAKEFILE=ON',
                             '-DENABLE_NVIDIA=ON',
                             '-DENABLE_PYTHON=ON',
                             f'-DPython3_EXECUTABLE={sys.executable}',
                             f'-DPython3_INCLUDE_DIR={python_include_dir}',
                             f'-DPYTHON_EXECUTABLE={sys.executable}',
                             f'-DPYTHON_INCLUDE_DIR={python_include_dir}',
                             '-DNGRAPH_PYTHON_BUILD_ENABLE=ON',
                             f'-DCMAKE_BUILD_TYPE={build_configuration_name}',
                             f'-DOPENVINO_EXTRA_MODULES={self.openvino_contrib_src_dir}/modules/nvidia_plugin',
                             '-DENABLE_WHEEL=ON',
                             f'-DWHEEL_VERSION={WHEEL_VERSION}']
        self.announce("Configuring OpenVINO CMake Project", level=3)
        run_command(configure_command,
                    cwd=self.openvino_build_dir,
                    on_fail_msg='Failed to configure OpenVINO CMake Project. Ensure you have all build dependencies '
                                'installed, by running '
                                f'{os.path.join(self.openvino_src_dir, OPENVINO_INSTALL_BUILD_DEPS_SCRIPT)} '
                                'script with admin rights.')

    def build_openvino(self):
        self.announce("Building OpenVINO Project", level=3)
        run_command([self.cmake_exec, '--build', self.openvino_build_dir, '-j', f'{multiprocessing.cpu_count()}'],
                    cwd=self.openvino_build_dir,
                    on_fail_msg='Failed to build OpenVINO Project.')

    def get_build_env(self):
        build_env = os.environ.copy()
        build_env['BUILD_TYPE'] = build_configuration_name
        build_env['BUILD_TARGETS'] = NVIDIA_PLUGIN_CMAKE_TARGET_NAME
        build_env['OPENVINO_HOME'] = self.openvino_src_dir
        build_env['OPENVINO_CONTRIB'] = self.openvino_contrib_src_dir
        build_env['OPENVINO_BUILD_PATH'] = self.openvino_build_dir
        build_env['ENABLE_TESTS'] = "OFF"

        # Update 'CUDACXX' environment variable
        if 'CUDACXX' not in build_env:
            if 'CUDA_PATH' in build_env:
                build_env['CUDACXX'] = f"{build_env['CUDA_PATH']}/bin/nvcc"
            elif shutil.which("nvcc") is not None:
                build_env['CUDACXX'] = shutil.which("nvcc")

        if 'CUDACXX' not in build_env:
            raise RuntimeError("Cannot detect nvcc compiler path !!"
                               "Please, specify either 'CUDACXX' or 'CUDA_PATH' environment variables")

        # Update run-time path to shared library
        cuda_lib_path = os.path.normpath(os.path.join(os.path.dirname(build_env['CUDACXX']), "../lib64"))
        python_lib_path = os.path.normpath(os.path.join(os.path.dirname(sys.executable), "../lib"))
        libraries_path = cuda_lib_path + os.pathsep + python_lib_path
        env_lib_path_variable = platform_specifics.get_env_lib_path_variable()
        if env_lib_path_variable not in build_env:
            build_env[env_lib_path_variable] = libraries_path
        else:
            if libraries_path not in build_env[env_lib_path_variable]:
                build_env[env_lib_path_variable] += os.pathsep + libraries_path

        return build_env

    def build_nvidia_plugin(self):
        if not os.path.isdir(self.openvino_build_dir):
            self.mkpath(self.openvino_build_dir)

        build_env = self.get_build_env()

        self.announce("Building OpenVINO CUDA Plugin Project", level=3)
        run_command([os.path.join(self.nvidia_plugin_src_dir, "../build.sh"), '--build'],
                    cwd=self.nvidia_plugin_src_dir, env=build_env)

    def locate_built_lib(self):
        libs = []
        lib_ext = platform_specifics.get_lib_file_extension()
        bin_dir = os.path.join(self.openvino_src_dir, "bin")
        for name in [NVIDIA_PLUGIN_CMAKE_TARGET_NAME]:
            libs.extend(list(glob.iglob(f"{bin_dir}/**/*{name}*{lib_ext}", recursive=True)))
        if not libs:
            raise Exception("NVIDIA Plugin library not found. Possibly build was failed or was written to unknown "
                            "directory")
        self.mkpath(self.build_lib_dir)
        for lib in libs:
            self.copy_file(lib, os.path.join(self.build_lib_dir, os.path.basename(lib)))
            # set rpath if applicable
            if sys.platform != 'win32':
                file_types = ['.so'] if sys.platform == 'linux' else ['.dylib', '.so']
                for path in filter(lambda p: any(item in file_types for item in p.suffixes),
                                   Path(self.build_lib_dir).glob('*')):
                    set_rpath(LIBS_RPATH, os.path.realpath(path))


class InstallLib(install_lib):
    def finalize_options(self):
        install_lib.finalize_options(self)
        
        self.git_exec = shutil.which("git")
        self.force = None
        self.set_undefined_options('install', ('force', 'force'))
        print(f"self.force = {self.force}")

    def run(self):
        openvino_nvidia_gpu_library = f'{openvino_src_dir}/bin/intel64/{build_configuration_name}/lib/libopenvino_nvidia_gpu_plugin.{platform_specifics.get_lib_file_extension()}'
        package_data.update({
            '': [openvino_nvidia_gpu_library]
        })
        
        self.build()
        self.install()
        
        try:
            if self.force:
                self.install_openvino_package()
            self.install_python_dependencies()
            self.register_nvidia_plugin()
            self.test_nvidia_plugin()
        except Exception as ex:
            self.unregister_nvidia_plugin()

    def install_openvino_package(self):
        py_tag=tags.interpreter_name() + tags.interpreter_version()
        abi_tag=wheel.bdist_wheel.get_abi_tag()
        platform_tag=next(tags.platform_tags())
        git_commits=get_command_output([self.git_exec, 'rev-list', '--count', '--first-parent', 'HEAD'],
                                        cwd=openvino_src_dir,
                                        on_fail_msg='Failed to count OpenVINO commits').strip()
        openvino_wheel_name="-".join(["openvino", WHEEL_VERSION, git_commits, py_tag, abi_tag, platform_tag]) + ".whl"
        wheels_path = os.path.abspath(os.path.join(openvino_src_dir, "build/wheels", openvino_wheel_name))
        self.announce(f"Installing OpenVINO package with {wheels_path}", level=3)
        openvino_install_py = create_pip_command(wheels_path)
        run_command(openvino_install_py,
                    on_fail_msg=f'Failed to install OpenVINO wheel package with {wheels_path}')

    def install_python_dependencies(self):
        path_to_requirements_txt = os.path.join(os.path.abspath(os.path.dirname(__file__)), "requirements.txt")
        requirements_py = create_pip_command('-r', path_to_requirements_txt)
        run_command(requirements_py,
                    on_fail_msg=f'Failed to install dependencies from {path_to_requirements_txt}')

    def check_plugins_xml_if_exists(self):
        openvino_package_libs_dir = self.get_openvino_package_dir()
        xml_file = os.path.join(openvino_package_libs_dir, "plugins.xml")
        if not os.path.exists(xml_file):
            plugins_xml_src = dedent('''\
                <ie>
                    <plugins>
                    </plugins>
                </ie>
                ''')
            tree = ET.fromstring(plugins_xml_src)
            with open(xml_file, "w") as f:
                f.write(ET.tostring(tree).decode('utf8'))

    def update_plugins_xml(self, xml_file, openvino_nvidia_gpu_library):
        if not os.path.exists(xml_file):
            plugins_xml_src = dedent('''\
            <ie>
                <plugins>
                </plugins>
            </ie>
            ''')
            tree = ET.fromstring(plugins_xml_src)
        else:
            tree = ET.parse(xml_file).getroot()
        
        plugins = tree.find("plugins")
        if all(plugin.get('name') != 'NVIDIA' for plugin in plugins.iter('plugin')):
            plugins.append(Element('plugin', {'name': 'NVIDIA', 'location': openvino_nvidia_gpu_library}))
            with open(xml_file, "w") as f:
                f.write(ET.tostring(tree).decode('utf8'))

    def get_openvino_package_dir(self):
        import openvino
        openvino_package_dir = os.path.dirname(os.path.abspath(openvino.__file__))
        openvino_package_libs_dir = os.path.join(openvino_package_dir, "libs")
        return openvino_package_libs_dir

    def get_openvino_nvidia_lib_path(self):
        import openvino_nvidia
        openvino_nvidia_package_dir = os.path.dirname(os.path.abspath(openvino_nvidia.__file__))
        openvino_nvidia_gpu_library = f'{openvino_nvidia_package_dir}/libopenvino_nvidia_gpu_plugin.{platform_specifics.get_lib_file_extension()}'
        return openvino_nvidia_gpu_library

    def register_nvidia_plugin(self):
        self.check_plugins_xml_if_exists()
        openvino_package_libs_dir = self.get_openvino_package_dir()
        openvino_nvidia_gpu_library = self.get_openvino_nvidia_lib_path()
        xml_file = os.path.join(openvino_package_libs_dir, "plugins.xml")
        self.update_plugins_xml(xml_file, openvino_nvidia_gpu_library)

    def unregister_nvidia_plugin(self):
        openvino_package_libs_dir = self.get_openvino_package_dir()

        xml_file = os.path.join(openvino_package_libs_dir, "plugins.xml")
        tree = ET.parse(xml_file).getroot()
        plugins = tree.find("plugins")
        for plugin in plugins.iter('plugin'):
            if plugin.get('name') == 'NVIDIA':
                plugins.remove(plugin)
                with open(xml_file, "w") as f:
                    f.write(ET.tostring(tree).decode('utf8'))
                break

    def test_nvidia_plugin(self):
        import openvino as ov
        test_model_convert_fp32 = """
            <?xml version="1.0"?>
            <net name="Function_1208" version="10">
                <layers>
                    <layer id="0" name="Parameter_4829" type="Parameter" version="opset1">
                        <data shape="" element_type="i32" />
                        <output>
                            <port id="0" precision="I32" />
                        </output>
                    </layer>
                    <layer id="1" name="Convert_4830" type="Convert" version="opset1">
                        <data destination_type="f32" />
                        <input>
                            <port id="0" />
                        </input>
                        <output>
                            <port id="1" precision="FP32" />
                        </output>
                    </layer>
                    <layer id="2" name="Result_1684641" type="Result" version="opset1">
                        <input>
                            <port id="0" />
                        </input>
                    </layer>
                </layers>
                <edges>
                    <edge from-layer="0" from-port="0" to-layer="1" to-port="0" />
                    <edge from-layer="1" from-port="1" to-layer="2" to-port="0" />
                </edges>
            </net>
            """.encode('ascii')
        core = ov.Core()
        model = core.read_model(model=test_model_convert_fp32)
        try:
            core.compile_model(model=model, device_name="NVIDIA")
        except RuntimeError as e:
            recommendations_msg = ''
            if not self.force:
                recommendations_msg = 'Try to uninstall the openvino package and run "setup.py install --force" ' \
                                      'to build OpenVINO libraries also.'
            raise RuntimeError('The NVIDIA GPU plugin loading test was failed. '
                               'The NVIDIA GPU plugin library is not compatible with OpenVINO libraries. '
                               f'Possible ABI version mismatch. {recommendations_msg}') from e


setup(
    version=config('WHEEL_VERSION', WHEEL_VERSION),
    author_email=config('WHEEL_AUTHOR_EMAIL', 'openvino_pushbot@intel.com'),
    name=PACKAGE_NAME,
    license=config('WHEEL_LICENCE_TYPE', 'OSI Approved :: Apache Software License'),
    author=config('WHEEL_AUTHOR', 'Intel Corporation'),
    description=config('WHEEL_DESC', 'NVIDIA Plugin for OpenVINO Inference Engine Python* API'),
    long_description=get_description(config('WHEEL_OVERVIEW',
                                            f'{os.path.abspath(os.path.dirname(__file__))}/../README.md')),
    long_description_content_type='text/markdown',
    download_url=config('WHEEL_DOWNLOAD_URL', 'https://github.com/openvinotoolkit/openvino/tags'),
    url=config('WHEEL_URL', 'https://docs.openvinotoolkit.org/latest/index.html'),
    libraries=[(PACKAGE_NAME, {'sources': []})],
    packages=["openvino_nvidia"],
    package_dir={
        "openvino_nvidia": f"{os.path.abspath(os.path.dirname(__file__))}/packages/openvino_nvidia",
    },
    package_data=package_data,
    cmdclass={
        'build_clib': BuildCMakeLib,
        'install_lib': InstallLib,
    },
    zip_safe=False,
)
