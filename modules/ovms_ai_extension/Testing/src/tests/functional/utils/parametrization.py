#
# Copyright (c) 2020-2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os

IMAGES_DIR = os.path.join(os.path.dirname(
    os.path.realpath(__file__)), '../fixtures/test_images')

def get_ports_prefixes():
    ports_prefixes = os.environ.get("PORTS_PREFIX", "90 55")
    grpc_ports_prefix, rest_ports_prefix = [
        port_prefix for port_prefix in ports_prefixes.split(" ")]
    return {"grpc_ports_prefix": grpc_ports_prefix,
            "rest_ports_prefix": rest_ports_prefix}


def get_tests_suffix():
    return os.environ.get("TESTS_SUFFIX", "default")


def get_ports_for_fixture(port_suffix):
    ports_prefixes = get_ports_prefixes()
    grpc_port = ports_prefixes["grpc_ports_prefix"]+port_suffix
    rest_port = ports_prefixes["rest_ports_prefix"]+port_suffix
    return grpc_port, rest_port

def object_detection_image_type() -> dict:
    images_dict = {
        "png_type": os.path.join(IMAGES_DIR, 'single_car_small.png'),
        "jpg_type": os.path.join(IMAGES_DIR, 'single_car_small.jpg'),
        "bmp_type": os.path.join(IMAGES_DIR, 'single_car_small.bmp')
    }
    return images_dict

def object_detection_image_size() -> dict:
    images_dict = {
        "small_size": os.path.join(IMAGES_DIR, 'single_car_small.jpg'),
        "medium_size": os.path.join(IMAGES_DIR, 'single_car_medium.jpg'),
        "large_size": os.path.join(IMAGES_DIR, 'single_car_large.png')
    }
    return images_dict