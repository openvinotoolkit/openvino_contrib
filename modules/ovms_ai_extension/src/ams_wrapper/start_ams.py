#!/usr/bin/env python3
#
# Copyright (c) 2021 Intel Corporation
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

import argparse
import json
import subprocess
import sys
import os

def validate_input(inputType, input):
    """Validate the user input for security checks
    """
    if inputType=="port":
        if input.isdecimal() and int(input) >= 2000 and int(input) <= 50000:
            return input
        else:
            sys.exit("Invalid port number")

    elif inputType=="workers": 
        if input.isdecimal() and int(input) <= 100:
            return input
        else:
            sys.exit("Invalid AMS service workers. number should be less than 100")

    elif inputType=="ovms_model_devices":
        ovms_config_path = '/opt/ams_models/ovms_config.json'
        with open(ovms_config_path, mode='r') as ovms_config_file:
            ovms_config = json.load(ovms_config_file)
            ovms_config_model_list=[]
            devices_config_model_list=[]
            devices_config_list=[]
            supported_device_list= {'CPU', 'GPU', 'MYRIAD', 'HDDL'}
            for model_config in ovms_config.get('model_config_list'):
                model_config = model_config['config']
                ovms_config_model_list.append(model_config['name'])
            for model_name in input:
                devices_config_model_list.append(model_name)    
                devices_config_list.append(input[model_name])
        if set(devices_config_model_list).issubset(set(ovms_config_model_list)) and set(devices_config_list).issubset(supported_device_list):
            return input
        else:
            sys.exit('Invalid model devices config')


def execute_subprocess(cmd, dir_path=None):
    """Execute input list of commands as subprocess"""
    try:
        if (dir_path==None):
            proc = subprocess.Popen(cmd)
            return proc
        else:
            proc = subprocess.Popen(cmd, cwd=dir_path)
            return proc

    except (subprocess.SubprocessError, OSError,  ValueError) as err:
        sys.exit(err)

def parse_ovms_model_devices_config(config: str) -> dict:
    """Extract target device for each model from user input"""
    if not config:
        return {}
    try:
        return {
            model_name: device for model_name, device in [item.split('=') for item in config.split(';')]
        }
    except Exception as e:
        print('Invalid model devices config: {}'.format(config))
        raise ValueError from e


def modify_ovms_config_json(devices_config: dict,
                            ovms_config_path: str = '/opt/ams_models/ovms_config.json'):
    """Update OVMS config with target devices for each model"""
    with open(ovms_config_path, mode='r') as ovms_config_file:
        ovms_config = json.load(ovms_config_file)
        for model_config in ovms_config.get('model_config_list'):
            model_config = model_config['config']
            if devices_config.get(model_config['name']):
                model_config['target_device'] = devices_config[model_config['name']]
    with open(ovms_config_path, mode='w') as ovms_config_file:
        json.dump(ovms_config, ovms_config_file)


def parse_args():
    parser = argparse.ArgumentParser(
        description='This script runs OpenVINO Model Server and AMS Service in the background. '
                    'OVMS will served models available under path /opt/models with configuration '
                    'defined in /opt/models/config.json file. ',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--ams_port', type=int, default=5000,
                        help='Port for AMS Service to listen on')
    parser.add_argument('--ovms_port', type=int, default=9000,
                        help='Port for OVMS to listen on')
    parser.add_argument('--workers', type=int, default=20,
                        help='AMS service workers')
    parser.add_argument('--ovms_model_devices', type=str,
                        help='Colon delimited list of model devices, '
                        'in following format: \'<model_1_name>=<device_name>;<model_2_name>=<device_name>\'',
                        default=os.environ.get('OVMS_MODEL_DEVICES', ''))
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    ovms_port_arg = str(args.ovms_port)
    ams_port_arg = str(args.ams_port)
    workers_arg = str(args.workers)
    model_devices_config_arg = args.ovms_model_devices
    ovms_model_devices_config = parse_ovms_model_devices_config(
        model_devices_config_arg)

    ovms_port = validate_input("port", ovms_port_arg)
    ams_port = validate_input("port", ams_port_arg)
    workers = validate_input("workers", workers_arg)
    ovms_model_devices = validate_input("ovms_model_devices", ovms_model_devices_config)

    if ovms_model_devices != {} and ovms_model_devices != None:
        modify_ovms_config_json(ovms_model_devices)

    ovms_process_command = ['/ovms/bin/ovms',
                                     '--config_path', '/opt/ams_models/ovms_config.json',
                                     '--port', ovms_port]

    ams_process_command = ['python3', '-m', 'src.wrapper', '--port',
                                    ams_port, '--workers', workers]

    ovms_process = execute_subprocess(ovms_process_command)
    ams_process = execute_subprocess(ams_process_command, '/ams_wrapper')
    retcodes = [ovms_process.wait(), ams_process.wait()]
    sys.exit(max(retcodes))




if __name__ == "__main__":
    main()
