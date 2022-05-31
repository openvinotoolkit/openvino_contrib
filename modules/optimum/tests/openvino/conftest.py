import docker
from datetime import datetime
import time

from optimum.intel.openvino import OVAutoModel

def start_ovms_with_single_model(hf_model_name, ovms_model_name, model_class=OVAutoModel, **kwargs):

    tmp_dir_name = datetime.strftime(datetime.now(), "optimum_tests_%m%d%Y%H%M%S")
    tmp_model_dir = f"/tmp/{tmp_dir_name}"
    tmp_model_version_dir = f"{tmp_model_dir}/1"

    model = model_class.from_pretrained(
            hf_model_name, **kwargs
        )
    model._load_network()
    model.save_pretrained(tmp_model_version_dir)

    docker_client = docker.from_env()
    ovms_container = docker_client.containers.run(image="openvino/model_server:latest",
                                                  command=f"/ovms/bin/ovms --model_name {ovms_model_name} --model_path /model --port 9000",
                                                  ports= {"9000/tcp": 9000},
                                                  volumes={tmp_model_dir: {'bind': '/model', 'mode': 'ro'}},
                                                  detach=True,
                                                  auto_remove=True)
    max_retries = 5
    retry_counter = 0
    while retry_counter < max_retries:
        if "Server started on port" in ovms_container.logs().decode():
            break
        time.sleep(1)
    return ovms_container, tmp_model_dir
