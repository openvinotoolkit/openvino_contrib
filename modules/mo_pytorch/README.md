# PyTorch extension for Model Optimizer

This module let you convert PyTorch models directly to OpenVINO IR without using ONNX

## Usage

1. Clone repository

    ```bash
    git clone --depth 1 https://github.com/openvinotoolkit/openvino_contrib
    ```

2. Setup environment

    ```bash
    source /opt/intel/openvino_2021/bin/setupvars.sh
    export PYTHONPATH=openvino_contrib/modules/mo_pytorch:$PYTHONPATH
    ```

3. Convert PyTorch model to OpenVINO IR

    ```python
    import torchvision.models as models

    # Create model
    model = models.alexnet(pretrained=True)

    # Convert to OpenVINO IR
    import mo_pytorch
    mo_pytorch.convert(model, input_shape=[1, 3, 227, 227], model_name='alexnet')
    ```

## Supported networks

* `torchvision.models.alexnet`
* `torchvision.models.resnet18`
* `torchvision.models.segmentation.deeplabv3_resnet50`
* `Detectron2 RetinaNet`
