import unittest
import numpy as np
import torch
import torchvision.models as models

from openvino.inference_engine import IECore
import mo_pytorch

class TestModels(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.ie = IECore()


    def check_torchvision_model(self, model_func, size):
        inp_size = [1, 3, size[0], size[1]]

        # Create model
        model = model_func(pretrained=True, progress=False)
        inp = torch.randn(inp_size)
        model.eval()
        ref = model(inp)

        # Convert to OpenVINO IR
        mo_pytorch.convert(model, input_shape=inp_size, model_name='model')

        # Run model with OpenVINO and compare outputs
        net = self.ie.read_network('model.xml', 'model.bin')
        exec_net = self.ie.load_network(net, 'CPU')
        out = exec_net.infer({'input': inp.detach().numpy()})

        if isinstance(ref, torch.Tensor):
            ref = {'': ref}
        for out0, ref0 in zip(out.values(), ref.values()):
            diff = np.max(np.abs(out0 - ref0.detach().numpy()))
            self.assertLessEqual(diff, 1e-5)


    def test_alexnet(self):
        self.check_torchvision_model(models.alexnet, (227, 227))

    def test_resnet18(self):
        self.check_torchvision_model(models.resnet18, (227, 227))

    def test_deeplabv3_resnet50(self):
        self.check_torchvision_model(models.segmentation.deeplabv3_resnet50, (240, 320))


if __name__ == '__main__':
    unittest.main()
