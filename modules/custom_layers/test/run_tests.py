# NOTE: import order is critical for now: extensions, openvino and only then numpy
from openvino_extensions import get_extensions_path
from openvino.inference_engine import IECore

import sys
import subprocess
import unittest
from pathlib import Path
import numpy as np
import mo_onnx

class TestLayers(unittest.TestCase):
    def convert_model(self):
        subprocess.run([sys.executable,
                        mo_onnx.__file__,
                        '--input_model=model.onnx',
                        '--extension', Path(__file__).absolute().parent.parent / 'mo_extensions'],
                       check=True)

    def run_test(self, convert_ir=True, threshold=1e-5):
        if convert_ir:
            self.convert_model()

        inputs = {}
        shapes = {}
        data = np.load('inp.npy')
        inputs['input'] = data
        shapes['input'] = data.shape

        ref = np.load('ref.npy')

        ie = IECore()
        ie.add_extension(get_extensions_path(), 'CPU')

        net = ie.read_network('model.xml', 'model.bin')
        net.reshape(shapes)
        exec_net = ie.load_network(net, 'CPU')

        out = exec_net.infer(inputs)
        out = next(iter(out.values()))

        diff = np.max(np.abs(ref - out))
        self.assertLessEqual(diff, threshold)


    def test_unpool(self):
        from unpool.export_model import export
        export('default', [5, 3, 6, 8])
        self.run_test()

    def test_unpool_reshape(self):
        from unpool.export_model import export
        export('dynamic_size', [5, 3, 6, 9])
        self.run_test()

        export('dynamic_size', [4, 3, 17, 8])
        self.run_test(convert_ir=False)


if __name__ == '__main__':
    unittest.main()
