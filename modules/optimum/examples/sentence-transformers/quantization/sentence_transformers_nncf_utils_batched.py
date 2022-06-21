from typing import Any, Tuple, Dict, Union, List

from numpy import ndarray

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from nncf.torch.initialization import PTInitializingDataLoader

class NNCFSTDataLoader(PTInitializingDataLoader):
    def __init__(self, data_loader: DataLoader):
        self._data_loader = data_loader
    
    @property
    def batch_size(self):
        return self._data_loader.batch_size

    def __len__(self):
        return len(self._data_loader)

    def get_inputs(self, sample: Any) -> Tuple[Tuple, Dict]:
        return (sample[0][0]["input_ids"], sample[0][0]["attention_mask"]), {}

    def get_target(self, sample: Any):
        return (sample['label'])

class InputExample:
    """
    Structure for one input example with texts, the label and a unique id
    """
    def __init__(self, guid: str = '', texts: List[str] = None,  label: Union[int, float] = 0):
        """
        Creates one InputExample with the given texts, guid and label
        :param guid
            id for the example
        :param texts
            the texts for the example.
        :param label
            the label for the example
        """
        self.guid = guid
        self.texts = texts
        self.label = label

    def __str__(self):
        return "<InputExample> label: {}, texts: {}".format(str(self.label), "; ".join(self.texts))

