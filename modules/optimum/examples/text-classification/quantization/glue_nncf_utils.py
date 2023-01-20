from typing import Any
from typing import Tuple
from typing import Dict

from torch.utils.data import DataLoader

from nncf.torch.initialization import PTInitializingDataLoader

class NNCFGLUEDataLoader(PTInitializingDataLoader):
    def __init__(self, data_loader: DataLoader):
        self._data_loader = data_loader
    
    @property
    def batch_size(self):
        return self._data_loader.batch_size

    def __len__(self):
        return len(self._data_loader)

    def get_inputs(self, sample: Any) -> Tuple[Tuple, Dict]:
        return (sample['input_ids'],sample['attention_mask']), {}

    def get_target(self, sample: Any):
        label = (sample['labels'])
        return label
