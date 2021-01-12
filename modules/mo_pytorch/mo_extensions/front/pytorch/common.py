import numpy as np

def get_pads(module):
    if hasattr(module.padding, "__len__"):
        pads = []
        for pad in module.padding:
            pads.append([pad,pad])
        return np.array([*pads, *pads], dtype=np.int64)
        
    elif isinstance(module.padding, int):
        pads = np.array([module.padding, module.padding], dtype=np.int64).reshape(1, 2)
        pads = np.repeat(pads, 2, axis=0)
        return np.array([[0, 0], [0, 0], *pads], dtype=np.int64)

    else:
        raise Exception("Unsupported type of padding!")
