import numpy as np

def get_pads(padding):
    if hasattr(padding, "__len__"):
        pads = []
        for pad in padding:
            pads.append([pad,pad])
        return np.array([*pads, *pads], dtype=np.int64)
        
    elif isinstance(padding, int):
        pads = np.array([padding, padding], dtype=np.int64).reshape(1, 2)
        pads = np.repeat(pads, 2, axis=0)
        return np.array([[0, 0], [0, 0], *pads], dtype=np.int64)
