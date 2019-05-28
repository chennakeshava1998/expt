import numpy as np
from TPM_from_VCS import config

def get_vector(m, v):
    a = np.zeros((config.MAX_NODES, config.MAX_VC_DIM))
    
    for i in range(0, v.shape[0]):
        for j in range(0, v.shape[1]):
            a[i, j] = v[i, j]

    return a