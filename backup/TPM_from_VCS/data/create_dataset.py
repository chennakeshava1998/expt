import numpy as np
import os

import datetime
from TPM_from_VCS.util import tpm_from_vcs as get_TPM
from TPM_from_VCS.util import broadcast_vector as BV
from TPM_from_VCS import config

def create_dataset(size):
    X = [] 
    y = []
    
    
    for i in range(0, size):
        # effectively 100 numbers in one-timestep is fed into rnn (m distances)
        m = np.random.randint(low=3, high=10)
        # number of anchors could be 1% of total nodes
        n = np.random.randint(low=m, high=100) 
        P = np.random.rand(n, m)

        ans = get_TPM.generate_tpm_from_vcs(P)

        if np.iscomplex(ans.flatten()).any():
            print("Is the TC matrix complex : {}".format(np.iscomplex(ans.flatten())))
        
        # broadcast the input vector into MAX_VC_DIM length
        P = BV.get_vector(config.MAX_VC_DIM, P)
        ans = BV.get_vector(config.MAX_VC_DIM, ans)

        X.append(P)
        y.append(ans)
        
        if i % 50 == 0:
            print('Created {} records in the dataset\n'.format(i))
        
    
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def save_dataset(dataset, filename):
    np.save(os.path.join(config.dataset_dir, filename + '.npy'), dataset)
