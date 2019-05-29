import numpy as np
import os

import datetime
from TPM_from_VCS.util import tpm_from_vcs as get_TPM
from TPM_from_VCS.util import broadcast_vector as BV
from TPM_from_VCS import config
from TPM_from_VCS.util import calculate_virtual_coordinates as calc_VCS


# returns the physical coordinates and virtual coordinates.
def create_dataset(size):

    phy_coordinates, vcs = calc_VCS.get_VC(np.random.randint(low=10, high=1000))

    tcs = get_TPM.generate_tpm_from_vcs(vcs)

    
    # broadcast the input vector into MAX_VC_DIM length
    vcs = BV.get_vector(config.MAX_VC_DIM, vcs)
    tcs = BV.get_vector(config.MAX_VC_DIM, tcs)

    X = np.array(vcs)
    y = np.array(tcs)
    PC = np.array(phy_coordinates)
    
    
    for i in range(0, size):

        phy_coordinates, vcs = calc_VCS.get_VC(np.random.randint(low=10, high=1000))

        tcs = get_TPM.generate_tpm_from_vcs(vcs)

        
        # broadcast the input vector into MAX_VC_DIM length
        vcs = BV.get_vector(config.MAX_VC_DIM, vcs)
        tcs = BV.get_vector(config.MAX_VC_DIM, tcs)

        X = np.append(X, vcs, axis=0)
        y = np.append(y, tcs, axis=0)
        PC = np.append(PC, phy_coordinates, axis=0)
        # y.append(tcs)
        # PC.append(phy_coordinates)
        
        if i % 50 == 0:
            print('Created {} records in the dataset\n'.format(i))
        
    
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), PC

def save_dataset(dataset, filename):
    np.save(os.path.join(config.dataset_dir, filename + '.npy'), dataset)
