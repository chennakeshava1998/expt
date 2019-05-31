import numpy as np
import os

import datetime
from TPM_from_VCS.util import tpm_from_vcs as get_TPM
from TPM_from_VCS.util import broadcast_vector as BV
from TPM_from_VCS import config
from TPM_from_VCS.util import calculate_virtual_coordinates as calc_VCS


# returns the physical coordinates and virtual coordinates.
def create_dataset(size):

    
    for i in range(0, size):

        phy_coordinates, vcs = calc_VCS.get_VC(np.random.randint(low=10, high=1000))

        # dataset = np.array(((vcs[0]), (phy_coordinates[0])), dtype=np.float32)

        dataset = [(vcs[0], phy_coordinates[0])]
        # dataset = np.array

        for i in range(1, phy_coordinates.shape[0]):
            dataset.append(((vcs[i]), (phy_coordinates[i])))


        # dataset.append((vcs, phy_coordinates))
        
        if i % 50 == 0:
            print('Created {} records in the dataset\n'.format(i))
    
    return dataset

def save_dataset(dataset, filename):
    np.save(os.path.join(config.dataset_dir, filename + '.npy'), dataset)
