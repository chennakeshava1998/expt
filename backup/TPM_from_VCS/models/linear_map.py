import numpy as np
import generate_physical_coordinates
import generate_virtual_coordinates
import tpm_from_vcs as gen_tpm
import create_dataset
import json

import tensorflow as tf
import numpy as np
import os
import time




data = create_dataset.create_dataset(5)
# create_dataset.save_dataset(data)


# Model to predict the x values of the TPM.





