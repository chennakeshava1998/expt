import datetime

from TPM_from_VCS.data.create_dataset import create_dataset 
from TPM_from_VCS.data.create_dataset import save_dataset 



if __name__ == '__main__':
    size = input('Enter the size of the dataset\n\n')
    filename = str(datetime.datetime.now()) + '_' + str(size)
    dataset = create_dataset(int(size))
    
    
    
    save_dataset(dataset, filename)
    print('Saved the dataset in {}.npy file\n'.format(filename))
