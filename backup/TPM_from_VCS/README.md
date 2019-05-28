# TPM_from_VCS
Generate Topology Preserving Maps from Virtual Coordinate Systems. Please refer to https://github.com/chennakeshava1998/notes-and-summaries/blob/master/notes/networks/virtual_coord/README.md if any abbreviations are not clear.<br><br>
### Description of some files: <br>
test_<> : Unit tests for specific functions <br>
calculate_virtual_coordinates.py : Uses Dynamic Programming to optimally find the nearest hop distances<br>
create_dataset.py : This file is used to synthetically generate training data <br>
dataset01 : A json file containing 1000 exmaples of training data that is (Virtual Coordinate, Topological Coordinate) pairs.<br>
linear_map.py : An attempt(most probably a failed one :P) to find a linear map between VC and Topological Coordinates.<br>
baseline.ipynb : Uses LSTM to map between the VC space and TC Space. <br>
