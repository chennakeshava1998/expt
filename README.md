# Topology Preserving Maps
**Input:** 1-hop neighbourhood information for every node, Geographic Coordinates of Perimeter Nodes.<br>
**Output:** Topological Coordinates of all the nodes, which closely resemble the geographic map.<br>
As an intermediate Step, this algorithm also produces the virtual coorinates of all the nodes in the network.

**How to run**:
1. Execute the `python3 calculate_virtual_coordinates.get_VC(num_of_nodes)` function in the file `backup/TPM_from_VCS/util/calculate_virtual_coordinates.py`. 
This function makes use of the other functions in defined in that file.

2. Then execute the function `generate_tpm_from_vcs(VC_Matrix)` in the file `backup/TPM_from_VCS/util/tpm_from_vcs.py`. This produces the Topological Cordinates as proposed in the below paper.


**Info about other folders:**
1. `backup/TPM_from_VCS/data/` contains the programs to generate synthetic datasets.
2. `backup/TPM_from_VCS/models/` contains different RNN models for generating Topological Coordinates.
3. `backup/TPM_from_VCS/plot_functions/` contains generic matplotlib funtions to visualise the output of the algorithm.

`*.meta` and `*.index` are tensorflow checkpoint files

The below files are works in progress:
1. `backup/pg_1.ipynb` : A Policy Gradient approach
2. `backup/ae-basic-Copy1.ipynb`: Auto Encoder Approach


## References
1. Dhanapala, Dulanjalie C., and Anura P. Jayasumana. "Topology preserving maps: extracting layout maps of wireless sensor networks from virtual coordinates." 
IEEE/ACM Transactions on Networking (TON) 22.3 (2014): 784-797.
