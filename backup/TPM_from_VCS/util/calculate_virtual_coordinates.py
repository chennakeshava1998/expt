import generate_physical_coordinates as PC
import numpy as np

# function ge_VC will be called by external modules to calculate VC
def distance(x, y, dist_matrix):
    return np.sqrt((dist_matrix[x][0] - dist_matrix[y][0])**2 + (dist_matrix[x][1] - dist_matrix[y][1])**2) 
    
def is_neighbour(x, y, dist_matrix):
    return distance(x, y, dist_matrix) <= 1


# finding shortest paths based on Dynamic Programming
def get_shortest_path(source, dest, dist_matrix, VC_matrix, anchors):
    print('DEBUG: Source = {}'.format(source))
    print('DEBUG: Destination = {}'.format(dest))

    if VC_matrix[source][dest] != 100:
        return VC_matrix[source][dest]

    # if these nodes are neighbours, then hop_dist =  1
    if(distance(source, anchors[dest], dist_matrix) <= 1):
        return 1

    # return 1 + min distance from source to a neighbour of destination
    for i in range(0, dist_matrix.shape[0]):
        if is_neighbour(anchors[dest], i, dist_matrix):
            VC_matrix[source][dest] = min(VC_matrix[source][dest], 1 + VC_matrix[source][i])

    return VC_matrix[source][dest]

# dist_matrix is the matrix of geographical coordinates: Number_of_nodes X 2
# anchors is the list of anchor nodes: Number_of_anchors X 1

def calculate_shortest_hops(dist_matrix, anchors):
    VC_matrix = 1000 * np.ones((dist_matrix.shape[0], len(anchors)))

    print('DEBUG: dist_matrix {}'.format(dist_matrix.shape))
    print('DEBUG: VC_matrix {}'.format(VC_matrix.shape))

    for i in range(0, dist_matrix.shape[0]):
        for j in range(0, len(anchors)):
            VC_matrix[i][j] = get_shortest_path(i, j, dist_matrix, VC_matrix, anchors)
            # VC_matrix could be rectangular in nature. Hence this is not valid
            # VC_matrix[j][i] = VC_matrix[i][j]
    
    return VC_matrix

def select_anchor_nodes(dist_matrix):
    # select a random set of nodes (<1%) as anchors
    # num_of_anchors = 0.01 * dist_matrix.shape[0]
    return np.random.randint(low=0, high=dist_matrix.shape[0], size=(3, 1, 1))
    

def get_VC(num_of_nodes):
    dist_matrix = PC.generate_physical_coordinates(num_of_nodes)
    anchors = select_anchor_nodes(dist_matrix)
    return calculate_shortest_hops(dist_matrix, anchors)
