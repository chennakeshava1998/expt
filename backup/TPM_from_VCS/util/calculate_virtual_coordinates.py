
import numpy as np

def generate_physical_coordinates(n):
    phy_coord = np.random.rand(n, 2)
    return phy_coord

# function ge_VC will be called by external modules to calculate VC
def distance(x, y, dist_matrix):
    return np.sqrt((dist_matrix[x][0] - dist_matrix[y][0])**2 + (dist_matrix[x][1] - dist_matrix[y][1])**2) 
    
def is_neighbour(x, y, dist_matrix):
    return distance(x, y, dist_matrix) <= 1


# finding shortest paths based on Dynamic Programming
def get_shortest_path(source, dest, dist_matrix, VC_matrix, anchors):
    # print('DEBUG: Source = {}'.format(source))
    # print('DEBUG: Destination = {}'.format(dest))

    if VC_matrix[source][dest] != 100:
        return VC_matrix[source][dest]

    if source == dest:
        VC_matrix[source][dest] = 0
        VC_matrix[dest][source] = VC_matrix[source][dest]

        return VC_matrix[source][dest]


    # if these nodes are neighbours, then hop_dist =  1
    if(distance(source, anchors[dest], dist_matrix) <= 1):
        VC_matrix[source][dest] = 1
        VC_matrix[dest][source] = VC_matrix[source][dest]

        return VC_matrix[source][dest]

    # return 1 + min distance from source to a neighbour of destination
    for i in range(0, dist_matrix.shape[0]):
        if is_neighbour(anchors[dest], i, dist_matrix):
            VC_matrix[source][dest] = min(VC_matrix[source][dest], 1 + VC_matrix[source][i])

    # symmetric distances
    VC_matrix[dest][source] = VC_matrix[source][dest]
    return VC_matrix[source][dest]

# dist_matrix is the matrix of geographical coordinates: Number_of_nodes X 2
# anchors is the list of anchor nodes: Number_of_anchors X 1

def calculate_shortest_hops(dist_matrix, anchors):
    VC_matrix = 1000 * np.ones((dist_matrix.shape[0], len(anchors)))

    # print('DEBUG: dist_matrix {}'.format(dist_matrix.shape))
    # print('DEBUG: VC_matrix {}'.format(VC_matrix.shape))

    for i in range(0, dist_matrix.shape[0]):
        for j in range(0, len(anchors)):
            VC_matrix[i][j] = get_shortest_path(i, j, dist_matrix, VC_matrix, anchors)

    return VC_matrix

def select_anchor_nodes(dist_matrix):
    return np.random.randint(low=0, high=dist_matrix.shape[0], size=(5, 1, 1))
    

def get_VC(num_of_nodes):
    # print('get_VC: Need to create {} nodes'.format(num_of_nodes))
    dist_matrix = generate_physical_coordinates(num_of_nodes)
    anchors = select_anchor_nodes(dist_matrix)
    return dist_matrix, calculate_shortest_hops(dist_matrix, anchors)
