
import numpy as np

def generate_physical_coordinates(n):
    phy_coord = np.random.randn(n, 2)
    return phy_coord

# function ge_VC will be called by external modules to calculate VC
def distance(x, y, dist_matrix):
    return np.sqrt((dist_matrix[x, 0] - dist_matrix[y, 0])**2 + (dist_matrix[x, 1] - dist_matrix[y, 1])**2) 
    
def is_neighbour(x, y, dist_matrix):
    return distance(x, y, dist_matrix) <= 1


# finding shortest paths based on Dynamic Programming
def get_shortest_path(source, dest, dist_matrix, VC_matrix):

    # return 1 + min distance from source to a neighbour of destination
    for _ in range(0, dist_matrix.shape[0]):
        for i in range(0, dist_matrix.shape[0]):
            if is_neighbour(source, i, dist_matrix):
                VC_matrix[source][dest] = min(VC_matrix[source][dest], 1 + get_shortest_path(i, dest, dist_matrix, VC_matrix))


    return VC_matrix[source][dest]

# dist_matrix is the matrix of geographical coordinates: Number_of_nodes X 2
# anchors is the list of anchor nodes: Number_of_anchors X 1

def calculate_shortest_hops(dist_matrix, anchors):
    VC_matrix = 1000 * np.ones((dist_matrix.shape[0], len(anchors)))

    for temp in range(len(anchors)):
        VC_matrix[anchors[temp]][temp] = 0

    for source in range(dist_matrix.shape[0]):
        for dest in range(len(anchors)):

            if VC_matrix[source][dest] > 0:
                # if these nodes are neighbours, then hop_dist =  1
                if(distance(source, anchors[dest], dist_matrix) <= 1):
                    VC_matrix[source][dest] = 1
                    # VC_matrix[dest][source] = VC_matrix[source][dest]

    # print('Prelim investigation done: {}'.format(VC_matrix))

    while not processed_all_nodes(VC_matrix, dist_matrix, anchors) and baked_rice(VC_matrix, dist_matrix, anchors):
        for source in range(0, dist_matrix.shape[0]):
            for dest in range(0, len(anchors)):
                for i in range(dist_matrix.shape[0]):
                    if is_neighbour(source, i, dist_matrix):
                        VC_matrix[source][dest] = min(VC_matrix[source][dest], 1 + VC_matrix[i][dest])

    print('Completed VC processing')
    return VC_matrix

def baked_rice(VC_matrix, dist_matrix, anchors):
    for source in range(0, dist_matrix.shape[0]):
            for dest in range(0, len(anchors)):
                if VC_matrix[source][dest] == 1000:
                    for i in range(dist_matrix.shape[0]):
                        if is_neighbour(source, i, dist_matrix):

                            if VC_matrix[source][dest] > 1 + VC_matrix[i][dest]:
                                VC_matrix[source][dest] = 1 + VC_matrix[i][dest]
                                return 0

    return 1

def processed_all_nodes(VC_matrix, dist_matrix, anchors):
    ''' Function to verify if processing is complete '''

    for source in range(0, dist_matrix.shape[0]):
            for dest in range(0, len(anchors)):
                for i in range(dist_matrix.shape[0]):
                    if is_neighbour(source, i, dist_matrix):

                        if VC_matrix[source][dest] > 1 + VC_matrix[i][dest]:
                            VC_matrix[source][dest] = 1 + VC_matrix[i][dest]
                            return 0

    return 1



def select_anchor_nodes(dist_matrix):
    anchor_list = np.random.choice(range(20), 5, replace=False)

    print('Anchor List: {}'.format(anchor_list))
    return anchor_list

def get_VC(num_of_nodes):
    # print('get_VC: Need to create {} nodes'.format(num_of_nodes))
    dist_matrix = generate_physical_coordinates(num_of_nodes)
    print('dist_matrix shape : {}'.format(dist_matrix))
    anchors = select_anchor_nodes(dist_matrix)
    return dist_matrix, calculate_shortest_hops(dist_matrix, anchors)
