import matplotlib.pyplot as plt

# X is a (N by 2) matrix
def plot(X, heading):
    plt.xlabel('X-Coordinate')
    plt.ylabel('Y-Coordinate')
    plt.plot(X[:, 0], X[:, 1], 'ro')

    plt.title(heading)

    print('DEBUG: Input maatrix: ')
    print(X)

    plt.show()



    