import numpy as np

# A and B are two (N by 2) matrices
def get_etp_without_rotation(A, B):
        n = A.shape[0]
        count_invs = 0

        for i in range(0, n):
                for j in range(0, n):

                        # check for inversions along X-axis
                        count_invs += (A[i][0]<A[j][0] and B[j][0]<B[i][0])
                        count_invs += (A[i][0]>A[j][0] and B[j][0]>B[i][0])

                        # if A[i][0]<A[j][0] and B[j][0]<B[i][0]:
                        #     print('DEBUG: Case 1 {} and {}'.format(i, j))

                        # if A[i][0]>A[j][0] and B[j][0]>B[i][0]:
                        #     print('DEBUG: Case 2 {} and {}'.format(i, j))

                        # check for inversions along Y-axis
                        count_invs += (A[i][1]<A[j][1] and B[j][1]<B[i][1])
                        count_invs += (A[i][1]>A[j][1] and B[j][1]>B[i][1])

        count_invs/=(n * (n - 1))
        count_invs *= 100
        return count_invs

def get_best_etp(A, B):
        x = 0
        final_etp = 100
        for x in range(0, 360):
                rot_matrix = np.array([[np.cos(np.radians(x)), -np.sin(np.radians(x))], [np.sin(np.radians(x)), np.cos(np.radians(x))]])
                # temp = get_etp_without_rotation(A, np.matmul(B, rot_matrix))
                temp = get_etp_without_rotation(A, B)
                final_etp = min(final_etp, temp)
                print('Current best of count_invs = {} at Angle {}'.format(final_etp, x))

                if x < 100:
                        print('The two arrays are not equal : {}\n\n'.format(A == np.matmul(B, rot_matrix)))

        return final_etp
