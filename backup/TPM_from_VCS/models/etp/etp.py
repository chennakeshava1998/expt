import numpy as np
import tensorflow as tf
tf.enable_eager_execution()
# A and B are two (N by 2) matrices
def get_etp_without_rotation(A, B):
        n = tf.convert_to_tensor(A.get_shape())

        print("Type of n : {}".format(type(n)))

        # n = n[0]
        print('Values in n : {}'.format(n))

        
        count_invs = 0
        for i in range(0, n.numpy()):
                for j in range(0, n):

                        # check for inversions along X-axis
                        count_invs += (A[i, 0]<A[j, 0] and B[j, 0]<B[i, 0])
                        count_invs += (A[i, 0]>A[j, 0] and B[j, 0]>B[i, 0])

                        # check for inversions along Y-axis
                        count_invs += (A[i, 1]<A[j, 1] and B[j, 1]<B[i, 1])
                        count_invs += (A[i, 1]>A[j, 1] and B[j, 1]>B[i, 1])

        count_invs/=(n * (n - 1))
        count_invs *= 100
        return tf.constant(count_invs, dtype=tf.float32)

# A is true value
# B is predicted value
def get_best_etp(A, B):
        print('DEBUG: A Shape: {}'.format(A.shape))
        print('DEBUG: B Shape: {}'.format(B.shape))

        if A.shape != B.shape:
                print('Something is wrong!!\n')

        x = 0
        final_etp = 100
        for x in range(0, 360):
                rot_matrix = np.array([[np.cos(np.radians(x)), -np.sin(np.radians(x))], [np.sin(np.radians(x)), np.cos(np.radians(x))]])
                rot_matrix = tf.convert_to_tensor(rot_matrix, dtype=tf.float32)
                temp = get_etp_without_rotation(A, tf.matmul(B, rot_matrix))
                # temp = get_etp_without_rotation(A, B)
                final_etp = min(final_etp, temp)

                if x % 60 == 0:
                        print('Current best of count_invs = {} at Angle {}'.format(final_etp, x))

                
        return tf.constant(final_etp, dtype=tf.float32)
