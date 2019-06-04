import numpy as np
import tensorflow as tf
tf.enable_eager_execution()
# A and B are two (N by 2) matrices
def get_etp_without_rotation(A, B):

        # print('DEBUG - get_etp_without_rotation: Shape Matrix-A : {}'.format(A.shape))
        # print('DEBUG - get_etp_without_rotation: Shape Matrix-B : {}'.format(B.shape))

        n = tf.convert_to_tensor(A.get_shape()[0])

        count_invs = 0
        for i in range(0, n):
                for j in range(0, n):

                        # check for inversions along X-axis
                        count_invs += tf.cast((A[i, 0]<A[j, 0] and B[j, 0]<B[i, 0]), dtype=tf.int32)
                        count_invs += tf.cast((A[i, 0]>A[j, 0] and B[j, 0]>B[i, 0]), dtype=tf.int32)

                        # check for inversions along Y-axis
                        count_invs += tf.cast(A[i, 1]<A[j, 1] and B[j, 1]<B[i, 1], dtype=tf.int32)
                        count_invs += tf.cast(A[i, 1]>A[j, 1] and B[j, 1]>B[i, 1], dtype=tf.int32)

        count_invs/=(n * (n - 1))
        count_invs *= 100
        return tf.constant(count_invs, dtype=tf.float64)

# A is true value
# B is predicted value
def get_best_etp(A, B):
        A = tf.reshape(A, shape=(-1, 2))
        B = tf.reshape(B, shape=(-1, 2))


        x = 0
        final_etp = 100
        for x in range(0, 360):
                rot_matrix = np.array([[np.cos(np.radians(x)), -np.sin(np.radians(x))], [np.sin(np.radians(x)), np.cos(np.radians(x))]])
                rot_matrix = tf.convert_to_tensor(rot_matrix, dtype=tf.float64)
                # temp = get_etp_without_rotation(A, tf.matmul(B, tf.cast(rot_matrix, dtype=tf.float64)))
                temp = get_etp_without_rotation(A, tf.matmul(B, rot_matrix))

                final_etp = min(final_etp, temp)

                if x % 60 == 0:
                        print('Current best of count_invs = {} at Angle {}'.format(final_etp, x))

                
        return tf.constant(final_etp, dtype=tf.float32)
