import etp
import numpy as np


def test1():
    A = np.random.randint(low=10, size=(70, 2))
    B = A
    print('TEST1 : Identical Arrays: Result of test1 = {}'.format(etp.get_best_etp(A, B)))
    # assert etp.get_best_etp(A, B) == 0, "test1 failed"


def test2():
    A = [[1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0]]

    A = np.asarray(A, dtype=np.float32)
    C = A[::-1]
    B = [[1, 0], [2, 0], [4, 0], [5, 0], [3, 0], [6, 0]]
    B = np.asarray(B, dtype=np.float32)

    print('TEST2 : = {}'.format(etp.get_best_etp(A, B)))
    print('TEST3 : = {}'.format(etp.get_best_etp(A, C)))
    # assert etp.get_best_etp(A, B) == 13.3333, "test2 failed"

    # assert etp.get_best_etp(A, C) == 0, "test3 failed"

    

def init():
    np.random.seed(23)
    test1() 
    test2()

init()
