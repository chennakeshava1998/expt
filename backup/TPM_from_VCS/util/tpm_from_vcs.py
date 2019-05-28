import numpy as np

def generate_tpm_from_vcs(P):

    U, s, V = np.linalg.svd(P)
    S = np.zeros((U.shape[0], V.shape[0]))
    S[:len(s), :len(s)] = np.diag(s)
    
    # print("DEBUG INFO:")
    # print("P Shape {}".format(P.shape))
    # print("U Shape {}".format(U.shape))
    # print("s Shape {}".format(len(s)))
    # print("S Shape {}".format(S.shape))
    # print("V Shape {}".format(V.shape))

    # print("Are the decomposed matrices within tolerance : {}".format(np.allclose(P, np.dot(U, np.dot(S, V)))))

    # print("TPM:")
    TC = np.dot(U, S)[:, [1, 2]] # extracting columnss 2 and 3
    return TC

# P = np.random.randn(19, 23)
# generate_tpm_from_vcs(P)




