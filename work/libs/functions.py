import numpy as np
import numpy.linalg as LA


def sigmoid(x):
    return 1/(np.exp(-x) + 1)


def deriv_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

def g_up(t, s):
    return sigmoid(s) + deriv_sigmoid(s)*(t-s) + (t-s)**2/20

def g_lo(t, s):
    return sigmoid(s) + deriv_sigmoid(s)*(t-s) - (t-s)**2/20

def sample_wise_vec_mat_vec(A, x):
    r = (x[:,np.newaxis,:]@A@x[:,:,np.newaxis])
    return np.squeeze(r)

def _getAplus(A):
    eigval, eigvec = LA.eig(A)
    Q = np.matrix(eigvec)
    xdiag = np.matrix(np.diag(np.maximum(eigval, 0)))
    return Q*xdiag*Q.T

def _getPs(A, W=None):
    W05 = np.matrix(W**.5)
    return  W05.I * _getAplus(W05 * A * W05) * W05.I

def _getPu(A, W=None):
    Aret = np.array(A.copy())
    Aret[W > 0] = np.array(W)[W > 0]
    return np.matrix(Aret)

def nearPD(A, nit=10):
    n = A.shape[0]
    W = np.identity(n) 
# W is the matrix used for the norm (assumed to be identity matrix here)
# the algorithm should work for any diagonal W
    deltaS = 0
    Yk = A.copy()
    for k in range(nit):
        Rk = Yk - deltaS
        Xk = _getPs(Rk, W=W)
        deltaS = Xk - Rk
        Yk = _getPu(Xk, W=W)
    return np.array(Yk)


def sample_wise_outer_product(x, y):
    return x[:,:,np.newaxis]@y[:,np.newaxis,:]

def mat_vec(A, x): 
    return A@x


def vec_mat_vec(A, x):
    return x@A@x


def mean_outer_product(z1, z2): #(n, d)*(n, d)  vector_wise_dot->  (n, d, d)  mean-> (d, d)
    outer_product = z2[:, np.newaxis, :]*z1[:, :, np.newaxis]
    return np.mean(outer_product, axis = 0)

    
def outer_product():
    return 

