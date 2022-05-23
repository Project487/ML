import numpy as np


def convolve2D_valid(I, K, B):
    rows = I.shape[0] - K.shape[0] + 1
    cols = I.shape[1] - K.shape[1] + 1
    k = K.shape[0]
    Y = np.zeros((rows, cols), dtype=float)
    for i in range(0, Y.shape[0]):
        for j in range(0, Y.shape[1]):
            if len(B.shape) == 1:
                Y[i,j] = np.sum(I[i:i+k,j:j+k] * K) + B[i]
            if len(B.shape) == 2:
                Y[i,j] = np.sum(I[i:i+k,j:j+k] * K) + B[i][j]
    return Y

def zero_border(I, K, B):
    k = K.shape[0]
    b = B.shape[0]
    if np.atleast_3d(I) is I:
        depth = I.shape[0]
        rows = I.shape[1] + 2*(k -1)
        cols = I.shape[2] + 2*(k -1)
        M = np.zeros((depth, rows, cols), dtype=float)
        M[:,k-1:1-k,k-1:1-k] = I
        rows = B.shape[0] + 2*(k -1)
        cols = B.shape[1] + 2*(k -1)
        N = np.zeros((depth, rows, cols), dtype=float)
        N[:,b-1:1-b,b-1:1-b] = B
    else:
        rows = I.shape[0] + 2*(k -1)
        cols = I.shape[1] + 2*(k -1)
        M = np.zeros((rows, cols), dtype=float)
        M[k-1:1-k,k-1:1-k] = I
        N = np.zeros(b+2*(k-1), dtype=float)
        N[k-1:1-k] = B
    return M, N

def convolve2D_full(I, K, B):
    M, N = zero_border(I, K, B)
    Y = convolve2D_valid(M, K, N)
    return Y

def convolve3D_valid(I, K, B):
    depth = I.shape[0]
    rows = I.shape[1] - K.shape[1] + 1
    cols = I.shape[2] - K.shape[2] + 1
    Y = np.zeros((depth, rows, cols), dtype=float)
    for i in range(0, depth):
        Y[i]= convolve2D_valid(I[i], K[i], B[i])
    return Y

def convolve3D_full(I, K, B):
    M, N = zero_border(I, K, B)
    depth = M.shape[0]
    rows = M.shape[1] - K.shape[1] + 1
    cols = M.shape[2] - K.shape[2] + 1
    Y = np.zeros((depth, rows, cols), dtype=float)
    for i in range(0, depth):
        Y[i]= convolve2D_valid(M[i], K[i], N[i])
    return Y

####### TESTING  ################################################
#################################################################
if input == 0:
    B = np.array([2])

    I = np.array([[1, 1, 1],
        [2, 2, 2],
        [3, 3, 3]])

    K = np.array([[5, 5],
        [6, 6]])
##################################################################
if input == 1:
    B = np.array([[2, 3],
            [2, 3]])

    I = np.array([[1, 1, 1, 1],
        [2, 2, 2, 2],
        [3, 3, 3, 3],
        [4, 4, 4, 4]])

    K = np.array([[5, 5, 5],
        [6, 6, 6],
        [7, 7, 7]])

### 3D VALID ####### I and K same depth, B..  ########################
input = 2
if input == 2:
    I = np.array([[[1, 1, 1],
        [2, 2, 2],
        [3, 3, 3]],
        [[1, 1, 1],
        [2, 2, 2],
        [3, 3, 3]]])

    K = np.array([[[5, 5],
        [6, 6]],
        [[7, 7],
        [8, 8]]])

    B = np.array([[2, 3],
            [2, 3]])
    Y = convolve3D_valid(I, K, B)
    print("3D VALID=\n\r",Y)

### 3D FULL ############################################################
input = 3
if input == 3:
    I = np.array([[[1, 1, 1],
        [2, 2, 2],
        [3, 3, 3]],
        [[1, 1, 1],
        [2, 2, 2],
        [3, 3, 4]]])

    K = np.array([[[5, 5],
        [6, 6]],
        [[7, 7],
        [8, 8]]])

    # B = np.array([[2, 3],
    #         [2, 4]])

    B = np.array([[[5, 5],
        [6, 6]],
        [[7, 7],
        [8, 8]]])
    
    B = np.array([[[0, 0],
        [0, 0]],
        [[0, 0],
        [0, 0]]])
    Y = convolve3D_full(I, K, B)
    print("3D FULL=\n\r",Y)

####### 2D FULL  #######################################################
input = 4
if input == 4:
    I = np.array([[1, 1, 1],
        [2, 2, 2],
        [3, 3, 3]])

    K = np.array([[5, 5],
        [6, 6]])
    B = np.array([1, 2])
    Y = convolve2D_full(I, K, B)
    print("2D FULL=\n\r",Y)

####### 2D VALID 1 #######################################################
input = 5
if input == 5:
    I = np.array([[1, 1, 1],
        [2, 2, 2],
        [3, 3, 3]])
    K = np.array([[5, 5],
        [6, 6]])
    B = np.array([2, 3])
    Y = convolve2D_valid(I, K, B)
    print("2D VALID=\n\r",Y)

####### 2D VALID 2 #######################################################
input = 6
if input == 6:
    I = np.array([[1, 1, 1],
        [2, 2, 2],
        [3, 3, 3]])
    K = np.array([[7, 7],
        [8, 8]])
    B = np.array([2, 3])
    Y = convolve2D_valid(I, K, B)
    print("2D VALID=\n\r",Y)

