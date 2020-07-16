import numpy as np
from numpy.linalg import svd
from PIL import Image
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform


def rbf(x, gamma = 1):
    print(x.shape)
    sq_dists = pdist(x, 'sqeuclidean')
    mat_sq_dists = squareform(sq_dists)
    return np.exp(-gamma*mat_sq_dists)

def KPCA(data,n_dims=10,kernel=rbf,gamma=0.1):
#     data1 = np.reshape(data,(data.shape[0],-1))
    data1 = data
    K = kernel(data1,gamma)
    N = K.shape[0]
    one_n = np.ones((N,N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
    eig_values, eig_vector = np.linalg.eig(K)
    idx = eig_values.argsort()[::-1]
    eigval = eig_values[idx][:n_dims]
    eigvector = eig_vector[:,idx][:,:n_dims]
    eigval = eigval ** (1/2)
    vi = eigvector/eigval.reshape(-1,n_dims)
    data_n = np.dot(K,vi)
    print("KPCA处理过后数据维度为{0}".format(data_n.shape))
    return data_n

if __name__ == "__main__":
    path = r'data//train//orl_001_001.bmp'
    image = Image.open(path)
    image = np.array(image)
    print(np.shape(image))
    plt.imshow(image)
    plt.show()
    res = KPCA(image,n_dims=3,gamma=0.9)
    plt.imshow(image2)
    plt.title('k = %s' % k)