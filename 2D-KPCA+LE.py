import os
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import scipy
from scipy.spatial.distance import pdist, squareform
from sklearn import datasets,manifold
flag=0
def make_swiss_roll(n_samples=100, noise=0.0, random_state=None):
    # 生成瑞士卷数据集
    # if random_state is not None: np.random.seed(random_state)
    t = 1.5 * np.pi * (1 + 2 * np.random.rand(1, n_samples))
    x = t * np.cos(t)
    y = 83 * np.random.rand(1, n_samples)
    z = t * np.sin(t)
    X = np.concatenate((y, x, z))
    X += noise * np.random.randn(3, n_samples)
    X = X.T
    t = np.squeeze(t)
    return X, t

def read_img_mat(path):
    # 读取图片文件数据集
    img_data = []
    content_path = path
    for file in os.listdir(content_path):
        # 拼接完整文件路径
        cmp_path = os.path.join(content_path,file)
        I = Image.open(cmp_path)
        tmp_img = np.array(I.convert('L')) / 118
        img_data.append(tmp_img)
    data = np.array(img_data)
    return data

# 对应第二步，得到投影矩阵
def twoDPCA(imgs,n_dims):
    a,b,c = imgs.shape
    average = np.zeros((b,c)) # 求平均值矩阵
    for i in range(a):
        average += imgs[i,:,:]/(a*1.0)
    G_t = np.zeros((c,c))
    for j in range(a):
        img = imgs[j,:,:]
        tmp = img - average  # 均值归零
        G_t += np.dot(tmp.T,tmp)/(a*1.0) # 求协方差矩阵
    eig_values, eig_vector = np.linalg.eig(G_t)
    idx = eig_values.argsort()[::-1]  # 获取特征值降序索引
    eigval = eig_values[idx][:n_dims]
    eigvector = eig_vector[:,idx][:,:n_dims]
#     print('eigvector',eigvector.shape)
    data_n = [np.dot(emp, eigvector) for emp in imgs]
    return np.array(data_n)

def rbf_kpca(x, gamma = 20):
    sq_dists = pdist(x, 'sqeuclidean')
    # tmplen = len(sq_dists)
    mat_sq_dists = squareform(sq_dists)
    return np.exp(-gamma*mat_sq_dists)

def rbf_le(dist, t = 1.0):
    return np.exp(-(dist/t))

# 对应第三步，进行核主成分分析特征提取
# 搜索最佳参数，gamma=86
def KPCA(data,n_dims=10,kernel=rbf_kpca,gamma=10):
    data1 = np.reshape(data,(data.shape[0],-1))
    print("data1.shape",data1.shape)
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
    return data_n

def cal_pairwise_dist(x):
    sum_x = np.sum(np.square(x),1)
    dist = np.add(np.add(-2 * np.dot(x, x.T), sum_x).T, sum_x)
    return dist 

def cal_rbf_dist(x, n_neighbors = 15, t = 0.1):
    dist = cal_pairwise_dist(x)
    # dist[dist < 0] = 0
    n = dist.shape[0]
    rbf_dist = rbf_le(dist, t)
    W = np.zeros((n, n))
    for i in range(n):
        index_ = np.argsort(dist[i])[1:1+n_neighbors]
        W[i, index_] = rbf_dist[i, index_]
        W[index_, i] = rbf_dist[index_, i]

    return W

# 对应四五六步
# LE算法论文及实验效果见https://www2.imm.dtu.dk/projects/manifold/Papers/Laplacian.pdf
def LE(data, n_dims=10, n_neighbors=15, t=0.1):
    
    N = data.shape[0]
    W = cal_rbf_dist(data, n_neighbors, t)  # 构造权值矩阵
    D = np.zeros_like(W)
    for i in range(N):
        D[i,i] = np.sum(W[i])

    D_inv = np.linalg.inv(D)
    L = D - W  # 计算拉普拉斯矩阵L
    eig_values, eig_vector = np.linalg.eig(np.dot(D_inv,L))  # 根据公式4.16

    idx = eig_values.argsort()   
    eig_values = eig_values[idx]
    positive_start = 0  # 获取第一个非零特征值索引
    while eig_values[positive_start] < 1e-6:
        positive_start += 1
    idx = idx[positive_start:positive_start+n_dims]
    low_dim_value = eig_values[idx]
    low_dim_vector = eig_vector[:, idx][:,:n_dims]
    if flag==1:return low_dim_vector
    data_n = data.dot(np.dot(np.dot(low_dim_vector.T, D), low_dim_vector))
    # data_n = np.dot(data,low_dim_vector)
    return abs(data_n)

# 计算余弦相似度
def cosine_similarity(x,y):
    print('x为{0},type={1}'.format(x[0],type(x[0])))
    print('y为{0},type={1}'.format(y[0],type(y[0])))
    num = x.dot(y.T)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return num / denom

def visualizeLE():
    global flag
    flag=1
    X,Y = make_swiss_roll(n_samples=2000,random_state=3)

    fig = plt.figure(figsize=(12,6))
    ax1 = fig.add_subplot(131,projection = '3d')
    ax1.scatter(X[:,0], X[:,1], X[:,2], c=Y)
    ax1.title.set_text('First Plot'+str(1))
    ax2 = fig.add_subplot(132)
    X_ndim = LE(X, n_dims=2, n_neighbors=10, t=5)
    ax2.scatter(X_ndim[:,0], X_ndim[:, 1], c=Y)
    ax2 = fig.add_subplot(133)
    X_ndim = LE(X, n_dims=2,n_neighbors=10, t=15)
    ax2.scatter(X_ndim[:,0], X_ndim[:, 1], c=Y)
    flag=0
    plt.show()

def evaluate_TwoPCA(train, test):
    train_2dpca = twoDPCA(train,n_dims=10)
    test_2dpca = twoDPCA(test,n_dims=10)
    examples,row,col = train_2dpca.shape
    sim_avg = []
    for person in range(8):
        for te in range(5):
            # 遍历测试集
            avg = 0.0
            cur_test = test_2dpca[te,:,:] 
            sim_list = []  # 相似度集合
            for tr in range(5):
            # 遍历训练集
                avg = 0.0
                for r in range(row):
                    avg += abs(cosine_similarity(train_2dpca[tr,r,:],cur_test[r,:]))
                sim_list.append(avg / row)
            cur_test_sim = sum(sim_list)/len(sim_list)
            sim_avg.append(cur_test_sim)
            # print("第{0}张图片的相似度为{1}".format(te, str(cur_test_sim) ) )
        # print('第{0}个人相似度为{1}'.format(str(person),str(sum(sim_avg)/len(sim_avg))))
    print("2dpca算法识别率为%s"%str(sum(sim_avg)/len(sim_avg)))

def evaluate_kpca_le(train,test):
    result = KPCA(train,n_dims=10,gamma=1)
    result = LE(result,n_dims=10,n_neighbors=15,t=2)
    ans = cosine_similarity(abs(result[0]),abs(result[2]))
    print("kpca+le算法识别率为%s"%str(ans))

def evaluate_2dpca_kpca_le(train,test):
    train_alg = twoDPCA(train,n_dims=10)/1.7  # 修正参数进行放缩
    result = KPCA(train_alg,n_dims=10,gamma=1)
    result = LE(result,n_dims=10,n_neighbors=15,t=2)
    ans = cosine_similarity(abs(result[0]),abs(result[2]))
    print("2d+pca+le算法识别率为%s"%str(ans))

if __name__ == "__main__":
    # 可视化LE算法
    # visualizeLE()
    imgs_train = read_img_mat('data/train')
    imgs_test = read_img_mat('data/test/')
    evaluate_TwoPCA(imgs_train,imgs_test)
    evaluate_kpca_le(imgs_train,imgs_test)
    evaluate_2dpca_kpca_le(imgs_train,imgs_test)
