# -*- coding=utf-8 -*-
'''
本代码实现基于约化密度矩阵的K2DKPCA算法来处理图片数据（图片来源于ORL人脸数据库）
算法思路如下
（1）图片数据的量子态映射
（2）水平方向和垂直方向 相邻像素等距约化
（3）图片恢复后采用KPCA提取
（4）采用图片对比和准确率对比算法处理效果
'''
from scipy.spatial.distance import pdist, squareform
import tensornetwork as tn
import matplotlib.pyplot as plt
from PIL import Image
from sklearn import preprocessing
from functools import reduce
import numpy as np
import math
import datetime
import os

# 图片维度
ORG_ORG_IMG_SHAPE = ()

# 读取图片文件数据集
def readImgMat(path):
    '''
    path: 图片文件父目录
    '''
    global ORG_IMG_SHAPE
    img_data = []
    content_path = path
    for file in os.listdir(content_path):
        # 拼接完整文件路径
        cmp_path = os.path.join(content_path,file)
        I = Image.open(cmp_path).convert('L')
        ORG_IMG_SHAPE = np.array(I).shape
        vec_img = list(I.getdata()) # 将m*n的像素矩阵转为向量
        tmp_img = np.array(vec_img) / 255  # 像素值归一化
        img_data.append(tmp_img)
    data = np.array(img_data)
    return data

# 采用式3.6进行像素特征映射
def resetPixel(pixel):
    '''
    pixel: 待处理像素
    '''
    return [math.pow(math.cos(pixel*np.pi/2),2),math.pow(math.sin(pixel*np.pi/2),2)]

# 矩阵特征映射
def resetMatrix(pic_matrix: np.array) -> np.array:
    '''
    pic_matrix: 待映射矩阵
    '''
    n_samples, pixel = pic_matrix.shape
    new_mat = np.array([resetPixel(pixel) for sample in pic_matrix for pixel in sample])
    return new_mat.reshape(n_samples, -1,2)

# 矩阵相邻像素张量合并
def mergeTensor(pic_matrix: np.array,n_dims=2) -> np.array:
    '''
    pic_matrix: 待合并矩阵
    ndims: 截断维数
    '''
    # print('this',ORG_IMG_SHAPE)
    # print('this / 2',[i/2 for i in ORG_IMG_SHAPE])
    new_img_row = int(ORG_IMG_SHAPE[0] / 2)
    new_img_col = int(ORG_IMG_SHAPE[1] / 2)

    n_samples = len(pic_matrix) # 样本个数
    len_pixel = len(pic_matrix[0]) # 像素张量数目
    formal_pic = []
    for sample in range(n_samples):
        tmp_pix = []
        for i in range(0,len_pixel,2):
            product = np.kron(pic_matrix[sample][i],pic_matrix[sample][i+1]) # 对两个像素张量求张量积
            conj_product = np.matrix(product).H # 求张量积的复共轭
            density = np.dot(product.reshape(-1,1),conj_product.T) # 求出约化密度矩阵
            eig_values, eig_vector = np.linalg.eig(density) # 特征分解
            idx = eig_values.argsort()[::-1] # 获取特征值降序索引
            eigval = eig_values[idx][:n_dims] # 特征值列表
            eigvector = eig_vector[:,idx][:,:n_dims] # 最小特征值对应的特征向量,即U1,作为等距层
            ans = product.reshape(1,-1).dot(eigvector.A) # 将像素张量进行等距层投影变换1*4 · 4*2
            tmp_pix.append(ans)
        # cnt=0
        # print(np.array(tmp_pix).shape)
        new_pix = []
        for j in range(0,ORG_IMG_SHAPE[0],2):
            for i in range(new_img_col):
                product = np.kron(tmp_pix[j*new_img_col+i],tmp_pix[(j+1)*new_img_col+i])
                conj_product = np.matrix(product).H
                density = np.dot(product.reshape(-1,1),conj_product.T)
                eig_values, eig_vector = np.linalg.eig(density)
                idx = eig_values.argsort()[::-1]
                eigval = eig_values[idx][:n_dims]
                eigvector = eig_vector[:,idx][:,:n_dims]
                ans = product.reshape(1,-1).dot(eigvector.A)
                new_pix.append(ans)
        # print('cnt',cnt)
        formal_pic.append(new_pix)
        
    transform_pic = np.array(formal_pic).reshape(n_samples,int(len_pixel/4),-1)
    return transform_pic

# 恢复过程是将（example,N/2,2）维的数据重新恢复为(example,N/2)即原始像素格式
def pixelRecovery(pic_matrix:np.array) -> np.array:
    '''
    pic_matrix: 待映射矩阵
    '''
    # 这一过程基于cosx及sinx在（0,1）之间单调
    rcv_mat = []  # 定义恢复后的矩阵
    for example in pic_matrix:
        acos = [pixel[0] for pixel in example]
        tmp_rcv = np.arccos(acos)*2/np.pi
        rcv_mat.append(tmp_rcv)
    print('经过等距层张量合并将{0}的张量转化为{1}的形式，实现了降维'.format(pic_matrix.shape,np.array(rcv_mat).shape))
    return np.array(rcv_mat).real

# 比较处理前后的图片信息
def cmpImg(org_mat,prep_mat):
    '''
    org_mat: 原图矩阵
    prep_mat: 待比较矩阵
    '''
    # 将粗粒化处理后的图片显示出来
    plt.figure(figsize=(20,10))
    for i in range(0,8,2):
        pos = 241+i
        plt.subplot(pos)
        plt.imshow(org_mat[i].reshape(ORG_IMG_SHAPE)*255,cmap ='gray')
        pos = 241+i+1
        plt.subplot(pos)
        plt.imshow(prep_mat[i].reshape(56,46)*255,cmap ='gray')
    # plt.figure(figsize=(20,10))
    # plt.imshow(org_mat[0].reshape(ORG_IMG_SHAPE)*255,cmap ='gray')
    # plt.show()
    # plt.imshow(prep_mat[0].reshape(int(ORG_IMG_SHAPE[0]/2),int(ORG_IMG_SHAPE[1]/2))*255,cmap ='gray')
    plt.show()

# 高斯核函数
def rbf(x, gamma = 1):
    sq_dists = pdist(x, 'sqeuclidean')
    mat_sq_dists = squareform(sq_dists)
    # 注意这里要对距离向量归一化（Normalization）
    return np.exp(-gamma*mat_sq_dists*2/np.linalg.norm(mat_sq_dists))

def myrbf(x, gamma):
    sq_dists = []
    for i in range(len(x)):
        for j in range(i+1,len(x)):
            # 遍历图片像素张量
            cur_dis = 0
            for idx in range(len(x[0])):
                cur_dis += (x[j][idx][0] - x[i][idx][0])**2 + (x[j][idx][1] - x[i][idx][1])**2 # 求张量模长
            sq_dists.append(cur_dis / len(x[0]))
    mat_sq_dists = squareform(sq_dists)
    # print('mat.shape',mat_sq_dists.shape)
    return np.exp(-gamma*mat_sq_dists)

# 核主成分分析特征提取
def KPCA(data,n_dims=10,kernel=rbf,gamma=0.1):
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

# 计算矩阵余弦相似度
def cosineSimilarity(x,y):
    num = abs(x).dot(abs(y.T))
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return num / denom

# 高斯核超参数搜索
def search_best_gamma(transform_data):
    best_gamma = -1
    best_result = 1
    start = 0.0
    gamma_list = []
    effect_list = []

    while start < 2:
        gamma_list.append(start)
        process_res = KPCA(transform_data, n_dims=3,gamma=start)
        effect = 0.0
        for i in range(1,5):
            effect += cosineSimilarity(process_res[0],process_res[i])/4
    
        print(start,"---",effect)
        effect_list.append(effect)
        if effect > best_result:
            best_gamma = start
            best_result = effect
        start += 0.02
    plt.plot(gamma_list,effect_list)
    plt.show()
    print('best_gamma%s'%best_gamma)
    print('best_effect%s'%best_result)

# 主函数入口
if __name__ == "__main__":
    imgs_train = readImgMat('data/train')[:20]
    # imgs_test = read_img_mat('F:/Python-Project/PCA_Series/data/test')
    print('读取了{0}张图片，并将像素矩阵转化为张量存储，张量大小为{1}'.format(imgs_train.shape[0],imgs_train.shape))

    # 定义预处理Pipeline，减少无效variable name
    transform_data = reduce(
        lambda value, function: function(value),
        (
            resetMatrix,
            mergeTensor,
            pixelRecovery
        ),
        imgs_train,
    )
    # 比较图片处理前后的效果
    # cmpImg(imgs_train,transform_data)
    starttime = datetime.datetime.now()
    res = KPCA(imgs_train,n_dims=2,gamma=0.1)
    kpcaTimeSpan = (datetime.datetime.now() - starttime).microseconds
    newAlgoStart = datetime.datetime.now()
    rds_res = KPCA(transform_data, n_dims=2,gamma=0.1)
    newAlgoTimeSpan = (datetime.datetime.now() - newAlgoStart).microseconds
    
    search_best_gamma(transform_data)
    ans0 = 0
    ans1 = 0
    ans2 = 0
    for i in range(1,5):
        ans0 += cosineSimilarity(imgs_train[0],imgs_train[i])/4
        ans1 += cosineSimilarity(res[0],res[i])/4
        ans2 += cosineSimilarity(rds_res[0],rds_res[i])/4
    print("KPCA，识别率为：{0}\t，处理耗时：{1}ms".format(ans1,kpcaTimeSpan))
    print("约化密度等距层处理+KPCA，识别率为：{0}\t，处理耗时：{1}ms".format(ans2,newAlgoTimeSpan))

