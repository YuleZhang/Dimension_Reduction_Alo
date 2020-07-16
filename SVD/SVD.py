import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import eig, svd
from PIL import Image


# 保留前 k 个奇异值
def compression(image, k):
    image2 = np.zeros_like(image)
    for i in range(image.shape[2]):
        # svd版本
        U, S, Vt = svd(image[:,:,i])
        image2[:,:,i] = U[:,:k].dot(np.diag(S[:k])).dot(Vt[:k,:])


    # eig_values,eig_vector = eig(image)
    # print('eig_vector[:,1]',Vt[:,1].flatten())
    # idx = eig_values.argsort()[::-1][:k]
    # eigval = eig_values[idx]
    # eigvector = eig_vector[:,idx]
    # vi = eigvector/eigval.reshape(-1,k)
    # image2 = np.dot(image,vi)

    plt.imshow(image2)
    plt.title('k = %s' % k)

if __name__ == "__main__":
    path = 'lena.jpg'
    image = Image.open(path)
    image = np.array(image)
    print(np.shape(image))
    plt.imshow(image)
    plt.show()

    plt.figure(figsize=(20,10))

    k = image.shape[0]/2
    for i in range(8):
        k = int(k/2)
        pos = 241+i
        plt.subplot(pos)
        compression(image, k)
    plt.show()
