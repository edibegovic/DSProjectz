
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce

# importing data
data = np.load("../../data/fashion_train.npy")
ds = np.array([x[:-1] for x in data])/255
labels = np.array([x[-1] for x in data])


# ---------------- PCA ----------------

# calculating principal components
mean = sum(ds)/1000
X = np.array([ds[:, i]-mean[i] for i in range(ds.shape[1])]).T

C = np.matmul(np.transpose(X), X)/1000

e_val, e_vec = np.linalg.eig(C)
e_vec = [e_vec[i] for i in e_val.argsort()[::-1]]

Z = np.dot((e_vec[:3]), np.transpose(X))

# 3D plot
cols = { 0:'red', 1:'green', 2:'blue', 3:'yellow', 4:'black'}
col_map = [cols[i] for i in labels]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Z[2, :], Z[1, :], Z[0, :], color=col_map, s=5, alpha=0.8)
ax.view_init(30, 185)
plt.show()

# ---------------- LDA ----------------

get_class = lambda i: np.take(ds, np.argwhere(labels == i), axis=0)
classes = np.array([get_class(i) for i in set(labels)])

mean = lambda c: np.mean(c, axis=0)
prod = lambda x, m: np.outer((x-m), (x-m))
get_si = lambda c: reduce(lambda mat, x: mat + prod(x, mean(c)), c, np.zeros((784, 784)))

within_scat = np.sum([get_si(c) for c in classes], axis=0)
between_scat = np.sum([len(c)*prod(mean(c), mean(ds)) for c in classes], axis=0)

eig_vals, eig_vecs = np.linalg.eig(np.dot(np.linalg.inv(within_scat), between_scat))
sorted_eig_vecs = eig_vecs[:, eig_vals.argsort()[::-1]].real

projection = np.dot(ds, sorted_eig_vecs[:, :3]).T

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(projection[0], projection[1], projection[2], c = labels, cmap='rainbow')

# -------------------------------------

def showimg(imvec):
    plt.figure(figsize=(3,3))
    plt.imshow(imvec.reshape(28, 28), cmap="gray") 
