
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce

na = lambda x: np.array(x)
apa = lambda func, a, x: np.apply_along_axis(func, a, x)

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
mean = lambda c: np.mean(c, axis=0)
prod = lambda x, m: np.outer((x-m), (x-m))
get_si = lambda c: reduce(lambda mat, x: mat + prod(x, mean(c)), c, np.zeros((784, 784)))

classes = np.array([get_class(i) for i in set(labels)])

Sw = np.sum([get_si(c) for c in classes], axis=0)
Sb = np.sum([len(c)*prod(mean(c), mean(ds)) for c in classes], axis=0)

eig_vals, eig_vecs = np.linalg.eig(np.dot(np.linalg.inv(Sw), Sb))

# t = eig_vecs[:, eig_vals.argsort()[::-1]].T
t = eig_vecs[:, eig_vals.argsort()[::-1]].T

# projection = np.dot(ds, t[:3, :].T.real)
# projection = np.dot(ds, t[:3, :].T.real)

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(projection[:,0], projection[:,1], projection[:,2], c = labels)

# -------------------------------------


def showimg(imvec):
    plt.figure(figsize=(3,3))
    plt.imshow(imvec.reshape(28, 28), cmap="gray") 

mean = lambda c: np.apply_along_axis(np.mean, 0, c)

test = np.take(ds, np.argwhere(labels == 0), axis=0)
showimg(test)

test2 = test.reshape(2033, 784)

map(get_class, set(labels))

# classes = [[np.array(ds[j])[0] for j in np.argwhere(labels == i)] for i in range(5)]
# means = [np.mean(a, axis=0)[0] for a in classes]

# get_si = lambda c, m: np.sum([np.outer((x-m), (x-m)) for x in c])
# Sw = ([get_si(c, means[i]).shape for i, c in enumerate(classes)])
# Sw

# np.array(classes[0]).shape


# ([(np.outer((a-means[0]), (a-means[0]))) for a in classes[0]])

# [np.dot(np.reshape((classes[0][2]-means[0]), (784, 1)), np.reshape((classes[0][2]-means[0]), (784, 1)).T) for _ in range(20000)]

