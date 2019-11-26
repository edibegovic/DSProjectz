
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

# importing data
ds = np.load("input_train.npy")
labels = np.load("labels_train.npy")


# ---------------- PCA ----------------

# calculating principal components
mean = sum(ds)/1000
X = np.array([ds[:, i]-mean[i] for i in range(ds.shape[1])]).T

C = np.matmul(np.transpose(X), X)/1000

e_val = np.linalg.eig(C)[0]
e_vec = np.linalg.eig(C)[1][0:3]

Z = np.dot((e_vec), np.transpose(X))


# 3D plot
cols = { 0:'red', 1:'green', 2:'blue', 3:'yellow', 4:'black'}
col_map = [cols[i] for i in labels]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Z[2, :], Z[1, :], Z[0, :], color=col_map, s=5, alpha=0.8)
ax.view_init(30, 185)
plt.show()

# ---------------- LDA ----------------

classes = [[np.array(ds[j])[0] for j in np.argwhere(labels == i)] for i in range(5)]
means = [np.mean(a, axis=0)[0] for a in classes]

get_si = lambda c, m: np.sum([np.outer((x-m), (x-m)) for x in c])
Sw = ([get_si(c, means[i]).shape for i, c in enumerate(classes)])
Sw

np.array(classes[0]).shape


([(np.outer((a-means[0]), (a-means[0]))) for a in classes[0]])

[np.dot(np.reshape((classes[0][2]-means[0]), (784, 1)), np.reshape((classes[0][2]-means[0]), (784, 1)).T) for _ in range(20000)]

#---------------------------------------

eigs = np.load("eig.npy")
e_vec = eigs[0:3]
Z = np.dot(e_vec, np.transpose(X))

# 3D plot
cols = { 0:'red', 1:'green', 2:'blue', 3:'yellow', 4:'black'}
col_map = [cols[i] for i in labels]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Z[2, :], Z[1, :], Z[0, :], color=col_map, s=5, alpha=0.8)
ax.view_init(30, 185)
plt.show()
