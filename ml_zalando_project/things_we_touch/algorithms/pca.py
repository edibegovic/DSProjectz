
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


