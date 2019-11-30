
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Import data
data = np.load("../../data/fashion_train.npy")
ds = np.array([x[:-1] for x in data])/255
labels = np.array([x[-1] for x in data])


# Calculating principal components
mean = sum(ds)/1000
X = np.array([ds[:, i]-mean[i] for i in range(ds.shape[1])]).T
C = np.matmul(np.transpose(X), X)/1000

eig_vals, eig_vecs = np.linalg.eig(C)
sorted_eig_vecs = eig_vecs[:, eig_vals.argsort()[::-1]].real

projection = np.dot(X, sorted_eig_vecs[:, :3]).T

# 3D plot
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(projection[0], projection[1], projection[2], c = labels, cmap='rainbow')
