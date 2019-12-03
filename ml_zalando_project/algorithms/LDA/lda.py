
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from functools import reduce

# Importing data
data = np.load("../../data/fashion_train.npy")
ds = np.array([x[:-1] for x in data])/255
labels = np.array([x[-1] for x in data])

# Calculating linear components
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
projection4 = np.dot(ds, sorted_eig_vecs[:, :4])

print(projection.T.shape)
print(projection4.shape)

np.save('sorted_eigvecs_3.npy', sorted_eig_vecs[:, :3])
np.save('sorted_eigvecs_4.npy', sorted_eig_vecs[:, :4])
#np.save('lda3_projected.npy', projection.T)
#np.save('lda4_projected.npy', projection4)

# 3D plot of projected data
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(projection[0], projection[1], projection[2], c = labels, cmap='rainbow')
