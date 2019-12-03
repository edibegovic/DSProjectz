
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from functools import lru_cache, reduce
from sklearn.metrics import accuracy_score

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

projection.shape

# 3D plot of projected data
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(projection[0], projection[1], projection[2], c = labels, cmap='rainbow')


#------------- LDA classification -----------------

n = len(ds)
prior = lambda c: len(c)/n
inv_class_cov_est = np.linalg.inv(1/(n-len(classes)) * within_scat)
means = [mean(c) for c in classes]
priors = [prior(c) for c in classes]

def predict(x):
    x = x.reshape(784, 1)
    disc = [(np.matmul(np.matmul(x.T, inv_class_cov_est), means[i].T)
    - 0.5*np.matmul(np.matmul(means[i], inv_class_cov_est), means[i].T)
    + np.log(priors[i])) for i,c in enumerate(classes)]
    return np.argmax(disc)

# Testing data
data_test = np.load("../../data/test_data_no_touch/fashion_test.npy")
ds_test = np.reshape((np.array([x[:-1] for x in data_test], dtype = np.float32) / 255), (-1, 784))
labels_test = np.array([x[-1] for x in data_test])

pred = [predict(x) for x in ds_test]
accuracy_score(labels_test, pred)

# Functional, but super inefficient impplementation
# def predict(x):
#     x = x.reshape(784, 1)
#     disc = [(np.matmul(np.matmul(x.T, np.linalg.inv(class_cov_est)), mean(c).T)
#     - 0.5*np.matmul(np.matmul(mean(c), np.linalg.inv(class_cov_est)), mean(c).T)
#     + np.log(prior(c))) for c in classes]
#     return np.argmax(disc)
