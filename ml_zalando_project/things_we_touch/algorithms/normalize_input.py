import numpy as np

ds = np.load("fashion_test.npy")

labels = [x[-1] for x in ds]
data = np.array([x[:-1] for x in ds])/255

np.save("labels_test.npy", data)
