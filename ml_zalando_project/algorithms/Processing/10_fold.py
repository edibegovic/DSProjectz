
import numpy as np
from sklearn.model_selection import KFold

ds = np.load("../../data/original_files/fashion_train.npy")

labels = [x[-1] for x in ds]
data = np.array([x[:-1] for x in ds])/255

cv = KFold(n_splits=10, random_state=42, shuffle=True)
idx = 0
for train_index, test_index in cv.split(labels):
    np.save(f"../../data/10_fold/{idx}_train_input.npy", [data[i] for i in train_index])
    np.save(f"../../data/10_fold/{idx}_train_label.npy", [labels[i] for i in train_index])
    np.save(f"../../data/10_fold/{idx}_test_input.npy", [data[i] for i in test_index])
    np.save(f"../../data/10_fold/{idx}_test_label.npy", [labels[i] for i in test_index])
    idx += 1


