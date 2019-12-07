
import numpy as np

k = 10
k_f = set(range(k))
ds = np.load("../../data/fashion_train.npy")
np.random.shuffle(ds)
folds = np.split(ds, k)

for fold in range(k):
    np.save(f"../../data/10_fold/{fold}_train.npy", np.take(folds, list(k_f-{fold}), axis=0).reshape(9000, 785).shape)
    np.save(f"../../data/10_fold/{fold}_test.npy", folds[fold])
