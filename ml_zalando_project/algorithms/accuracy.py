
import numpy as np

def confusion_matrix(y_true, y_pred):
    match = lambda t, p: len([_ for a, b in zip(y_true, y_pred) if a==t and b==p])
    true_dist = lambda t: np.array([match(t, p) for p in set(y_true)])
    return np.array([true_dist(t) for t in set(y_true)])

def one_vs_all(t, p, c):
    tp = lambda c: len([_ for a, b in zip(t, p) if a==b==c])
    tn = lambda c: len([_ for a, b in zip(t, p) if a!=c and b!=c])
    return (tp(c)+tn(c))/len(t)

def macro_accuracy(t, p):
    return np.mean([one_vs_all(t, p, c) for c in set(t)])

accuracy = macro_accuracy

