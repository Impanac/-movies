import numpy as np
from mf import MF


# A rating matrix with ratings from 5 users on 4 items
# zero entries are unknown values
R = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [1, 0, 0, 4],
    [0, 1, 5, 4],
])

# Perform training and obtain the user and item matrices 
mf = MF(R, K=2, alpha=0.1, beta=0.01, iterations=1000)
training_process = mf.train()
print(mf.P)
print(mf.Q)
print(mf.full_matrix())
