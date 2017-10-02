import numpy as np
la = np.linalg

words = ["i", "like", "enjoy", "deep", "learning","NLP", "flygin", "."]
x = np.array([[0,2,1,0,0,0,0,0],
     [2,0,0,0,0,1,0,0],
     [1,0,0,0,0,0,1,0],
     [0,1,0,0,1,0,0,0],
     [0,0,0,1,0,0,0,1],
     [0,1,0,0,0,0,0,1],
     [0,0,1,0,0,0,0,1],
     [0,0,0,0,1,1,1,0],
     ])

U, s, Vh = la.svd(x, full_matrices=False)

import matplotlib.pyplot as plt
for i in range(len(words)):
    plt.text(U[i, 0], U[i, 1], words[i])