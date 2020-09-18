import numpy as np
from matplotlib import pyplot as plt
from MotionModel import *


model = MotionModel()
file = open("robotdata1.log", "r")

X = [np.array([0, 0, 0])]
lines = file.readlines()
u_t0 = np.asarray(lines[0].split(" ")[1:4], dtype=np.float32)

for i in range(1, len(lines)):
	u_t1 = np.asarray(lines[i].split(" ")[1:4], dtype=np.float32)
	X.append(model.update(u_t0, u_t1, X[-1]))
	u_t0 = np.copy(u_t1)

X = np.asarray(X)
plt.plot(X[:, 0], X[:, 1])
plt.show()

print(X[-1] - X[0])
print(np.asarray(lines[-1].split(" ")[1:4], dtype=np.float32) - np.asarray(lines[0].split(" ")[1:4], dtype=np.float32))