import numpy as np
from numpy import cos, sin
from matplotlib import pyplot as plt
import time

def get_points_coord(X, scan):
	rot = np.arange(-np.pi / 2, np.pi / 2, np.pi / 180)
	scan_points = []
	for i in range(0, len(scan)):
		ang = X[2] + rot[i] 
		x = X[0] + scan[i] * cos(ang)
		y = X[1] + scan[i] * sin(ang) 
		scan_points.append([x, y])
	return scan_points


from MotionModel import *


model = MotionModel()
file = open("robotdata1.log", "r")

X = [np.array([0, 0, 0])]
lines = file.readlines()
u_t0 = np.asarray(lines[0].split(" ")[1:4], dtype=np.float32)
scan_points = [np.array([0, 0])]

# plt.ion()
plt.show()
 
axes = plt.gca()
axes.set_xlim(-300, 1500)
axes.set_ylim(-1500, 1500)
line, = axes.plot(np.asarray(X)[:, 0], np.asarray(X)[:, 1])
scan, = axes.plot(np.asarray(scan_points)[:, 0], np.asarray(scan_points)[:, 1], 'ro', markersize=2)


for i in range(1, len(lines)):
	try:
		print(i)
		u_t1 = np.asarray(lines[i].split(" ")[1:4], dtype=np.float32)
		X.append(model.update(u_t0, u_t1, X[-1]))
		# for p in scan_points:
		# 	print(p)
		# 	p = [p[0], p[1], -X[-1][2] + np.pi / 2]
		# 	p = model.update(u_t0, u_t1, p)
		# 	print(p)
		if len(lines[i].split(" ")) > 5:
			laser_pos = [X[-1][0] + 25 * cos(X[-1][2]), X[-1][1] + 25 * sin(X[-1][2]), X[-1][2]]
			scan_data = np.array(lines[i].split(" ")[7:187], dtype=np.float32)
			scan_points = get_points_coord(laser_pos, scan_data)
			# axes.set_xlim(-np.min(np.asarray(scan_points)[:, 0]), np.max(np.asarray(scan_points)[:, 0]))
			# axes.set_ylim(-np.min(np.asarray(scan_points)[:, 1]), np.max(np.asarray(scan_points)[:, 1]))
			scan.set_xdata(np.asarray(scan_points)[:, 0])
			scan.set_ydata(np.asarray(scan_points)[:, 1])
		u_t0 = np.copy(u_t1)
		line.set_xdata(np.asarray(X)[:, 0])
		line.set_ydata(np.asarray(X)[:, 1])
		plt.draw()
		plt.pause(1e-17)
	except KeyboardInterrupt:
		break

X = np.asarray(X)
scan_points = np.asarray(scan_points)

print(X[-1] - X[0])
print(np.asarray(lines[-1].split(" ")[1:4], dtype=np.float32) - np.asarray(lines[0].split(" ")[1:4], dtype=np.float32))