import numpy as np
import post_process
import matplotlib.pyplot as plt
pts = np.array([[0.0, 0.0],
                [1.0, 2.0],
                [3.0, 1.0],
                [4.0, 3.0]])

num = 100
s = np.linspace(0.0, 1.0, num + 1).reshape(-1, 1)
q = post_process.bezier(pts, num)
qdot = post_process.bezier_diff(pts, num)
q2dot = post_process.bezier_diff2(pts, num)

plt.plot(pts[:, 0], pts[:, 1], marker='o')
plt.plot(q[:, 0], q[:, 1])

fig = plt.figure()
ax1 = fig.add_subplot(131)
ax1.plot(s, q[:, 0])
ax1.plot(s, q[:, 1])
ax2 = fig.add_subplot(132)
ax2.plot(s, qdot[:, 0])
ax2.plot(s, qdot[:, 1])
ax3 = fig.add_subplot(133)
ax3.plot(s, q2dot[:, 0])
ax3.plot(s, q2dot[:, 1])
plt.show()
