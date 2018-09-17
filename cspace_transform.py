import numpy as np
from math import pi
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def calc_intersection_point(A, B, C, D):
    denominator = ((B[0] - A[0]) * (C[1] - D[1])
                   - (B[1] - A[1]) * (C[0] - D[0]))
    # If two lines are parallel,
    if abs(denominator) < 1e-6:
        return True, None, None
    AC = A - C
    r = ((D[1] - C[1]) * AC[0] - (D[0] - C[0]) * AC[1]) / denominator
    s = ((B[1] - A[1]) * AC[0] - (B[0] - A[0]) * AC[1]) / denominator
    # If the intersection is out of the edges
    if r < -1e-6 or r > 1.00001 or s < -1e-6 or s > 1.00001:
        return True, r, s
    # If the intersection exists,
    return False, r, s


def collision(p, q):
    collide = []
    for i in range(len(p) - 1):
        for j in range(len(q) - 1):
            is_valid, _, _ = calc_intersection_point(p[i], p[i + 1],
                                                     q[j], q[j + 1])
            collide.append(is_valid)
    return all(collide)


def forward_kinematics(p0, l1, l2, theta1, theta2):
    p1 = p0 + np.array([l1 * np.cos(theta1), l1 * np.sin(theta1)])
    p2 = p1 + np.array(
        [l2 * np.cos(theta1 + theta2), l2 * np.sin(theta1 + theta2)])
    p = np.vstack((p0, p1, p2))
    return p


def inverse_kinematics(a, b, l1, l2):

    def calc_theta1(theta2, l1, l2, cos2, a, b):
        A = l2 * s * cos2 + l1
        B = l2 * np.sin(theta2)
        sin1 = (A * b - B * a)/(A**2 + B**2)
        cos1 = (A * a + B * b)/(A**2 + B**2)
        theta1 = np.arctan2(sin1, cos1)
        return theta1

    theta_list1 = []
    theta_list2 = []
    slist = np.linspace(0.0, 1.0, 101)
    for s in slist[1:]:

        cos2 = (a**2 + b**2 - (l1**2 + (l2 * s)**2)) / (2.0 * l1 * l2 * s)

        if cos2 > 1:
            theta1 = None
            theta2 = None
        elif cos2 == 1:
            theta2 = np.array([0.0, 0.0])
            theta1 = calc_theta1(theta2, l1, l2, cos2, a, b)
        elif -1 < cos2 and cos2 < 1:
            theta2 = np.array([np.arccos(cos2), -np.arccos(cos2)])
            theta1 = calc_theta1(theta2, l1, l2, cos2, a, b)
        elif cos2 == -1:
            theta2 = np.array([-pi, -pi])
            theta1 = calc_theta1(theta2, l1, l2, cos2, a, b)
        elif cos2 < -1:
            theta1 = None
            theta2 = None

        if theta1 is not None:
            theta_list1.append([theta1[0], theta2[0]])
            theta_list2.append([theta1[1], theta2[1]])

    theta_list = theta_list1 + theta_list2

    return theta_list


# In[]:
# %%time
p0 = np.array([0.0, 0.0])
l1 = 6.0
l2 = 6.0

q0 = np.array([10.0, 10.0])
theta1 = -140 * pi / 180
theta2 = 30 * pi / 180
q = forward_kinematics(q0, l1, l2, theta1, theta2)

m = 36
n = 36
theta1_range = np.linspace(0.0, 2.0 * pi, m + 1)
theta2_range = np.linspace(0.0, 2.0 * pi, n + 1)

rslt = np.zeros([m + 1, n + 1])
for i, theta1 in enumerate(theta1_range):
    for j, theta2 in enumerate(theta2_range):
        p = forward_kinematics(p0, l1, l2, theta1, theta2)
        rslt[i, j] = collision(p, q)

# In[]:
fig = plt.figure(figsize=(8, 4))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.plot(q[:, 0], q[:, 1])
ax2.imshow(rslt)
plt.show()
