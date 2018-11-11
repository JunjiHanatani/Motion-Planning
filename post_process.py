import numpy as np
import math
from collision_checker import CollisionChecker
from scipy import interpolate


def interpolate_spline(pts, n=50):
    if len(pts) == 3:
        print('Only 3 points exist')
        xnew = (pts[0, 0] + pts[1, 0]) / 2.0
        ynew = (pts[0, 1] + pts[1, 1]) / 2.0
        pt_new = np.array([xnew, ynew])
        pts = np.insert(pts, 1, pt_new, axis=0)
    if len(pts) == 2:
        print('Only 2 points exist')
        xnew1 = (pts[0, 0] + pts[1, 0]) / 3.0
        ynew1 = (pts[0, 1] + pts[1, 1]) / 3.0
        xnew2 = (pts[0, 0] + pts[1, 0]) * 2.0 / 3.0
        ynew2 = (pts[0, 1] + pts[1, 1]) * 2.0 / 3.0
        pt_new = np.array([[xnew1, ynew1], [xnew2, ynew2]])
        pts = np.insert(pts, 1, pt_new, axis=0)
    if len(pts) == 1:
        print('Only 1 point exists')
    if len(pts) == 1:
        print('No points exist')

    tck, u = interpolate.splprep([pts[:, 0], pts[:, 1]], s=0)
    unew = np.linspace(0, 1.0, n + 1)
    out = interpolate.splev(unew, tck)
    return np.array(np.array(out).T)


def bezier(pts, num):
    def comb(n, r):
        return math.factorial(n) / (math.factorial(n - r) * math.factorial(r))

    s = np.linspace(0.0, 1.0, num + 1).reshape(-1, 1)
    n = len(pts) - 1
    sum = 0.0
    for i, p in enumerate(pts):
        sum = sum + p * comb(n, i) * s**i * (1 - s)**(n - i)
    return sum


def bezier_diff(pts, num):
    def comb(n, r):
        return math.factorial(n) / (math.factorial(n - r) * math.factorial(r))

    s = np.linspace(0.0, 1.0, num + 1).reshape(-1, 1)
    n = len(pts) - 1
    sum = 0.0
    for i, p in enumerate(pts):
        if i == 0:
            sum += - p * n * (1 - s)**(n - 1)
        elif i == n:
            sum += p * n * s**(n - 1)
        else:
            sum += p * comb(n, i) * (i * s**(i - 1) * (1 - s)**(n - i) -
                                     s**i * (n - i) * (1 - s)**(n - i - 1))
    return sum


def bezier_diff2(pts, num):
    def comb(n, r):
        return math.factorial(n) / (math.factorial(n - r) * math.factorial(r))

    s = np.linspace(0.0, 1.0, num + 1).reshape(-1, 1)
    n = len(pts) - 1
    sum = 0.0
    for i, p in enumerate(pts):
        if i == 0:
            sum += p * n * (n - 1) * (1 - s)**(n - 2)
        elif i == 1:
            sum += p * n * (- (n - 1) * (1 - s)**(n - 2)
                            - (n - 1) * (1 - s)**(n - 2)
                            + s * (n - 1) * (n - 2) * (1 - s)**(n - 3))
        elif i == n - 1:
            sum += p * n * (+ (1 - s) * (n - 1) * (n - 2) * s**(n - 3)
                            - (n - 1) * s**(n - 2)
                            - (n - 1) * s**(n - 2))
        elif i == n:
            sum += p * n * (n - 1) * s**(n - 2)
        else:
            sum += p * comb(n, i) * (
                + i * (i - 1) * s**(i - 2) * (1 - s)**(n - i)
                - i * s**(i - 1) * (n - i) * (1 - s)**(n - i - 1)
                - i * s**(i - 1) * (n - i) * (1 - s)**(n - i - 1)
                + s**i * (n - i) * (n - i - 1) * (1 - s)**(n - i - 2)
                )
    return sum


def resampling(pts, resampling_length):
    p0 = pts[0]
    res_position_list = [p0]
    for p in pts[1:]:
        p1 = p
        vector = p1 - p0
        dist = np.linalg.norm(vector)
        if dist > resampling_length:
            norm_vector = vector / dist
            num_sample = int(dist / resampling_length)
            dx = dist / num_sample
        else:
            norm_vector = vector
            num_sample = 1
            dx = 1.0
        for i in range(1, num_sample + 1):
            q_sample = p0 + norm_vector * dx * i
            res_position_list.append(q_sample)
        p0 = p1
    return np.array(res_position_list)


def shortcut(path_list, world):
    cc = CollisionChecker(world)
    q0 = np.array(path_list[0])
    res_path_list = [q0]
    i = 1
    while i < len(path_list):
        q1 = np.array(path_list[i])
        if cc.line_validation(q0, q1):
            i += 1
        else:
            q0 = np.array(path_list[i - 1])
            res_path_list.append(q0)
            i += 1
    res_path_list.append(path_list[-1])
    return np.array(res_path_list)


def ferguson_spline(path, num=None, s=None):
    '''
    Ferguson_spline
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.453.5654&rep=rep1&type=pdf
    http://markun.cs.shinshu-u.ac.jp/learn/cg/cg3/index4.html
    '''
    if s is None:
        s = np.linspace(0.0, 1.0, num + 1)

    # Ferguson Spline Matrix
    trans_matrix = np.matrix([[2, 1, -2, 1],
                              [-3, -2, 3, -1],
                              [0, 1, 0, 0],
                              [1, 0, 0, 0]])

    # velocity on each point
    velo = path[1] - path[0]
    for pt0, pt2 in zip(path[:-2], path[2:]):
        velo = np.vstack((velo, (pt2 - pt0)/2))
    velo = np.vstack((velo, path[-1] - path[-2]))

    # Calc. spline
    q = np.matrix([[] for i in range(2)]).reshape(0, 2)

    for p0, v0, p1, v1 in zip(path[:-1], velo[:-1], path[1:], velo[1:]):
        s_vec = np.matrix([s**3, s**2, s**1, s**0])
        vec = np.matrix([p0, v0, p1, v1])
        q_add = s_vec.T * trans_matrix * vec
        q = np.vstack((q, q_add[:-1]))
    q = np.vstack((q, path[-1]))
    return np.array(q)
