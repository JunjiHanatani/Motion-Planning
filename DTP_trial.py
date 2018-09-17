#!/usr/bin/env python

# Decoupled_Trajectory_planning Trial
# In[]:

import numpy as np
import matplotlib.pyplot as plt

m1 = 5.0
I1 = 0.1
r1 = 0.2
m2 = 3.0
I2 = 0.05
ag = 9.8
umax = [40.0, 40.0]
umin = [-40.0, -40.0]
dt = 0.005

s_ini = 0.0
s_fin = 1.0
sd_ini = 0.0
sd_fin = 0.0


def calc_abc(s):
    s = np.array(s)
    s2 = s * s

    a0 = ((I1 + I2 + 2.0 * m2 + m1 * r1**2 - 4.0 * m2 * s + 4.0 * m2 * s2) /
          (-2.0 * s2 + 2 * s - 1))
    a1 = (np.sqrt(2.0) * m2 * (2.0 * s - 1.0) /
          np.sqrt(2.0 * s2 - 2.0 * s + 1.0))

    b0 = (2.0 * (I1 + I2 + m1 * r1**2) * (2.0 * s - 1.0) /
          (2.0 * s2 - 2.0 * s + 1)**2)
    b1 = np.zeros(np.size(b0))

    c0 = ((m1 * r1 + m2 * np.sqrt(4.0 * s2 - 4.0 * s + 2.0)) *
          ag * np.cos(np.arctan2(1, 2.0 * s - 1.0)))
    c1 = m2 * ag * np.sin(np.arctan2(1, 2.0 * s - 1.0))

    a = np.array([a0, a1]).T
    b = np.array([b0, b1]).T
    c = np.array([c0, c1]).T

    return a, b, c


def velocity_limit_curve(n=101):
    v = []
    slist = np.linspace(0, 1, n)
    for s in slist:
        a, b, c = calc_abc(s)
        v_sd = np.inf
        v_sdd = None
        cos = 0.0
        for i in range(2):
            for j in range(2):
                if i != j:

                    if a[i] > 0:
                        ui = umax[i]
                    elif a[i] < 0:
                        ui = umin[i]
                    else:
                        break

                    if a[j] > 0:
                        uj = umax[j]
                    elif a[j] < 0:
                        uj = umin[j]
                    else:
                        break

                    numer = (a[i] * (uj - c[j]) - a[j] * (ui - c[i]))
                    denom = a[i] * b[j] - a[j] * b[i]

                    if denom != 0.0:
                        sd2 = numer / denom
                        if sd2 > 0:
                            sd = np.sqrt(sd2)
                            U, L = ul_calc(s, sd)
                            if min(U) == max(L):
                                v_sd = sd
                                v_sdd = min(U)

        v.append([s, v_sd, v_sdd])

    # calc cos
    cos_list = []
    for i in range(len(v) - 1):
        s0, sd0, sdd0 = v[i][0], v[i][1], v[i][2]
        s1, sd1 = v[i + 1][0], v[i + 1][1]
        if not np.isinf(sd0) and not np.isinf(sd1):
            vec1 = np.array([sd0, sdd0])
            vec2 = np.array([s1 - s0, sd1 - sd0])
            cos = (np.dot(vec1, vec2) /
                   (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
        else:
            cos = None
        cos_list.append(cos)

    return np.array(v), np.array(cos_list),


def ul_calc(s, sd, d=2):
    sd2 = sd * sd

    U = np.zeros(d)
    L = np.zeros(d)

    a, b, c = calc_abc(s)

    for i in range(d):
        if a[i] == 0:
            print('zero inertia point')
            U[i] = None
            L[i] = None
        elif a[i] > 0:
            U[i] = (umax[i] - b[i] * sd2 - c[i]) / a[i]
            L[i] = (umin[i] - b[i] * sd2 - c[i]) / a[i]
        elif a[i] < 0:
            U[i] = (umin[i] - b[i] * sd2 - c[i]) / a[i]
            L[i] = (umax[i] - b[i] * sd2 - c[i]) / a[i]
    return U, L


def calc_crosssection(A, B, C, D):
    if np.isinf(A[0] + B[0] + C[0] + D[0] + A[1] + B[1] + C[1] + D[1]):
        return None

    denominator = (B[0] - A[0]) * (C[1] - D[1]) - (B[1] - A[1]) * (C[0] - D[0])
    # If two lines are parallel,
    if abs(denominator) < 1e-8:
        # print('Two lines are parallel')
        return None

    # Calculate r and s.
    AC = A - C
    r = ((D[1] - C[1]) * AC[0] - (D[0] - C[0]) * AC[1]) / denominator
    s = ((B[1] - A[1]) * AC[0] - (B[0] - A[0]) * AC[1]) / denominator
    # If the intersection point is out of the edges,
    if r < -1e-6 or r > 1.00001 or s < -1e-6 or s > 1.00001 or np.isnan(r * s):
        # print('The intersection point is out of the edges.')
        return None

    # If the intersection point exists,
    point_intersection = A + r * (B - A)
    return point_intersection


def calc_q(s, sd, sdd):
    s2 = s * s
    s3 = s2 * s
    sd2 = sd * sd
    q0 = np.arctan2(1.0, 2.0 * s - 1.0)
    q1 = np.sqrt(4.0 * s2 - 4.0 * s + 2.0)
    qd0 = -sd / (2.0 * s2 - 2.0 * s + 1.0)
    qd1 = (4 * s - 2) * sd / np.sqrt(4 * s2 - 4 * s + 2)
    qdd0 = ((4 * s - 2) * sd2 + (-2 * s2 + 2 * s - 1) * sdd /
            (2.0 * s2 - 2.0 * s + 1.0)**2)
    qdd1 = (np.sqrt(2) * (sd2 + (4 * s3 - 6 * s2 + 4 * s - 1) * sdd) /
            (2.0 * s2 - 2.0 * s + 1.0)**1.5)
    q = np.array([q0, q1]).T
    qd = np.array([qd0, qd1]).T
    qdd = np.array([qdd0, qdd1]).T
    return q, qd, qdd


def calc_u(s, sd, sdd):
    a, b, c = calc_abc(s)
    sd = np.array(sd).reshape(-1, 1)
    sdd = np.array(sdd).reshape(-1, 1)
    return a * sdd + b * sd * sd + c


def backward(s_ini, sd_ini):
    s = s_fin
    sd = sd_fin
    F = [[s, sd, 0.0]]
    is_penetrate = None
    while s > 0 and sd >= 0 and is_penetrate is None:
        U, L = ul_calc(s, sd, d=2)
        lower_sdd = np.max(L)
        sd -= lower_sdd * dt
        s -= sd * dt
        for i in range(len(vlc) - 1):
            penetrate_vlc = calc_crosssection(np.array(F)[-1, 0:2],
                                              np.array([s, sd]),
                                              vlc[i, 0:2],
                                              vlc[i + 1, 0:2])
            if penetrate_vlc is not None:
                print('Reach the velocity limit curve', is_penetrate)
                break

        if is_penetrate is None:
            F.append([s, sd, lower_sdd])

    return(np.array(F))


def forward(s_ini, sd_ini):
    s = s_ini
    sd = sd_ini
    A0 = [[s, sd, 0.0]]
    penetrate_Fcurve = None
    penetrate_vlc = None

    while s < 1 and sd >= 0 and \
            penetrate_Fcurve is None and penetrate_vlc is None:

        U, L = ul_calc(s, sd, d=2)
        upper_sdd = np.min(U)
        sd += upper_sdd * dt
        s += sd * dt
        F_end_index = -1
        for i in range(len(vlc) - 1):
            penetrate_vlc = calc_crosssection(
                np.array(A0)[-1, :2], np.array([s, sd]),
                vlc[i, 0:2], vlc[i + 1, 0:2])
            if penetrate_vlc is not None:
                print('Reach the velocity limit curve', penetrate_vlc)
                break

        for i in range(len(F) - 1):
            penetrate_Fcurve = calc_crosssection(
                np.array(A0)[-1, :2], np.array([s, sd]),
                F[i, :2], F[i + 1, :2])
            if penetrate_Fcurve is not None:
                print('Reach the F curve')
                F_end_index = i + 1
                s = penetrate_Fcurve[0]
                sd = penetrate_Fcurve[1]
                A0.append([s, sd, 0.0])
                break

        if penetrate_Fcurve is None and penetrate_vlc is None:
            A0.append([s, sd, upper_sdd])

    return(np.array(A0), F_end_index)


def ul_graph(ax):
    m, n = 11, 11
    xr = [0, 1.0]
    yr = [0, 3.0]
    slist = np.linspace(xr[0], xr[1], m)
    sdlist = np.linspace(yr[0], yr[1], n)
    u0 = np.zeros((m, n))
    v0 = np.zeros((m, n))
    u1 = np.zeros((m, n))
    v1 = np.zeros((m, n))
    for j, s in enumerate(slist):
        for i, sd in enumerate(sdlist):
            U, L = ul_calc(s, sd)
            Umin = min(U)
            Lmax = max(L)
            u0[i, j] = sd / np.sqrt(sd * sd + Umin * Umin)
            v0[i, j] = Umin / np.sqrt(sd * sd + Umin * Umin)
            u1[i, j] = sd / np.sqrt(sd * sd + Lmax * Lmax)
            v1[i, j] = Lmax / np.sqrt(sd * sd + Lmax * Lmax)

    X, Y = np.meshgrid(slist, sdlist)

    ax.quiver(X, Y, u0, v0, color='red', angles='xy', scale=15.0, alpha=0.3)
    ax.quiver(X, Y, u1, v1, color='blue', angles='xy', scale=15.0, alpha=0.3)
    ax.set_xlim([xr[0] - 0.1, xr[1] + 0.1])
    ax.set_ylim([yr[0] - 0.5, yr[1] + 0.5])

    return ax


# In[]:

if __name__ == '__main__':

    # Calcurate a velocity limit curve
    vlc, tan_list = velocity_limit_curve(101)

    # Step1: Backward
    F = backward(s_fin, sd_fin)

    # Step2: Integrate the Umin forward in time from (s_ini, sd_ini)
    A0, F_end_index = forward(s_ini, sd_ini)

    # Combine all path
    F = F[:F_end_index]
    s_vec = np.vstack((A0, F[::-1]))
    s_vec = np.array(s_vec)[1:-1, :]
    q, qd, qdd = calc_q(s_vec[:, 0], s_vec[:, 1], s_vec[:, 2])
    u = calc_u(s_vec[:, 0], s_vec[:, 1], s_vec[:, 2])
    t = np.array([i * dt for i in range(len(s_vec))])

# In[]: Visualize

    fig = plt.figure(figsize=(18, 8))
    ax1 = fig.add_subplot(231)
    ax1.plot(s_vec[:, 0], s_vec[:, 1], marker='None')
    ax1.plot(vlc[:, 0], vlc[:, 1], marker='None')
    ax1 = ul_graph(ax1)
    ax1.set_xlabel('s')
    ax1.set_ylabel('s_dot')

    ax2 = fig.add_subplot(232)
    ax2.plot(t, s_vec[:, 0], label='s')
    ax2.plot(t, s_vec[:, 1], label='s_dot')
    ax2.plot(t, s_vec[:, 2], label='s_dotdot')
    ax2.set_xlabel('time[sec]')
    ax2.set_ylabel('s, s_dot, s_2dot')

    ax3 = fig.add_subplot(233)
    ax3.plot(t, u[:, 0])
    ax3.plot(t, u[:, 1])
    ax3.set_xlabel('time[sec]')
    ax3.set_ylabel('effort[Nm, N]')

    ax4 = fig.add_subplot(234)
    ax4.plot(t, q[:, 0])
    ax4.plot(t, q[:, 1])
    ax5 = fig.add_subplot(235)
    ax5.plot(t, qd[:, 0])
    ax5.plot(t, qd[:, 1])
    ax6 = fig.add_subplot(236)
    ax6.plot(t, qdd[:, 0])
    ax6.plot(t, qdd[:, 1])

    plt.show()

    print('The minimum-time exexution of the path: {}'.format(t[-1]))
