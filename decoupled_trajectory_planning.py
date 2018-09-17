#!/usr/bin/env python

'''
Decoupled_Trajectory_planning
Principles of Robot Motion Chapter 11.2
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


class Decoupled_Trajectory_Planning():

    def __init__(self, robot, num=100, dt=0.01):
        self.d = 2
        self.dt = dt
        self.s_array = np.linspace(0, 1, num + 1)
        self.dynamics_func = robot.dynamics_func
        self.path_func = robot.path_func
        self.a_array, self.b_array, self.c_array = self.calc_abc(self.s_array)
        self.vlc = []
        self.switch_list = []
        self.szero_list = []
        self.sdmax_list = []
        self.stan_list = []
        self.ppc = np.array([[] for i in range(3)]).reshape(-1, 3)
        self.F = np.array([[] for i in range(3)]).reshape(-1, 3)
        self.umax = robot.param['umax']
        self.umin = robot.param['umin']

    def ul_calc(self, s, sd, d=2):
        sd2 = sd * sd

        U = np.zeros(d)
        L = np.zeros(d)

        a, b, c = self.calc_abc(s)

        for i in range(d):
            if a[i] == 0:
                print('zero inertia point: s={}'.format(s))
                U[i] = None
                L[i] = None
            elif a[i] > 0:
                U[i] = (self.umax[i] - b[i] * sd2 - c[i]) / a[i]
                L[i] = (self.umin[i] - b[i] * sd2 - c[i]) / a[i]
            elif a[i] < 0:
                U[i] = (self.umin[i] - b[i] * sd2 - c[i]) / a[i]
                L[i] = (self.umax[i] - b[i] * sd2 - c[i]) / a[i]
        return U, L

    def calc_crosssection(self, A, B, C, D):
        if np.isinf(A[0] + B[0] + C[0] + D[0] + A[1] + B[1] + C[1] + D[1]):
            return None

        denominator = (B[0] - A[0]) * (C[1] - D[1]) - \
            (B[1] - A[1]) * (C[0] - D[0])
        # If two lines are parallel,
        if abs(denominator) < 1e-8:
            # print('Two lines are parallel')
            return None

        # Calculate r and s.
        AC = A - C
        r = ((D[1] - C[1]) * AC[0] - (D[0] - C[0]) * AC[1]) / denominator
        s = ((B[1] - A[1]) * AC[0] - (B[0] - A[0]) * AC[1]) / denominator
        # If the intersection point is out of the edges,
        if r <= 0.0 or r >= 1.0 or s <= 0.0 or s >= 1.0 or np.isnan(r * s):
            # print('The intersection point is out of the edges.')
            return None

        # If the intersection point exists,
        point_intersection = A + r * (B - A)
        return point_intersection

    def calc_u(self, A):
        s = A[:, 0]
        sd = A[:, 1]
        sdd = A[:, 2]
        a, b, c = self.calc_abc(s)
        sd = np.array(sd).reshape(-1, 1)
        sdd = np.array(sdd).reshape(-1, 1)
        return a * sdd + b * sd * sd + c

    def calc_q(self, A):
        s = A[:, 0]
        sd = A[:, 1]
        sdd = A[:, 2]

        q, dqds, dqdds = self.func_exe(self.path_func, s)
        sd = sd.reshape(-1, 1)
        sdd = sdd.reshape(-1, 1)

        qd = dqds * sd
        qdd = dqdds * sd * sd + dqds * sdd

        return q, qd, qdd

    def ul_graph(self, ax):
        m, n = 31, 31
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
                U, L = self.ul_calc(s, sd)
                Umin = np.amin(U)
                Lmax = np.amax(L)
                if Umin > Lmax:
                    u0[i, j] = sd / np.sqrt(sd * sd + Umin * Umin)
                    v0[i, j] = Umin / np.sqrt(sd * sd + Umin * Umin)
                    u1[i, j] = sd / np.sqrt(sd * sd + Lmax * Lmax)
                    v1[i, j] = Lmax / np.sqrt(sd * sd + Lmax * Lmax)

        X, Y = np.meshgrid(slist, sdlist)

        ax.quiver(X, Y, u0, v0, color='red',
                  angles='xy', scale=30.0, alpha=0.3)
        ax.quiver(X, Y, u1, v1, color='blue',
                  angles='xy', scale=30.0, alpha=0.3)
        ax.set_xlim([xr[0] - 0.1, xr[1] + 0.1])
        ax.set_ylim([yr[0] - 0.5, yr[1] + 0.5])

        return ax

    def velocity_limit_curve(self, slist):

        self.vlc = list(self.vlc)
        for s in slist:
            a, b, c = self.calc_abc(s)
            v_sd = np.inf
            v_sdd = None
            for i in range(2):
                for j in range(2):
                    if i != j:

                        if a[i] > 0:
                            ui = self.umax[i]
                        elif a[i] < 0:
                            ui = self.umin[i]
                        else:
                            break

                        if a[j] > 0:
                            uj = self.umin[j]
                        elif a[j] < 0:
                            uj = self.umax[j]
                        else:
                            break

                        numer = (a[i] * (uj - c[j]) - a[j] * (ui - c[i]))
                        denom = a[i] * b[j] - a[j] * b[i]

                        if denom != 0.0:
                            sd2 = numer / denom
                            if sd2 > 0:
                                sd = np.sqrt(sd2)
                                if sd < v_sd:
                                    U, L = self.ul_calc(s, sd)
                                    v_sd = sd
                                    v_sdd = np.min(U)
            if v_sd is not np.inf:
                self.vlc.append([s, v_sd, v_sdd])

        self.vlc = np.array(self.vlc)
        # self.vlc = self.vlc[np.argsort(self.vlc, axis=0)[:, 0]]

        return self.vlc

    def search_stan(self, verbose=False):

        diff_list = []
        vlc = self.vlc.copy()
        for i in range(len(vlc) - 1):
            s0, sd0, sdd0 = vlc[i][0], vlc[i][1], vlc[i][2]
            s1, sd1 = vlc[i + 1][0], vlc[i + 1][1]
            zero_chk = [1 for szero in self.szero_list
                        if s0 <= szero and szero <= s1]

            # At the zero inertia point, diff can not be zero.
            if (not np.isinf(sd0) and not np.isinf(sd1)) and \
                    (len(zero_chk) == 0):
                tan1 = np.arctan2(sdd0, sd0)
                tan2 = np.arctan2((sd1 - sd0), (s1 - s0))
                diff = tan1 - tan2
            else:
                diff = None
            diff_list.append(diff)

        # Search for tangent point
        for i in range(len(diff_list) - 1):
            diff0 = diff_list[i]
            diff1 = diff_list[i + 1]
            s0, sd0 = self.vlc[i][0], self.vlc[i][1]
            s1, sd1 = self.vlc[i + 1][0], self.vlc[i + 1][1]

            if diff0 is not None and diff1 is not None:
                # Find the point where diff=0.0
                if diff0 > 0 and diff1 < 0:
                    s_tan = interp1d([diff0, diff1], [s0, s1])([0.0])[0]

                    # Calculate sd and sdd
                    sd = interp1d([s0, s1], [sd0, sd1])([s_tan])[0]
                    sdd = sd * (sd1 - sd0) / (s1 - s0)
                    self.stan_list.append([s_tan, sd, sdd, sdd])

        self.stan_list = np.array(self.stan_list).reshape(-1, 4)

        if verbose:
            print(self.stan_list)
            plt.plot(vlc[:-1, 0], diff_list, marker='.')
            plt.scatter(self.stan_list[:, 0],
                        np.zeros(len(self.stan_list[:, 0])), c='red')
            plt.grid()

        return self.stan_list

    def search_szero(self, verbose=False):
        '''
        Search for zero inertia position
        '''

        # Initialize
        slist = self.s_array
        alist = self.a_array
        vlc_new = self.vlc.copy()

        # Search (s:0-1)
        for i in range(len(slist) - 1):
            # Component (2D)
            for j in range(2):

                a0 = alist[i][j]
                a1 = alist[i + 1][j]
                # If the variable 'a' changes minus to plus or plpus to minus,
                if a0 * a1 < 0:
                    # Find s_zero by linear interpolation
                    s_zero = interp1d(
                        [a0, a1], [slist[i], slist[i + 1]])([0.0])[0]

                    # Calc sd_max
                    _, b, c = self.calc_abc(s_zero)
                    sdmax = interp1d(
                        [slist[i], slist[i + 1]],
                        [self.vlc[i, 1], self.vlc[i + 1, 1]]
                    )([s_zero])[0]
                    if b[j] > 0:
                        sd_max_zip = np.sqrt((self.umax[j] - c[j]) / b[j])
                    if b[j] < 0:
                        sd_max_zip = np.sqrt((self.umin[j] - c[j]) / b[j])

                    if b[j] == 0:
                        print('Warning: b=0 at s={}'.format(slist[i]))
                    elif sdmax > sd_max_zip:
                        sdmax = sd_max_zip

                        U, L = self.ul_calc(s_zero, sdmax)
                        Lmax, Umin = L.max(), U.min()
                        sdd_tan_p = sdmax * \
                            (self.vlc[i + 1, 1] - sdmax) / \
                            (slist[i + 1] - s_zero)
                        sdd_tan_m = sdmax * \
                            (sdmax - self.vlc[i, 1]) / (s_zero - slist[i])
                        sdd_max = np.amin([sdd_tan_p, Umin])
                        sdd_min = np.amax([sdd_tan_m, Lmax])
                        self.sdmax_list.append([s_zero, sdmax,
                                                sdd_max, sdd_min])

                        vlc_new = np.insert(
                            vlc_new, i + len(self.sdmax_list),
                            np.array([s_zero, sdmax, None]),
                            axis=0)

                    self.szero_list.append(s_zero)

        if verbose:
            print('szero_list:\n{}\nsdmax_list:\n{}\n'
                  .format(self.szero_list, self.sdmax_list))
            plt.plot(slist, alist)
            plt.scatter(self.szero_list, np.zeros(len(self.szero_list)),
                        c='red')
            plt.grid()

        self.vlc = vlc_new
        self.sdmax_list = np.array(self.sdmax_list).reshape(-1, 4)

    def search_discon(self, n):
        s_discon = [i / (n - 1) for i in range(1, n - 1)]
        vlc = self.vlc.copy()
        sdiscon_list = []
        for i in range(len(vlc) - 1):
            s0, sd0, sdd0 = vlc[i][0], vlc[i][1], vlc[i][2]
            s1, sd1, sdd1 = vlc[i + 1][0], vlc[i + 1][1], vlc[i + 1][2]
            zero_chk = [1 for szero in s_discon
                        if s0 <= szero and szero <= s1]

            # At the zero inertia point, diff can not be zero.
            if len(zero_chk) != 0:
                if sd0 < sd1:
                    sdiscon_list.append([s0, sd0, sdd0, sdd0])
                if sd0 > sd1:
                    sdiscon_list.append([s1, sd1, sdd1, sdd1])
        sdiscon_list = np.array(sdiscon_list)

        return s_discon, sdiscon_list

    def backward(self, ini_pt, verbose=False):
        if verbose:
            print('- Backward search starts. Start:{}'.format(ini_pt[0]))

        # Initialize
        s = ini_pt[0]
        sd = ini_pt[1]

        if ini_pt[2] is None:
            U, L = self.ul_calc(s, sd, d=2)
            lower_sdd = np.amax(L)
        else:
            lower_sdd = ini_pt[3]

        F = [[s, sd, lower_sdd]]
        penetrate_vlc = None
        penetrate_ppc = None

        while True:

            # Update s and sd values
            sd -= lower_sdd * self.dt
            s -= sd * self.dt

            # --- Check the penetration to the velocity limit curve
            for i in range(len(self.vlc) - 1):
                penetrate_vlc = self.calc_crosssection(np.array(F)[-1, 0:2],
                                                       np.array([s, sd]),
                                                       self.vlc[i, 0:2],
                                                       self.vlc[i + 1, 0:2])
                if penetrate_vlc is not None:
                    break

            # --- Check the penetration to the previous phase plane curve
            for i in range(len(self.ppc) - 1):
                penetrate_ppc = self.calc_crosssection(np.array(F)[-1, 0:2],
                                                       np.array([s, sd]),
                                                       self.ppc[i, 0:2],
                                                       self.ppc[i + 1, 0:2])
                if penetrate_ppc is not None:
                    break

            # Termination check
            if s < 0:
                if verbose:
                    print('The backward search has been done. (s < 0)')
                termination_id = 0
                U, L = self.ul_calc(s, sd, d=2)
                lower_sdd = np.amax(L)
                F.append([s, sd, lower_sdd])
                if len(self.ppc) == 0:
                    self.F = np.array(F).copy()
                break

            elif sd < 0:
                if verbose:
                    print('The backward search fails. (sd < 0)')
                termination_id = 1
                U, L = self.ul_calc(s, sd, d=2)
                lower_sdd = np.amax(L)
                F.append([s, sd, lower_sdd])
                break

            elif penetrate_vlc is not None:
                if verbose:
                    print('Reach the velocity limit curve', penetrate_vlc)
                termination_id = 2
                s = penetrate_vlc[0]
                sd = penetrate_vlc[1]
                U, L = self.ul_calc(s, sd, d=2)
                lower_sdd = np.amax(L)
                F.append([s, sd, lower_sdd])

                if len(self.ppc) == 0:
                    self.F = np.array(F).copy()
                break

            elif penetrate_ppc is not None:
                if verbose:
                    print('Reach the previous phase plane curve',
                          penetrate_ppc)
                termination_id = 3
                s = penetrate_ppc[0]
                sd = penetrate_ppc[1]
                U, L = self.ul_calc(s, sd, d=2)
                lower_sdd = np.amax(L)
                F.append([s, sd, lower_sdd])
                F = np.array(F[::-1])
                self.ppc = np.vstack((self.ppc[:i + 1], F))
                self.switch_list.append(s)
                self.switch_list = [switch for switch in self.switch_list
                                    if switch <= s]
                break

            # Update sdd value
            U, L = self.ul_calc(s, sd, d=2)
            lower_sdd = np.amax(L)
            F.append([s, sd, lower_sdd])

        return termination_id

    def forward(self, ini_pt, verbose=False):
        if verbose:
            print('- Forward search starts. Start:{}'.format(ini_pt[0]))

        # Initialize
        s = ini_pt[0]
        sd = ini_pt[1]

        if ini_pt[2] is None:
            U, L = self.ul_calc(s, sd, d=2)
            upper_sdd = np.amin(U)
        else:
            upper_sdd = ini_pt[2]

        A = [[s, sd, upper_sdd]]
        penetrate_ppc = None
        penetrate_vlc = None

        while True:

            # Update s and sd values.
            sd += upper_sdd * self.dt
            s += sd * self.dt

            # --- Check the penetration to the velocity limit curve
            for i in range(len(self.vlc) - 1):
                penetrate_vlc = self.calc_crosssection(np.array(A)[-1, :2],
                                                       np.array([s, sd]),
                                                       self.vlc[i, 0:2],
                                                       self.vlc[i + 1, 0:2])
                if penetrate_vlc is not None:
                    break

            for i in range(len(self.F) - 1):
                penetrate_ppc = self.calc_crosssection(np.array(A)[-1, :2],
                                                       np.array([s, sd]),
                                                       self.F[i, :2],
                                                       self.F[i + 1, :2])
                if penetrate_ppc is not None:
                    break

            # Termination check

            if penetrate_vlc is not None:
                if verbose:
                    print('Reach the velocity limit curve', penetrate_vlc)
                termination_id = 2
                s = penetrate_vlc[0]
                sd = penetrate_vlc[1]

                U, L = self.ul_calc(s, sd, d=2)
                upper_sdd = np.min(U)
                A.append([s, sd, upper_sdd])
                self.ppc = np.vstack((self.ppc, np.array(A)))
                break

            elif penetrate_ppc is not None:
                if verbose:
                    print('Reach the phase plane curve', penetrate_ppc)
                termination_id = 3
                s = penetrate_ppc[0]
                sd = penetrate_ppc[1]

                U, L = self.ul_calc(s, sd, d=2)
                upper_sdd = np.min(U)
                A.append([s, sd, upper_sdd])
                self.ppc = np.vstack((self.ppc, np.array(A), self.F[i::-1]))

                self.switch_list.append(s)
                break

            elif s > 1:
                if verbose:
                    print('The forward search fails. (s > 1)')
                termination_id = 0
                s_inter = 1.0
                sd_inter = interp1d([A[-1][0], s], [A[-1][1], sd])([1.0])[0]

                U, L = self.ul_calc(s_inter, sd_inter, d=2)
                upper_sdd = np.min(U)
                A.append([s_inter, sd_inter, upper_sdd])
                self.ppc = np.vstack((self.ppc, np.array(A), self.F[i::-1]))

                break

            elif sd < 0:
                if verbose:
                    print('The forward search fails. (sd < 0)')
                termination_id = 1

                U, L = self.ul_calc(s, sd, d=2)
                upper_sdd = np.min(U)
                A.append([s, sd, upper_sdd])
                self.ppc = np.vstack((self.ppc, np.array(A), self.F[i::-1]))

                break

            # Update sdd value.
            U, L = self.ul_calc(s, sd, d=2)
            upper_sdd = np.min(U)
            A.append([s, sd, upper_sdd])

        return termination_id

    def calc_abc(self, slist):

        slist = np.array(slist).reshape(-1)

        # Initialize
        alist = np.array([[] for i in range(self.d)]).reshape(0, self.d)
        blist = np.array([[] for i in range(self.d)]).reshape(0, self.d)
        clist = np.array([[] for i in range(self.d)]).reshape(0, self.d)

        for s in slist:

            q, dqds, dqdds = self.func_exe(self.path_func, s)

            # 2R robot
            M, Gamma, g = self.dynamics_func(q)

            # Principles of Robot Motion / Chapter.11 / Eq.11.5
            dqds = np.matrix(dqds)
            dqdds = np.matrix(dqdds)
            a = M * dqds.T
            b = M * dqdds.T + np.vstack([dqds * G * dqds.T for G in Gamma])
            c = g

            alist = np.vstack((alist, np.array(a.T)))
            blist = np.vstack((blist, np.array(b.T)))
            clist = np.vstack((clist, np.array(c.T)))

        if np.size(slist) == 1:
            alist, blist, clist = alist[0], blist[0], clist[0]

        return alist, blist, clist

    def func_exe(self, func, s):
        val = list(map(func, [s]))[0]
        return val[0], val[1], val[2]

    def time_scaling(self):

        s_ini = 0.0
        s_fin = 1.0
        sd_ini = 0.0
        sd_fin = 0.0

        # Step1: Backward
        termination_id = self.backward([s_fin, sd_fin, None], verbose=True)

        # Step2: Integrate the Umin forward in time from (s_ini, sd_ini)
        termination_id = self.forward([s_ini, sd_ini, None], verbose=True)
        s_terminate = self.ppc[-1, 0]

        # Step3: Integrate the Umin forward in time from (s_ini, sd_ini)
        while termination_id == 2:

            sini_list = np.vstack((self.sdmax_list, self.stan_list))
            if len(sini_list) > 1:
                sini_list = sini_list[np.argsort(sini_list[:, 0])]

            for pt in sini_list:
                if pt[0] > s_terminate:
                    ini_pt = pt
                    break
            if ini_pt[0] == s_terminate:
                print('Fail.')
                break

            # backward
            termination_id = self.backward(ini_pt, verbose=True)

            if termination_id == 0:
                print('Error')
                break
            elif termination_id == 1:
                print('Fail.')
                break
            elif termination_id == 2:
                print('Try next.')
                s_terminate = ini_pt[0]
                continue
            elif termination_id == 3:
                print('OK.')
                pass
            self.switch_list.append(ini_pt[0])
            # forward
            termination_id = self.forward(ini_pt, verbose=True)

            if termination_id == 0:
                print('Error')
                break
            elif termination_id == 1:
                print('Fail.')
                break
            elif termination_id == 2:
                s_terminate = self.ppc[-1, 0]
                print('OK')
            elif termination_id == 3:
                print('Phase Plane Search has been successfully done.')
                break

    def visualize(self):
        # Combine all path
        q, qd, qdd = self.calc_q(self.ppc)
        u = self.calc_u(self.ppc)
        t = np.array([i * self.dt for i in range(len(self.ppc))])

        # Visualize
        fig = plt.figure(figsize=(18, 8))
        ax1 = fig.add_subplot(231)
        ax1.plot(self.ppc[:, 0], self.ppc[:, 1], marker='None')
        ax1.plot(self.vlc[:, 0], self.vlc[:, 1], marker='None')
        ax1 = self.ul_graph(ax1)
        ax1.set_xlabel('s')
        ax1.set_ylabel('s_dot')

        ax2 = fig.add_subplot(232)
        ax2.plot(t, self.ppc[:, 0], label='s')
        ax2.plot(t, self.ppc[:, 1], label='s_dot')
        ax2.plot(t, self.ppc[:, 2], label='s_dotdot')
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

        print('Switch points: {}'.format(self.switch_list))
        print('The minimum-time exexution of the path: {}'.format(t[-1]))

    def visualize_phase_plane_curve(self):
        # Visualize Phase Plane Curve
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111)

        # Phase Plane Curve
        ax.plot(self.ppc[:, 0], self.ppc[:, 1], marker='')

        # Velocivty Limit Curve
        ax.plot(self.vlc[:, 0], self.vlc[:, 1], marker='.')

        # Zero Inertia Point
        if len(self.sdmax_list) != 0:
            ax.scatter(self.sdmax_list[:, 0], self.sdmax_list[:, 1],
                       marker='*', c='red', s=100)
        if len(self.stan_list) != 0:
            ax.scatter(self.stan_list[:, 0], self.stan_list[:, 1],
                       marker='^', c='red', s=100)
        ax = self.ul_graph(ax)
        ax.set_xlabel('s')
        ax.set_ylabel('s_dot')
        plt.show()

    def main(self):
        self.velocity_limit_curve(self.s_array)
        self.search_szero()
        self.search_stan()
        self.time_scaling()


if __name__ == '__main__':
    from geometry import RobotArm

    def RProbot_linepath(s):

        s2 = s ** 2
        q0 = np.arctan2(1.0, 2.0 * s - 1.0)
        q1 = np.sqrt(4.0 * s2 - 4.0 * s + 2.0)
        dqds0 = -1 / (2.0 * s2 - 2.0 * s + 1)
        dqds1 = (4.0 * s - 2.0) / np.sqrt(4.0 * s2 - 4.0 * s + 2.0)
        dqdds0 = (4.0 * s - 2.0) / (2.0 * s2 - 2.0 * s + 1)**2
        dqdds1 = np.sqrt(2.0) / (2.0 * s2 - 2.0 * s + 1)**(3 / 2)

        if np.size(s) > 1:
            q = np.array([q0, q1]).T
            dqds = np.array([dqds0, dqds1]).T
            dqdds = np.array([dqdds0, dqdds1]).T

        elif np.size(s) == 1:
            q = np.array([q0, q1])
            dqds = np.array([dqds0, dqds1])
            dqdds = np.array([dqdds0, dqdds1])

        return q, dqds, dqdds

    '''
    Example.11.2 (RProbot)
    '''
    # Robot
    robot_parameter = {'base': [0, 0],
                       'length': [None, None],
                       'mass': [5.0, 3.0],
                       'I': [0.1, 0.05],
                       'r': [0.2, None],
                       'umax': [40.0, 40.0],
                       'umin': [-40.0, -40.0]}

    robotRP = RobotArm(robot_parameter)
    robotRP.dynamics_func = robotRP.RP_dynamics
    robotRP.path_func = RProbot_linepath

    dtp = Decoupled_Trajectory_Planning(robotRP)
    dtp.main()
    dtp.visualize()
    dtp.visualize_phase_plane_curve()
