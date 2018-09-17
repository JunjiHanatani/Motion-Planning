#!/usr/bin/env python
import numpy as np
from math import pi
from collision_checker import CollisionChecker


class Node():
    def __init__(self, index, position, parent):
        self.index = index
        self.position = position
        self.parent = parent


class World():
    def __init__(self, frame=None, objects=None, object_type=None, robot=None,
                 start=None, goal=None, type=None):
        print('\n[ World instance setup ]')
        self.frame = frame
        self.objects = objects
        self.start = start
        self.goal = goal
        self.object_type = object_type
        self.robot = robot
        self.type = type

    def generate_frame(self, xrange, yrange):
        self.frame = np.array([[xrange[0], yrange[0]],
                               [xrange[0], yrange[1]],
                               [xrange[1], yrange[1]],
                               [xrange[1], yrange[0]],
                               [xrange[0], yrange[0]]])
        print('- Frame has been set. \n xrange:{0}, yrange:{1}'
              .format(xrange, yrange))
        return self.frame

    def mcopy(self, rate=1.1):
        if self.object_type == 'grid':
            print('Error: Margin object cannot be created with grid type')
        world_copy = World()
        world_copy.frame = self.frame.copy()
        world_copy.objects = self.objects.copy()
        if self.start is not None:
            world_copy.start = self.start.copy()
        if self.start is not None:
            world_copy.goal = self.goal.copy()
        world_copy.object_type = self.object_type
        world_copy.robot = self.robot
        world_copy.type = self.type

        for i, obj in enumerate(self.objects):
            center = np.mean(obj[0:-1], axis=0)
            vec = obj - center
            world_copy.objects[i] = vec * rate + center

        print('- Copy has been completed with rate={}.'.format(rate))
        return world_copy

    def update_objects(self, delta_list):
        for obj, delta in zip(self.objects, delta_list):
            obj += np.array(delta)

    def update_start(self, path, dl):
        vec = (path[1] - path[0])
        vec_abs = np.linalg.norm(vec)
        if vec_abs >= dl:
            delta = vec / vec_abs * dl
        else:
            delta = vec
        self.start += delta

    def shift(self, new_center):
        d = len(self.start)
        self.length = [np.max(self.frame[:, i]) - np.min(self.frame[:, i])
                       for i in range(d)]
        self.center = [
            (np.max(self.frame[:, i]) + np.min(self.frame[:, i])) / 2.0
            for i in range(d)]

        shift = new_center - self.center

        def loop_shift(a, b, x):
            return (x - a) % (b - a) + a

        self.frame = self.frame + shift

        min = np.array([np.min(self.frame[:, i]) for i in range(d)])
        max = np.array([np.max(self.frame[:, i]) for i in range(d)])
        self.start = loop_shift(min, max, self.start)
        self.goal = loop_shift(min, max, self.goal)
        sign = np.sign(new_center)
        xcopy = [obj + sign *
                 np.array([self.length[0], 0]) for obj in self.objects]
        ycopy = [obj + sign *
                 np.array([0, self.length[1]]) for obj in self.objects]
        xycopy = [
            obj + sign * np.array([self.length[0], self.length[1]])
            for obj in self.objects]

        objects_shift_all = self.objects + xcopy + ycopy + xycopy

        # Narrow down valid objects
        valid_objects = []
        for obj in objects_shift_all:
            a = [max[i] > np.min(obj[:, i]) and min[i] <
                 np.max(obj[:, i]) for i in range(d)]
            if all(a):
                valid_objects.append(obj)
        self.objects = valid_objects

    def generate_cspace_objects(self, num, cartesian_objects):

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

        m = num[0]
        n = num[1]
        min_pt = np.min(self.frame, axis=0)
        max_pt = np.max(self.frame, axis=0)
        q0_range = np.linspace(min_pt[0], max_pt[0], m + 1)
        q1_range = np.linspace(min_pt[1], max_pt[1], n + 1)
        gridcenter0 = (q0_range[:-1] + q0_range[1:]) / 2.0
        gridcenter1 = (q1_range[:-1] + q1_range[1:]) / 2.0

        if self.objects is None:
            self.objects = np.ones((m, n))
        self.objects = np.array(self.objects)
        for i, q0 in enumerate(gridcenter0):
            for j, q1 in enumerate(gridcenter1):
                p = self.robot.forward_kinematics([q0, q1])
                for obj in cartesian_objects:
                    self.objects[i, j] = self.objects[i, j] * collision(p, obj)
        self.objects = list(self.objects)
        print('- Cspace objects have been created \
        from the objects in the cartesian space.')
        self.object_type = 'grid'
        print('- Object_type has been set to "grid"')

        return self.objects


class ObjectGenerator():
    def __init__(self, world):
        self.world = world

    def generate(self, num, max):
        self.world.objects = []
        for i in range(num):
            n = np.random.randint(2, max - 1)
            while True:
                angles = np.random.rand(n) * pi / 2.0
                angles.sort()
                angles0 = np.insert(angles[:-1], 0, 0.0)
                angles1 = angles.copy()
                is_valid = [a0 - a1 < 0.5 for a0, a1 in zip(angles0, angles1)]
                if(all(is_valid)):
                    break

            r = np.random.rand(n) * 0.5 + 0.5
            pts = np.array([r * np.cos(angles), r * np.sin(angles)]).T
            self.world.objects.append(
                np.vstack((np.zeros(2), pts, np.zeros(2))))

    def locate(self, scale_minmax):
        frame = np.array(self.world.frame)
        xmax = np.max(frame[:, 0])
        xmin = np.min(frame[:, 0])
        ymax = np.max(frame[:, 1])
        ymin = np.min(frame[:, 1])

        def collision_check(target):
            res = [cc.path_validation_new(target[i], target[i + 1])
                   for i in range(len(target) - 1)]
            res_start = [cc.crossing_number_algorithm
                         (self.world.start, target)]
            res_goal = [cc.crossing_number_algorithm
                        (self.world.goal, target)]
            return all(res + res_start + res_goal)

        new_objects = []
        for obj in self.world.objects:
            self.world.objects = new_objects
            cc = CollisionChecker(self.world, 0.05)
            for iter in range(10):
                # set random value
                # x, y: translate distance, theta: rotate angle, scale: scale
                x = np.random.rand() * (xmax - xmin) + xmin
                y = np.random.rand() * (ymax - ymin) + ymin
                theta = np.random.rand() * 2.0 * pi
                scale = np.random.rand() * \
                    (scale_minmax[1] - scale_minmax[0]) + scale_minmax[0]

                # Rotate, scale, translate
                rot_mat = np.array([[np.cos(theta), -np.sin(theta)],
                                    [np.sin(theta),  np.cos(theta)]])
                rslt = np.dot(rot_mat, obj.T) * scale + \
                    np.array([x, y]).reshape(-1, 1)

                # Check collisions with the other objects.
                if collision_check(rslt.T):
                    new_objects.append(rslt.T)
                    break

        self.world.objects = new_objects.copy()

    def generate_object_sample1(self):
        object1 = [[-pi, -2.0],
                   [-pi, -1.0],
                   [1.0, -1.0],
                   [1.0, -2.0],
                   [-pi, -2.0]]
        object2 = [[-1.0, 1.0],
                   [-1.0, 2.0],
                   [pi, 2.0],
                   [pi, 1.0],
                   [-1.0, 1.0]]
        self.world.objects = np.array([object1, object2])
        print('- Objects have been created. (Sample1)')

        self.world.object_type = 'poly'
        print('- Object_type has been set to "poly"')

        return self.world.objects

    def set_object_type(self):
        if self.world.objects is None:
            print('** Warning: Object does not exist')
        elif np.unique(self.world.objects)[0] == 0.0 and \
                np.unique(self.world.objects)[1] == 1.0:
            self.world.object_type = 'grid'
            print('- Object_type has been set to "grid"')
        else:
            self.world.object_type = 'poly'
            print('- Object_type has been set to "poly"')

    def generate_object(self, pts):
        object = np.array(pts)
        object = np.vstack((object, pts[0]))
        if self.world.objects is None:
            self.world.objects = np.array([object])
        else:
            tmp = list(self.world.objects)
            tmp.append(object)
            self.world.objects = np.array(tmp)
        print('- The object has been created.')

        self.world.object_type = 'poly'
        print('- Object_type has been set to "poly"')

        return self.world.objects


class RobotArm():
    def __init__(self, param, dynamics_func=None, path_func=None,
                 start=None, goal=None):
        self.param = param
        self.start = start
        self.goal = goal
        self.start = start
        self.goal = goal
        self.dim = len(param['base'])
        self.dynamics_func = dynamics_func
        self.path_func = path_func

    def forward_kinematics(self, qlist):
        rotmat_list = []
        for q in qlist:
            rotmat = np.matrix([[np.cos(q), -np.sin(q), 0.0],
                                [np.sin(q), np.cos(q), 0.0],
                                [0.0, 0.0, 1.0]])
            rotmat_list.append(rotmat)

        transmat_list = []
        for l in self.param['length']:
            transmat = np.matrix([[1.0, 0.0, l],
                                  [0.0, 1.0, 0.0],
                                  [0.0, 0.0, 1.0]])
            transmat_list.append(transmat)

        globalTbase = np.matrix([[1.0, 0.0, self.param['base'][0]],
                                 [0.0, 1.0, self.param['base'][1]],
                                 [0.0, 0.0, 1.0]])

        tmp = globalTbase
        T = []
        pts = [np.array(self.param['base'])]
        for i in range(self.dim):
            tmp = tmp * rotmat_list[i] * transmat_list[i]
            T.append(tmp)
            pt = tmp * np.matrix([0, 0, 1]).T
            pts.append(np.array(pt.T).flatten()[:2])
        return np.array(pts)

    def R2_dynamics(self, q):

        m1 = self.param['mass'][0]
        m2 = self.param['mass'][1]
        I1 = self.param['I'][0]
        I2 = self.param['I'][1]
        r1 = self.param['r'][0]
        r2 = self.param['r'][1]
        L1 = self.param['length'][0]
        ag = 9.8

        M1 = I1 + I2 + m1 * r1 * r1 + m2 * (L1 * L1 + r2 * r2)
        M2 = m2 * r2 * r2 + I2
        R = m2 * L1 * r2

        C1 = np.cos(q[0])
        C2 = np.cos(q[1])
        C12 = np.cos(q[0] + q[1])
        S2 = np.sin(q[1])

        M = np.matrix([[M1 + 2.0 * R * C2, M2 + R * C2],
                       [M2 + R * C2, M2]])
        G1 = np.matrix([[0.0, -R * S2],
                        [-R * S2, -R * S2]])
        G2 = np.matrix([[R * S2, 0.0],
                        [0.0, 0.0]])
        g = np.matrix([[r1 * C1, L1 * C1 + r2 * C12],
                       [0.0,     r2 * C12]]) * \
            np.matrix([[m1 * ag],
                       [m2 * ag]])
        G = [G1, G2]

        return M, G, g

    def RP_dynamics(self, q):

        m1 = self.param['mass'][0]
        m2 = self.param['mass'][1]
        I1 = self.param['I'][0]
        I2 = self.param['I'][1]
        r1 = self.param['r'][0]
        ag = 9.8

        q1 = q[0]
        q2 = q[1]

        M1 = I1 + I2 + m1 * r1 * r1
        M = np.matrix([[M1 + m2 * q2 * q2, 0.0],
                       [0.0, m2]])
        G1 = np.matrix([[0.0, m2 * q2],
                        [m2 * q2, 0.0]])
        G2 = np.matrix([[-m2 * q2, 0.0],
                        [0.0, 0.0]])
        g = np.matrix([[ag * (m1 * r1 + m2 * q2) * np.cos(q1)],
                       [ag * m2 * np.sin(q1)]])
        G = [G1, G2]

        return M, G, g

    def spline_path(self, s, path):
        '''
        Ferguson_spline
        http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.453.5654&rep=rep1&type=pdf
        http://markun.cs.shinshu-u.ac.jp/learn/cg/cg3/index4.html
        '''
        s = np.array(s).reshape(-1)
        n = len(path) - 1
        tlist = s * n % 1

        indices = np.rint(s * n - tlist)
        indices = np.array(indices, dtype='int8')

        # Ferguson Spline Matrix
        trans_matrix = np.matrix([[2, 1, -2, 1],
                                  [-3, -2, 3, -1],
                                  [0, 1, 0, 0],
                                  [1, 0, 0, 0]])

        # velocity on each point
        v = path[1] - path[0]
        for pt0, pt2 in zip(path[:-2], path[2:]):
            v = np.vstack((v, (pt2 - pt0) / 2))
        v = np.vstack((v, path[-1] - path[-2]))

        # Calc. spline
        q = np.matrix([[] for i in range(2)]).reshape(0, 2)
        dqds = np.matrix([[] for i in range(2)]).reshape(0, 2)
        dqdds = np.matrix([[] for i in range(2)]).reshape(0, 2)

        for i, t in zip(indices, tlist):

            if n == i:
                i = i - 1
                t = 1.0

            t_vec = np.matrix([t**3, t**2, t**1, t**0])
            ds_vec = np.matrix([3.0 * t**2, 2.0 * t**1, t**0, t * 0.0])
            dds_vec = np.matrix([6.0 * t**1, 2.0 * t**1, t * 0.0, t * 0.0])
            vec = np.matrix([path[i], v[i], path[i + 1], v[i + 1]])

            q_add = t_vec * trans_matrix * vec
            dqds_add = ds_vec * trans_matrix * vec * n
            dqdds_add = dds_vec * trans_matrix * vec * n**2

            q = np.vstack((q, q_add))
            dqds = np.vstack((dqds, dqds_add))
            dqdds = np.vstack((dqdds, dqdds_add))

        if np.size(s) > 1:
            q = np.array(q)
            dqds = np.array(dqds)
            dqdds = np.array(dqdds)

        elif np.size(s) == 1:
            q = np.array(q)[0]
            dqds = np.array(dqds)[0]
            dqdds = np.array(dqdds)[0]

        return q, dqds, dqdds
