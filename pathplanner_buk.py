#!/usr/bin/env python
import numpy as np
import time
import matplotlib
from math import pi
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
# np.random.seed(7)
'''
collision check: hierarchical method
frame structure: loop frame, configulation space
planner: PRM
'''


class Node():
    def __init__(self, index, position, parent):
        self.index = index
        self.position = position
        self.parent = parent


class CollisionChecker():
    def __init__(self, objects, frame, dl):
        self.path_validate_length = dl
        self.objects = objects
        self.frame = frame

    def crossing_number_algorithm(self, pt, poly):
        '''
        Crossing Number Algorithm
        https://www.nttpc.co.jp/technology/number_algorithm.html
        '''

        cn = 0
        poly = np.array(poly)
        for i in range(len(poly) - 1):
            if ((poly[i, 1] <= pt[1])
                    and (poly[i + 1, 1] > pt[1])) \
                    or ((poly[i, 1] > pt[1])
                        and (poly[i + 1, 1] <= pt[1])):

                xd = poly[i, 0] \
                    + (pt[1] - poly[i, 1]) \
                    / (poly[i + 1, 1] - poly[i, 1]) \
                    * (poly[i + 1, 0] - poly[i, 0])

                if pt[0] < xd:
                    cn = cn + 1

        if cn % 2 == 0:
            is_outer = True
        else:
            is_outer = False
        return is_outer

    def is_state_valid(self, pt):
        # Objects
        list = [self.crossing_number_algorithm(pt, obj)
                for obj in self.objects]

        # Frame
        if self.frame is not None:
            out_of_frame = self.crossing_number_algorithm(pt, self.frame)
        else:
            out_of_frame = False

        return all(list) and not out_of_frame

    def path_validation(self, p0, p1):
        delta = self.path_validate_length
        p_vector = p1 - p0
        dist = np.linalg.norm(p_vector)
        imax = int(dist / delta) + 1
        pts = [p0 + p_vector / dist * delta * i for i in range(imax)]
        pts.append(p1)
        np.random.shuffle(np.array(pts))
        for p in pts:
            res = self.is_state_valid(p)
            if not res:
                break
        return res


class ObjectGenerator():
    def __init__(self, frame, start, goal):
        self.objects = []
        self.start = start
        self.goal = goal
        self.frame = frame

    def generate(self, num, max):

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
            self.objects.append(np.vstack((np.zeros(2), pts, np.zeros(2))))

    def locate(self, scale_minmax):
        frame = np.array(self.frame)
        xmax = np.max(frame[:, 0])
        xmin = np.min(frame[:, 0])
        ymax = np.max(frame[:, 1])
        ymin = np.min(frame[:, 1])

        def collision_check(objects, target):
            res = [cc.path_validation(target[i], target[i + 1])
                   for i in range(len(target) - 1)]
            res_start = [cc.crossing_number_algorithm(self.start, target)]
            res_goal = [cc.crossing_number_algorithm(self.goal, target)]
            return all(res + res_start + res_goal)

        new_objects = []
        for obj in self.objects:
            cc = CollisionChecker(new_objects, frame, 0.05)
            while True:
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

                if collision_check(new_objects, rslt.T):
                    new_objects.append(rslt.T)
                    break

        self.objects = new_objects.copy()


class PathDrawer():
    def __init__(self, objects, frame, start, goal):
        matplotlib.rcParams['font.size'] = 14
        matplotlib.rcParams['axes.titlesize'] = 18
        matplotlib.rcParams['axes.labelsize'] = 14
        matplotlib.rcParams['xtick.labelsize'] = 14
        matplotlib.rcParams['ytick.labelsize'] = 14
        matplotlib.rcParams['legend.loc'] = 'best'
        matplotlib.rcParams['legend.frameon'] = True
        matplotlib.rcParams['legend.fontsize'] = 12
        matplotlib.rcParams['legend.edgecolor'] = 'k'

        fig = plt.figure(figsize=[8, 8])
        ax = fig.add_subplot(111)

        # --- Set title
        ax.set_title('2D path planning')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        # --- Draw objects
        for obj in objects:
            points = tuple([tuple(pt) for pt in obj])
            poly = plt.Polygon(points, fc="#10101090")
            ax.add_patch(poly)

        # --- Draw a frame
        points = tuple([tuple(pt) for pt in frame])
        poly = plt.Polygon(points, ec="#000000", fill=False)
        ax.add_patch(poly)

        # --- Draw a start/goal point
        ax.scatter(start[0], start[1], marker='*', s=300, c='red')
        ax.scatter(goal[0], goal[1], marker='*', s=300, c='red')

        # --- Set axis
        ax.axis('equal')
        self.ax = ax
        self.fig = fig

    def draw_path(self, path_list):
        for path in path_list:
            self.ax.plot(path[:, 0], path[:, 1], marker=None)

    def draw_tree(self, path_list):
        for path in path_list:
            self.ax.plot(path[:, 0], path[:, 1], marker=None, color='#BBBBBB')


class RRT():

    def __init__(self, objects, frame, start, goal,
                 branch_length=0.1, resampling_length=0.05, verbose=False):

        # set RRT parameter
        self.verbose = verbose
        self.rrt_length = branch_length
        self.resampling_length = resampling_length
        self.rrt_size = 1000
        self.start = start
        self.goal = goal
        self.frame = frame
        self.path = []
        self.smoothed_path = []
        self.resampled_path = []
        self.tree = []
        self.cc = CollisionChecker(objects, frame, 0.005)

    def collision_check(self, nodelist):
        i = 0
        flag = True
        q0 = np.array(nodelist[0].position)
        for node in nodelist[1:]:
            q1 = np.array(node.position)
            if not self.cc.path_validation(q0, q1):
                print('error', i)
                print(q0)
                print(q1)
                flag = False
            q0 = q1
            i = i + 1
        if flag:
            print('No Collision')

    def create_rrt_map(self):

        def generate_random_position(size):
            rand_array = []
            for i in range(size):
                min = np.min(self.frame[:, i])
                max = np.max(self.frame[:, i])
                rand_array.append(np.random.rand() * (max - min) + min)
            return np.array(rand_array)

        def find_nearest_node(nodelist, p):
            mindist = 999
            for target in nodelist:
                diff_vect = p - target.position
                dist = np.linalg.norm(diff_vect)
                if mindist > dist:
                    mindist = dist
                    nearest_node = target
            return nearest_node

        def create_new_node(index, q_rand, parent_node):
            q_parent = parent_node.position
            q_vector = q_rand - q_parent
            dist = np.linalg.norm(q_vector)
            norm_vector = q_vector / dist
            q_new = q_parent + norm_vector * self.rrt_length
            new_node = Node(index + 1, q_new, parent_node.index)
            return new_node

        def add_rrt(rrt, i):
            while True:
                q_rand = generate_random_position(len(self.goal))
                nearest_node = find_nearest_node(rrt, q_rand)
                new_node = create_new_node(i, q_rand, nearest_node)
                if self.cc.path_validation(nearest_node.position,
                                           new_node.position):
                    rrt.append(new_node)
                    break
            return rrt

        def connection_check(new_node, rrt):
            connect_node_candidates = []
            for find_connect_node in rrt:
                if self.cc.path_validation(new_node.position,
                                           find_connect_node.position):
                    connect_node_candidates.append(find_connect_node)
            if connect_node_candidates == []:
                return None, None
            else:
                connect_node_1 = new_node
                connect_node_2 = find_nearest_node(connect_node_candidates,
                                                   new_node.position)
                return connect_node_1, connect_node_2

        def connection_check2(new_node, rrt):
            nearest_node = find_nearest_node(rrt, new_node.position)
            if np.linalg.norm(np.array(new_node.position) -
                              np.array(nearest_node.position)) \
                    < self.rrt_length:
                return new_node, nearest_node
            else:
                return None, None

        rrt_start = [Node(0, self.start, 0)]
        rrt_goal = [Node(0, self.goal, 0)]

        for i in range(self.rrt_size):
            # if connect node pair is found, break.
            connect_node_start, connect_node_goal = \
                connection_check2(rrt_start[-1], rrt_goal)
            if connect_node_start is not None:
                break
            connect_node_goal, connect_node_start = \
                connection_check2(rrt_goal[-1], rrt_start)
            if connect_node_start is not None:
                break

            # add a new node to RRT_start
            rrt_start = add_rrt(rrt_start, i)
            rrt_goal = add_rrt(rrt_goal, i)

        return rrt_start, rrt_goal, \
            connect_node_start.index, connect_node_goal.index

    def path_generation(self, rrt_list, connect_index):
        path_list = [rrt_list[connect_index]]
        next_node = rrt_list[connect_index].parent
        for node in reversed(rrt_list):
            if node.index == next_node:
                path_list.append(node)
                next_node = node.parent
        return path_list

    def path_smoothing(self, path_list):
        q0 = np.array(path_list[0].position)
        res_path_list = [path_list[0]]
        i = 1
        while i < len(path_list):
            q1 = np.array(path_list[i].position)
            if self.cc.path_validation(q0, q1):
                i += 1
            else:
                res_path_list.append(path_list[i - 1])
                q0 = np.array(path_list[i - 1].position)
                i += 1
        res_path_list.append(path_list[-1])
        return res_path_list

    def resampling(self, path_list):
        q0 = np.array(path_list[0].position)
        res_position_list = [q0]
        for node in path_list[1:]:
            q1 = np.array(node.position)
            vector = q1 - q0
            dist = np.linalg.norm(vector)
            if dist > self.resampling_length:
                norm_vector = vector / dist
                num_sample = int(dist / self.resampling_length)
                dx = dist / num_sample
            else:
                norm_vector = vector
                num_sample = 1
                dx = 1.0
            for i in range(1, num_sample + 1):
                q_sample = q0 + norm_vector * dx * i
                res_position_list.append(q_sample)
            q0 = q1
        return res_position_list

    def rrt_visualize(self, rrt):
        parent_list = [node.parent for node in rrt]
        edges = [i for i in range(len(rrt)) if i not in parent_list]
        path_list = []
        for edge in edges:
            nodeset = self.path_generation(rrt, edge)
            path = np.array([node.position for node in nodeset])
            path_list.append(path)
        return path_list

    def make_path(self):
        time_0 = time.time()
        print('') if self.verbose else None
        print(' >>> Motion Planning Start') if self.verbose else None

        # --- Get Start and Goal Point
        if not self.cc.is_state_valid(self.start):
            print('     error: current joint state is invalid')
            return
        if not self.cc.is_state_valid(self.goal):
            print('     error: goal position is invalid')
            return

        # Rapidly Random Tree method
        time_1 = time.time()
        print(' >>> RRT start') if self.verbose else None
        rrt_start, rrt_goal, start_connect_index, goal_connect_index = \
            self.create_rrt_map()

        # Path Generation
        time_2 = time.time()
        print(' >>> Path Generation Start ') if self.verbose else None
        #   --- from mid-point to start-point)
        path_node_set_start = self.path_generation(
            rrt_start, start_connect_index)
        #   --- from mid-point to goal-point)
        path_node_set_goal = self.path_generation(
            rrt_goal, goal_connect_index)
        #   --- combined path
        path_node_set_start.reverse()
        path_node_set = path_node_set_start + path_node_set_goal

        # Smoothed Path
        time_3 = time.time()
        print(' >>> Smoothing Start') if self.verbose else None
        smooth_path_node_set = self.path_smoothing(path_node_set)

        # Re-sampling
        time_4 = time.time()
        print(' >>> Re-sampling Start') if self.verbose else None
        position_list = self.resampling(smooth_path_node_set)

        # Publishing
        time_5 = time.time()
        self.path = np.array([node.position for node in path_node_set])
        self.smoothed_path = np.array(
            [node.position for node in smooth_path_node_set])
        self.resampled_path = position_list
        self.tree = self.rrt_visualize(rrt_start) + \
            self.rrt_visualize(rrt_goal)
        print(' >>> Completed') if self.verbose else None

        # Report
        if self.verbose:
            print('')
            print(' ========== RESULTS SUMMARY ==========')
            print('')
            print('[start/goal]')
            print('start', np.round(self.start, 3))
            print('goal', np.round(self.goal, 3))
            print('')
            print('[RRT]')
            print('#RRT node: ', len(rrt_start), '/', len(rrt_goal))
            print('')
            print('[trajectory points]')
            for node in smooth_path_node_set:
                print(np.round(node.position, 3))
            self.collision_check(smooth_path_node_set)
            print('')
            print('[Re-Sampling Segment]')
            count = 1
            seg = 1
            length0 = np.linalg.norm(position_list[1] - position_list[0])
            for i in range(len(position_list) - 1):
                length1 = np.linalg.norm(
                    position_list[i + 1] - position_list[i])
                if np.round(length1, 8) == np.round(length0, 8) \
                        and i != len(position_list) - 2:
                    count += 1
                else:
                    print('seg' + str(seg), length0, count)
                    seg += 1
                    count = 1
                length0 = length1
            print('')
            print('[time]')
            print('Set Goal : ', time_1 - time_0)
            print('RRT      : ', time_2 - time_1)
            print('Path     : ', time_3 - time_2)
            print('Smoothing: ', time_4 - time_3)
            print('Re-sample: ', time_5 - time_4)
            print('---------------------------')
            print('TOTAL    : ', time_5 - time_0)


if __name__ == '__main__':

    # --- Set frame
    frame = [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [1.0, 0.0],
        [0.0, 0.0]]
    frame = np.array(frame) * 2 * pi - pi

    # --- Set start/goal point
    start = np.array([-pi / 2, -pi / 2]) * 1.9
    goal = np.array([pi / 2, pi / 2]) * 1.9

    # --- Generate objects
    og = ObjectGenerator(frame, start, goal)
    og.generate(10, 7)
    og.locate((1.5, 3.0))

    # --- Generate path
    rrt = RRT(og.objects, frame, start, goal, 0.5, 0.05, True)
    rrt.make_path()

    # --- Draw paths on the map
    pd = PathDrawer(og.objects, frame, start, goal)
    pd.draw_tree(rrt.tree)
    pd.draw_path([rrt.path, rrt.smoothed_path])
    plt.show()
