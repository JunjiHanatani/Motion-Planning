#!/usr/bin/env python
import numpy as np
import time
from math import pi
import matplotlib.pyplot as plt

from collision_checker import CollisionChecker
from geometry import Node
from geometry import World
from geometry import ObjectGenerator
from viewer import GraphDrawer


class RRT():

    def __init__(self, world,
                 branch_length=0.1, resampling_length=0.05, verbose=False):

        # set RRT parameter
        self.verbose = verbose
        self.rrt_length = branch_length
        self.resampling_length = resampling_length
        self.rrt_size = 5000
        self.world = world
        self.path = []
        self.smoothed_path = []
        self.resampled_path = []
        self.tree = []
        self.cc = CollisionChecker(world)

    def collision_check(self, nodelist):
        i = 0
        flag = True
        q0 = np.array(nodelist[0].position)
        for node in nodelist[1:]:
            q1 = np.array(node.position)
            if not self.cc.line_validation(q0, q1):
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
                min = np.min(self.world.frame[:, i])
                max = np.max(self.world.frame[:, i])
                rand_array.append(np.random.rand() * (max - min) + min)
            return np.array(rand_array)

        def find_nearest_node(nodelist, p):
            mindist = float('inf')
            for target in nodelist:
                dist = np.linalg.norm(p - target.position)
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
            # self.world.points = rrt
            while True:
                q_rand = generate_random_position(len(self.world.goal))
                # world.shift(q_rand)
                nearest_node = find_nearest_node(rrt, q_rand)
                new_node = create_new_node(i, q_rand, nearest_node)
                if self.cc.line_validation(nearest_node.position,
                                           new_node.position):
                    rrt.append(new_node)
                    break
            return rrt

        def connection_check(new_node, rrt):
            connect_node_candidates = []
            for find_connect_node in rrt:
                if self.cc.line_validation(new_node.position,
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

        rrt_start = [Node(0, self.world.start, 0)]
        rrt_goal = [Node(0, self.world.goal, 0)]

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
        else:
            print('Path cannot be found.')
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
            if self.cc.line_validation(q0, q1):
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
        if not self.cc.point_validation(self.world.start):
            print('     error: current joint state is invalid')
            return
        if not self.cc.point_validation(self.world.goal):
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
            print('start', np.round(self.world.start, 3))
            print('goal', np.round(self.world.goal, 3))
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

    # --- world class
    world = World()
    world.generate_frame([-pi, pi], [-pi, pi])

    # --- Set start/goal point
    world.start = np.array([-3.0, -3.0])
    world.goal = np.array([3.0, 3.0])

    # --- Set objects
    og = ObjectGenerator(world)
    og.generate_object_sample1()
    og.set_object_type()

    # --- Generate path
    rrt = RRT(world, 0.5, 0.1, True)
    rrt.make_path()

    # --- Draw paths on the map
    pd = GraphDrawer(world)
    pd.draw_tree(rrt.tree)
    pd.draw_path([rrt.path, rrt.smoothed_path])
    plt.show()
