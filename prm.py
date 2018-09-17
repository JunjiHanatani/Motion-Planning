import numpy as np
from math import pi
import matplotlib.pyplot as plt
import time

from collision_checker import CollisionChecker
from geometry import World
from geometry import ObjectGenerator
from viewer import GraphDrawer
from dijkstra import Dijkstra


class PRM():
    def __init__(self, world, n, k, delta=0.01):

        # --- Collision Checker
        self.cc = CollisionChecker(world, delta)

        # --- World
        self.world = world

        # --- parameter
        self.n = n
        self.d = 2
        self.k = k
        self.V = np.array([])
        self.E = []
        self.roadmap = []
        self.resampling_length = 0.1
        self.path = None
        self.smoothed_path = None
        self.resampled_path = None
        self.path_list = []

    def create_k_edges(self, nodes):
        self.E = []
        for i, target_node in enumerate(nodes):
            dist = [np.linalg.norm(target_node - node) for node in nodes]
            indices = np.argsort(dist)

            count_neighbors = 0
            for j in indices[1:]:

                # Collision check
                if self.cc.line_validation(nodes[i], nodes[j]):
                    pair = tuple(sorted([i, j]))

                    # Duplication check
                    if not (pair in [e[0] for e in self.E]):
                        self.E.append([pair, dist[j]])
                        # self.E.append([pair, 1.0])

                        # If the number of edges equal to k, then break
                        count_neighbors += 1
                        if count_neighbors == self.k:
                            break
        return self.E

    def find_nearest_node(self, nodes, p):
        mindist = np.inf
        nearest_node_index = None
        for i, target in enumerate(nodes):
            dist = np.linalg.norm(p - target)
            is_valid = self.cc.line_validation(p, target)
            if mindist > dist and is_valid:
                mindist = dist
                nearest_node_index = i
        return nearest_node_index

    def create_node(self):
        min = np.array([np.min(self.world.frame[:, i]) for i in range(self.d)])
        max = np.array([np.max(self.world.frame[:, i]) for i in range(self.d)])
        length = max - min
        self.V = []
        while len(self.V) < self.n:
            q = np.random.rand(self.d) * length + min
            if self.cc.point_validation(q):
                self.V.append(q)
        return np.array(self.V)

    def build_roadmap(self):
        # create nodes
        self.V = self.create_node()
        # Create edges
        self.E = self.create_k_edges(self.V)
        # Roadmap
        self.roadmap = [
            np.vstack((self.V[e[0][0]], self.V[e[0][1]])) for e in self.E]

    def query(self, start, goal):
        # Find nearest node to start/goal
        start_prm = self.find_nearest_node(self.V, start)
        goal_prm = self.find_nearest_node(self.V, goal)

        # If nearest node cannot be found, self.path = None
        if start_prm is None or goal_prm is None:
            self.path = None
        # else, find shortest path (Dijkstra's algorithm)
        else:
            djk = Dijkstra(self.V, self.E)
            djk.build(start_prm)
            djk.query(goal_prm)
            if djk.path is not None:
                self.path = np.vstack(
                    (self.world.start, djk.path, self.world.goal))
            else:
                self.path = None

    def smoothing(self, path_list):
        q0 = np.array(path_list[0])
        res_path_list = [q0]
        i = 1
        while i < len(path_list):
            q1 = np.array(path_list[i])
            if self.cc.line_validation(q0, q1):
                i += 1
            else:
                q0 = np.array(path_list[i - 1])
                res_path_list.append(q0)
                i += 1
        res_path_list.append(path_list[-1])
        return np.array(res_path_list)

    def resampling(self, path_list):
        q0 = np.array(path_list[0])
        res_position_list = [q0]
        for node in path_list[1:]:
            q1 = np.array(node)
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
        return np.array(res_position_list)

    def single_query(self, verbose=False):
        iter = 0
        while self.path is None:
            if verbose:
                t0 = time.time()
                print('Trial{:>2}: Start'.format(iter))

            self.build_roadmap()
            if verbose:
                t1 = time.time()
                print('build_roadmap:{:.3f}'.format(t1-t0))

            self.query(self.world.start, self.world.goal)
            if verbose:
                t2 = time.time()
                print('query:{:.3f}\n'.format(t2-t1))

            if iter == 10:
                break
            iter += 1

        if self.path is not None:
            self.smoothed_path = self.smoothing(self.path)
            # self.resampled_path = self.resampling(self.smoothed_path)
            self.path_list = [self.smoothed_path]
        else:
            print('Path cannot be found.')
            self.path_list = []

    def multi_query(self, n, verbose=False):
        if verbose:
            print('Start multi_query')
            t0 = time.time()
        # Find nearest node to start/goal
        start_prm = self.find_nearest_node(self.V, self.world.start)
        goal_prm = self.find_nearest_node(self.V, self.world.goal)

        # Initialize Dijkstra module
        djk1 = Dijkstra(self.V, self.E)
        djk2 = Dijkstra(self.V, self.E)

        # Build a distance map
        djk1.build(start_prm)
        djk2.build(goal_prm)

        if verbose:
            t1 = time.time()
            print('Build a distance map: {}'.format(t1 - t0))

        # Generate multiple paths
        self.path_list = []
        while len(self.path_list) < n:
            mid_point = np.random.randint(len(self.V))
            djk1.query(mid_point)
            djk2.query(mid_point)

            if djk1.path is not None and djk2.path is not None:
                djk1.path = np.vstack((self.world.start, djk1.path))
                djk2.path = np.vstack((djk2.path[-2::-1], self.world.goal))
                '''
                smoothed_path1 = self.smoothing(djk1.path)
                smoothed_path2 = self.smoothing(djk2.path)
                self.smoothed_path = np.vstack((
                    smoothed_path1,
                    smoothed_path2))
                '''
                self.path = np.vstack((djk1.path, djk2.path))
                self.smoothed_path = self.smoothing(self.path)
                # '''

                self.path_list.append(self.smoothed_path)

        if verbose:
            t2 = time.time()
            print('Generate multiple paths: {}\n'.format(t2 - t1))


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

    # '''
    # --- Generate 10x10 valid roadmaps
    t0 = time.time()
    prm_list = [PRM(world, 100, 5) for i in range(10)]
    path_list = []
    for prm in prm_list:
        prm.single_query(verbose=True)
        if prm.path_list != []:
            prm.multi_query(10, verbose=True)
            path_list += prm.path_list

    print('Total time:{}'.format(time.time()-t0))

    # --- Draw paths on the map
    pd = GraphDrawer(world)
    pd.draw_path(path_list)
    plt.show()

    '''
    # Single query
    t0 = time.time()
    prm = PRM(world, 30, 3)
    prm.single_query(verbose=True)
    print(time.time() - t0)

    pd = GraphDrawer(world)
    pd.draw_tree(prm.roadmap)
    pd.draw_path([prm.path])
    pd.draw_path([prm.smoothed_path])
    plt.show()

    '''
