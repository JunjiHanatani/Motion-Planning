import numpy as np
import matplotlib.pyplot as plt


class Dijkstra():

    def __init__(self, V, E):
        self.V = V
        self.E = E
        self.start = 0
        self.goal = 0
        self.d = []
        self.path = []
        self.path_index = []
        self.distance = []
        self.prev = []

    def build(self, start):

        # initialize
        self.start = start
        n = len(self.V)
        self.d = [np.inf] * n
        self.d[start] = 0.0
        self.prev = [None] * n
        searched_list = []

        while len(searched_list) < n:

            #
            masked_d = self.d.copy()
            for i in searched_list:
                masked_d[i] = np.inf

            # target node (minimum d)
            target_index = np.argmin(masked_d)

            # Edges from the target node
            edges = [e for e in self.E if target_index in e[0]]

            for edge in edges:
                if target_index == edge[0][0]:
                    neighbor_index = edge[0][1]
                else:
                    neighbor_index = edge[0][0]
                dist = edge[1]

                if (self.d[neighbor_index] > self.d[target_index] + dist):
                    self.d[neighbor_index] = self.d[target_index] + dist
                    self.prev[neighbor_index] = target_index

            # list of the searched nodes
            searched_list.append(target_index)

    def query(self, goal):
        # Set goal point
        self.goal = goal

        # Calc. path
        self.path_index = [self.goal]
        self.path = [self.V[self.goal]]
        while self.path_index[-1] != self.start:

            ind = self.prev[self.path_index[-1]]

            # If ind is None, previous node can be found,
            # that is, path does not exist.
            if ind is not None:
                self.path_index.append(ind)
                self.path.append(self.V[ind])
            else:
                self.path = None
                self.path_index = None
                break

        if self.path is not None:
            self.path = np.array(self.path[::-1])
            self.path_index = self.path_index[::-1]

        # Calc. distance
        self.distance = self.d[goal]


if __name__ == '__main__':
    # Node list
    V = np.array([[0.0, 0.0],
                  [0.3, -0.1],
                  [0.3, 0.3],
                  [0.6, 0.35],
                  [0.4, 0.5],
                  [0.1, 0.4]])
    # Edge list
    E = [[(0, 1), 7.0],
         [(0, 2), 9.0],
         [(0, 5), 14.0],
         [(1, 2), 10.0],
         [(1, 3), 15.0],
         [(2, 3), 11.0],
         [(2, 5), 2.0],
         [(3, 4), 6.0],
         [(4, 5), 9.0]]

    djk = Dijkstra(V, E)
    start_node = 0
    goal_node = 3
    # Build a distance map from the chosen start point
    djk.build(start_node)
    # Find the shortest path from the start point to the goal point
    djk.query(goal_node)

    # Visualize
    if djk.path is not None:
        print(djk.path_index)
        print(djk.path)
        print(djk.distance)
        plt.scatter(V[:, 0], V[:, 1], marker='o')
        plt.plot(djk.path[:, 0], djk.path[:, 1])
        plt.show()
    else:
        # If the nodes are not connected from the start to the goal,
        # any path cannot be found.
        print('Path cannot be found.')
