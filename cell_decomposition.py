# In[]:
import numpy as np
from math import pi
import matplotlib.pyplot as plt
import seaborn as sns
from geometry import World
from geometry import ObjectGenerator
from viewer import GraphDrawer
import post_process
sns.set()


# In[]:
class CellDecomposition():

    def __init__(self, world):
        self.world = world
        self.L_list = []
        self.cells = []
        self.edges = []
        self.path_list = []

    def set_objects(self):
        '''
        Node and Edges of the objects and frame
        '''
        # Create a node list
        V = []

        # --- Make a node list for frame
        head = np.min(self.world.frame[:-1, 0])
        tail = np.max(self.world.frame[:-1, 0])
        name = 'frame'
        for pt in self.world.frame[:-1]:
            if pt[0] == head:
                V.append([pt, name, 'head'])
            elif pt[0] == tail:
                V.append([pt, name, 'tail'])
            else:
                V.append([pt, name, 'mid'])
        #  --- Make a node list for objects
        frame_xmin = head
        frame_xmax = tail
        for i, obj in enumerate(self.world.objects):
            head = np.min(obj[:-1, 0])
            tail = np.max(obj[:-1, 0])
            name = 'obj' + str(i)
            for pt in obj[:-1]:
                if pt[0] == head:
                    type = 'head'
                elif pt[0] == tail:
                    type = 'tail'
                else:
                    type = 'mid'

                if pt[0] < frame_xmin:
                    pt[0] = frame_xmin
                elif pt[0] > frame_xmax:
                    pt[0] = frame_xmax
                V.append([pt, name, type])

        V.sort(key=lambda x: x[0][0])

        # --- Create Edge list
        E = []
        for v1, v2 in zip(V[::2], V[1::2]):
            E.append([v1[0], v2[0], v1[1], v1[2]])

        return E

    def decomposition(self, E):
        '''
        Cell decomposition
        [In]
        E
        [Out]
        L_list: a list of the lines
        Llist_prop: 'head' or 'tail' or 'mid'
        '''
        #
        frame_u = np.max(self.world.frame[:-1, 1])
        frame_d = np.min(self.world.frame[:-1, 1])
        obj0_u = np.max(self.world.objects[0][:-1, 1])
        obj0_d = np.min(self.world.objects[0][:-1, 1])
        obj1_u = np.max(self.world.objects[1][:-1, 1])
        obj1_d = np.min(self.world.objects[1][:-1, 1])

        in_the_frame = False
        obj0 = False
        obj1 = False
        L_list = []
        Lprop_list = []
        xlist = []

        for e in E:
            x = e[0][0]
            current_obj = [e[2], e[3]]
            L = [e[0], e[1]]
            if current_obj[0] == 'frame':
                if obj0:
                    L = [e[0]] + [np.array([x, obj0_u])] + \
                        [np.array([x, obj0_d])] + [e[1]]
                if obj1:
                    L = [e[0]] + [np.array([x, obj1_u])] + \
                        [np.array([x, obj1_d])] + [e[1]]
            else:
                if in_the_frame:
                    if obj0 and current_obj[0] != 'obj0':
                        if obj0_u < e[0][1]:
                            upper = np.array([x, frame_u])
                            lower = np.array([x, obj0_u])
                        if obj0_d > e[0][1]:
                            upper = np.array([x, obj0_d])
                            lower = np.array([x, frame_d])
                    elif obj1 and current_obj[0] != 'obj1':
                        if obj1_u < e[0][1]:
                            upper = np.array([x, frame_u])
                            lower = np.array([x, obj1_u])
                        if obj1_d > e[0][1]:
                            upper = np.array([x, obj1_d])
                            lower = np.array([x, frame_d])
                    else:
                        upper = np.array([x, frame_u])
                        lower = np.array([x, frame_d])
                    L = [upper] + L + [lower]

            if current_obj[0] == 'frame' and current_obj[1] == 'head':
                in_the_frame = True
            elif current_obj[0] == 'frame' and current_obj[1] == 'tail':
                in_the_frame = False
            elif current_obj[0] == 'obj0' and current_obj[1] == 'head':
                obj0 = True
            elif current_obj[0] == 'obj0' and current_obj[1] == 'tail':
                obj0 = False
            elif current_obj[0] == 'obj1' and current_obj[1] == 'head':
                obj1 = True
            elif current_obj[0] == 'obj1' and current_obj[1] == 'tail':
                obj1 = False

            if L[0][0] >= -pi and L[0][0] <= pi:
                if len(xlist) > 0 and x == xlist[-1] and (x == -pi or x == pi):
                    for pt in L:
                        if pt[1] not in L_list[-1][:, 1]:
                            L_new = np.vstack((L_list[-1], pt))
                            L_list[-1] = np.sort(L_new, axis=0)
                else:
                    L = np.sort(L, axis=0)
                    L_list.append(L)
                    Lprop_list.append(current_obj)
                xlist.append(x)

        L_list = np.array(L_list)
        return L_list, Lprop_list

    def create_cells(self, L_list, Lprop_list):
        '''
        Create Cells
        [In]
        L_list
        Lprop_list
        [Out]
        cells:
        cell_props:
        '''
        cells = []
        cell_props = []
        for k in range(len(L_list) - 1):
            start_index = k
            if Lprop_list[k][0] != 'frame' and Lprop_list[k][1] == 'head':
                ls = [0, 2]
                dl = 1
            elif Lprop_list[k][0] != 'frame' and Lprop_list[k][1] == 'tail':
                ls = [0]
                dl = 3
            elif Lprop_list[k][0] == 'frame' and Lprop_list[k][1] == 'head':
                num = len(L_list[k])
                ls = range(0, num, 2)
                dl = 1
            elif Lprop_list[k][0] == 'frame' and Lprop_list[k][1] == 'tail':
                ls = []
                dl = 0

            for l in ls:
                x0 = L_list[k][l, 0]
                y0 = L_list[k][l, 1]
                y1 = L_list[k][l + dl, 1]

                A = np.array([x0, y1])
                B = np.array([x0, y0])
                is_found0 = False
                is_found1 = False
                for i in range(k + 1, len(L_list)):
                    for j in range(0, 4, 2):
                        if y0 == L_list[i][j, 1]:
                            C = L_list[i][j]
                            is_found0 = True
                            end_index = i
                            break
                    for j in range(1, 4, 2):
                        if y1 == L_list[i][j, 1]:
                            D = L_list[i][j]
                            is_found1 = True
                            end_index = i
                            break
                    if is_found0 and is_found1:
                        break

                cells.append(np.vstack((A, B, C, D)))
                cell_props.append((start_index, end_index))

        cells = np.array(cells)
        return cells, cell_props

    def create_edges(self, cells, cell_props):
        '''
        Create Edges
        [In]
        cells
        cell_props
        [Out]
        edges:
        edge_positions
        '''
        num = len(cell_props)
        edges = [[] for i in range(num)]
        edge_positions = []
        for i in range(num):
            tail = cell_props[i][1]
            for j in range(i + 1, num):
                head = cell_props[j][0]
                if tail == head:
                    line0 = np.array([cells[i][2], cells[i][3]])
                    line1 = np.array([cells[j][0], cells[j][1]])
                    if line0[1][1] - line0[0][1] <= line1[0][1] - line1[1][1]:
                        line = line0
                        type = -1
                    else:
                        line = line1
                        type = 1
                    edge_positions.append([(i, j), line, type])
                    edges[i].append(j)
                    edges[j].append(i)
        return edges, edge_positions

    def create_path(self, cells, edges, edge_positions):
        '''
        Create paths
        '''

        def find_cell(pt, cells):
            '''
            Find a cell that contains a specific point.
            '''
            x = pt[0]
            y = pt[1]
            index = None
            for i, cell in enumerate(cells):
                x0 = cell[0][0]
                x1 = cell[2][0]
                y0 = cell[1][1]
                y1 = cell[0][1]
                if x >= x0 and x <= x1 and y >= y0 and y <= y1:
                    index = i
                    break
            if index is None:
                print('The cell cannot be found.')
            return index

        def random_sample(pts):
            xmax = np.max(pts[:, 0])
            ymax = np.max(pts[:, 1])
            xmin = np.min(pts[:, 0])
            ymin = np.min(pts[:, 1])
            x = np.random.rand() * (xmax - xmin) + xmin
            y = np.random.rand() * (ymax - ymin) + ymin
            return np.array([x, y])

        start_index = find_cell(self.world.start, cells)
        goal_index = find_cell(self.world.goal, cells)

        prev_index_list = []
        pts = []
        index = start_index

        while True:
            # Criate the next index candidate list.
            next_index_list = edges[index]
            next_index_list = [i for i in next_index_list
                               if i not in prev_index_list]

            # If next_index_list is empty, valid path cannot be found.
            if next_index_list == []:
                path_is_valid = False
                # print('fail')
                break

            # If next_index_list is not empty,
            # randomly choose the next index from the list.
            next_index = np.random.choice(next_index_list)

            # Find a boundary line between the current cell and the next cell.
            if index > next_index:
                invert = -1
                pair = (next_index, index)
            if index < next_index:
                invert = 1
                pair = (index, next_index)

            for ep in edge_positions:
                if ep[0] == pair:
                    line = ep[1]
                    type = ep[2] * invert

            # If the current cell is larger than the next cell,
            # add the point on the current cell and the next cell to the path.
            if type == 1:
                new_cell = cells[index].copy()
                for vertex in new_cell:
                    if vertex[1] > line[0][1]:
                        vertex[1] = line[0][1]
                    elif vertex[1] < line[1][1]:
                        vertex[1] = line[1][1]
                pts.append(random_sample(new_cell))
                pts.append(random_sample(cells[next_index]))

            # If the current cell is smaller than the next cell,
            # add the point on the next cell to the path.
            elif type == -1:
                new_cell = cells[next_index].copy()
                for vertex in new_cell:
                    if vertex[1] > line[0][1]:
                        vertex[1] = line[0][1]
                    elif vertex[1] < line[1][1]:
                        vertex[1] = line[1][1]
                pts.append(random_sample(new_cell))

            # If next index is goal index, loop ends.
            if next_index == goal_index:
                path_is_valid = True
                break

            # Update the value.
            prev_index_list.append(index)
            index = next_index
            # --- loop end

        pts = np.array(pts)
        pts = np.vstack((self.world.start, pts, self.world.goal))
        return pts, path_is_valid

    def main(self, num, shortcut):
        E = self.set_objects()
        self.L_list, Lprop_list = self.decomposition(E)
        self.cells, cell_props = self.create_cells(self.L_list, Lprop_list)
        self.edges, edge_positions = self.create_edges(self.cells, cell_props)

        count_failure = 0
        while len(self.path_list) < num:
            pts, path_is_valid = self.create_path(self.cells, self.edges,
                                                  edge_positions)

            if path_is_valid:
                if shortcut:
                    pts = post_process.shortcut(pts, self.world)
                self.path_list.append(pts)

            if not path_is_valid:
                count_failure += 1

                if count_failure > 1000:
                    print('fail')
                    break

        return self.path_list


# In[]:
if __name__ == '__main__':

    # --- world class
    world = World()

    # --- Set frame and objects
    og = ObjectGenerator(world)
    og.generate_frame([-pi, pi], [-pi, pi])
    og.generate_object_sample1()
    og.set_object_type()

    # world_margin = world.mcopy(rate=1.05)
    world.update_objects([[0.5, 0.0], [-0.5, 0.0]])

    # --- Set start/goal point
    world.start = np.array([-3.0, -3.0])
    world.goal = np.array([3.0, 3.0])

    cd = CellDecomposition(world)
    path_list = cd.main(10, shortcut=True)
    gd = GraphDrawer(world)
    # gd.draw_path(cd.L_list)
    gd.draw_path(path_list)
    plt.show()
