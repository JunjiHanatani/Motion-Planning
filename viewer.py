#!/usr/bin/env python
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set()


class GraphDrawer():
    def __init__(self, world):
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
        if world.type is not None:
            ax.set_title(world.type)
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        # --- Draw objects
        if world.object_type == 'poly':
            objects = world.objects
        elif world.object_type == 'grid':
            xmin, ymin = np.min(world.frame, axis=0)
            xmax, ymax = np.max(world.frame, axis=0)
            n, m = np.shape(world.objects)
            xr = np.linspace(xmin, xmax, m + 1)
            yr = np.linspace(ymin, ymax, n + 1)
            objects = []
            for i, (x0, x1) in enumerate(zip(xr[:-1], xr[1:])):
                for j, (y0, y1) in enumerate(zip(yr[:-1], yr[1:])):
                    if not world.objects[i][j]:
                        objects.append([[x0, y1],
                                        [x0, y0],
                                        [x1, y0],
                                        [x1, y1]])
        else:
            print('The value of "object_type":{} is invalid.'
                  .format(world.objects_type))

        # Draw objects
        for obj in objects:
            points = tuple([tuple(pt) for pt in obj])
            poly = plt.Polygon(points, fc="#10101090")
            ax.add_patch(poly)

        # --- Draw a frame
        points = tuple([tuple(pt) for pt in world.frame])
        poly = plt.Polygon(points, ec="#000000", fill=False)
        ax.add_patch(poly)

        # --- Draw a start/goal point
        if world.start is not None:
            ax.scatter(world.start[0], world.start[1],
                       marker='$S$', s=300, c='red')
        if world.goal is not None:
            ax.scatter(world.goal[0], world.goal[1],
                       marker='$G$', s=300, c='red')

        # --- Set axis
        ax.axis('equal')
        self.ax = ax
        self.fig = fig

    def draw_path(self, path_list):
        # If "path_list" contains only one path,
        if len(np.shape(path_list[0])) == 1:
            path = np.array(path_list)
            self.ax.plot(path[:, 0], path[:, 1], marker='o')
        # If "path_list" contains multiple paths,
        if len(np.shape(path_list[0])) == 2:
            for path in path_list:
                path = np.array(path)
                self.ax.plot(path[:, 0], path[:, 1], marker='o')

    def draw_tree(self, path_list):
        for path in path_list:
            self.ax.plot(path[:, 0], path[:, 1], marker='o', color='#BBBBBB')

    def draw_pts(self, pts_list):
        # If "pts_list" contains only one point group,
        if len(np.shape(pts_list[0])) == 1:
            pts = np.array(pts_list)
            self.ax.scatter(pts[:, 0], pts[:, 1], marker='o')
        # If "path_list" contains multiple point groups,
        elif len(np.shape(pts_list[0])) == 2:
            for pts in pts_list:
                pts = np.array(pts)
                self.ax.scatter(pts[:, 0], pts[:, 1], marker='o')


class RealtimePlot():
    def __init__(self, world, num, dt=0.01):
        self.dt = dt
        self.world = world
        matplotlib.rcParams['font.size'] = 14
        matplotlib.rcParams['axes.titlesize'] = 18
        matplotlib.rcParams['axes.labelsize'] = 14
        matplotlib.rcParams['xtick.labelsize'] = 14
        matplotlib.rcParams['ytick.labelsize'] = 14
        matplotlib.rcParams['legend.loc'] = 'best'
        matplotlib.rcParams['legend.frameon'] = True
        matplotlib.rcParams['legend.fontsize'] = 12
        matplotlib.rcParams['legend.edgecolor'] = 'k'

        self.fig = plt.figure(figsize=[12, 6])
        self.ax = self.fig.add_subplot(121)
        self.ax2 = self.fig.add_subplot(122)

        # --- Set title
        self.ax.set_title('2D path planning')
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')

        # --- Set title
        self.ax2.set_title('fitness')
        self.ax2.set_xlabel('genertion')
        self.ax2.set_ylabel('fitness')

        # --- Draw a frame
        points = tuple([tuple(pt) for pt in world.frame])
        poly = plt.Polygon(points, ec="#000000", fill=False)
        self.ax.add_patch(poly)

        # --- Set axis
        self.ax.axis('equal')

        # --- Set lines
        # Population
        self.lines = []
        for i in range(num):
            line, = self.ax.plot(0.0, 0.0, color='#BBBBBB')
            self.lines.append(line)

        # Best individual
        self.bestline, = self.ax.plot(0.0, 0.0, marker='o')

        # Trajectory
        self.traj_line, = self.ax.plot(0.0, 0.0, marker=None)

        # Objects
        objects = self.conversion(world)
        self.object_box = []
        for i, obj in enumerate(objects):
            points = tuple([tuple(pt) for pt in obj])
            poly = plt.Polygon(points, fc="#10101090")
            box = self.ax.add_patch(poly)
            self.object_box.append(box)

        # Start/Goal point
        self.startpoint, = self.ax.plot(world.start[0], world.start[1],
                                        marker='*', c='red')
        self.goalpoint, = self.ax.plot(world.goal[0], world.goal[1],
                                       marker='*', c='red')

        self.history, = self.ax2.plot(0.0, 0.0)

    def plot(self, path_list, best, traj1, hist):
        traj = np.array(traj1)
        hist = np.array(hist)

        # Population
        for i, path in enumerate(path_list):
            self.lines[i].set_data(path[:, 0], path[:, 1])

        # Best individual
        self.bestline.set_data(best[:, 0], best[:, 1])

        # Trajectory
        self.traj_line.set_data(traj[:, 0], traj[:, 1])

        # Start/Goal point
        self.startpoint.set_data(self.world.start[0], self.world.start[1])
        self.goalpoint.set_data(self.world.goal[0], self.world.goal[1])

        # Objects
        objects = self.conversion(self.world)
        for box, obj in zip(self.object_box, objects):
            points = [pt for pt in obj]
            box.set_xy(points)

        self.history.set_data(hist[:, 0], hist[:, 1])
        self.ax2.set_xlim((min(hist[:, 0]), max(hist[:, 0]) + 3))
        self.ax2.set_ylim((min(hist[:, 1]), max(hist[:, 1])))

        # self.ax.set_xlim([0, 3])
        # self.ax.set_ylim([-3, 0])

        if self.dt is None:
            self.fig.canvas.draw()
        else:
            plt.pause(self.dt)

    def conversion(self, world):
        if world.object_type == 'poly':
            objects = world.objects
        elif world.object_type == 'grid':
            xmin, ymin = np.min(world.frame, axis=0)
            xmax, ymax = np.max(world.frame, axis=0)
            n, m = np.shape(world.objects)
            xr = np.linspace(xmin, xmax, m + 1)
            yr = np.linspace(ymin, ymax, n + 1)
            objects = []
            for i, (x0, x1) in enumerate(zip(xr[:-1], xr[1:])):
                for j, (y0, y1) in enumerate(zip(yr[:-1], yr[1:])):
                    if not world.objects[i][j]:
                        objects.append([[x0, y1],
                                        [x0, y0],
                                        [x1, y0],
                                        [x1, y1]])
        else:
            print('The value of "object_type":{} is invalid.'
                  .format(world.objects_type))

        return objects


class LivePlotNotebook(object):
    """
    Live plot using %matplotlib notebook in jupyter notebook
    url: https://gist.github.com/wassname/04e77eb821447705b399e8e7a6d082ce
    """

    def __init__(self, world=None, num=1, jupyter=False, colors=None,
                 title="title", xlabel='x', ylabel='y', figsize=[8, 8]):

        self.jupyter = jupyter
        self.world = world
        matplotlib.rcParams['font.size'] = 14
        matplotlib.rcParams['axes.titlesize'] = 18
        matplotlib.rcParams['axes.labelsize'] = 14
        matplotlib.rcParams['xtick.labelsize'] = 14
        matplotlib.rcParams['ytick.labelsize'] = 14
        matplotlib.rcParams['legend.loc'] = 'best'
        matplotlib.rcParams['legend.frameon'] = True
        matplotlib.rcParams['legend.fontsize'] = 12
        matplotlib.rcParams['legend.edgecolor'] = 'k'

        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_subplot(111)

        # --- Set title
        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)

        if world is not None:
            # --- Draw a frame
            if world.frame is not None:
                points = tuple([tuple(pt) for pt in world.frame])
                poly = plt.Polygon(points, ec="#000000", fill=False)
                self.ax.add_patch(poly)

            # --- Draw Objects
            if world.objects is not None:
                self.object_box = []
                for i, obj in enumerate(self.world.objects):
                    points = tuple([tuple(pt) for pt in obj])
                    poly = plt.Polygon(points, fc="#10101090")
                    box = self.ax.add_patch(poly)
                    self.object_box.append(box)

        # --- Set axis
        # self.ax.axis('equal')

        # --- Set lines
        # Paths
        self.lines = []
        for i in range(num):
            if colors is None:
                line, = self.ax.plot(0.0, 0.0, marker='o')
            else:
                line, = self.ax.plot(0.0, 0.0, marker='o', color=colors[i])
            self.lines.append(line)

        # Trajectory
        self.traj_lines = []
        for i in range(num):
            if colors is None:
                line, = self.ax.plot(0.0, 0.0, marker=None)
            else:
                line, = self.ax.plot(0.0, 0.0, marker=None, color=colors[i])
            self.traj_lines.append(line)

        # Points
        self.points = []
        for i in range(num):
            if colors is None:
                line, = self.ax.plot(0.0, 0.0, marker='o', linestyle='None')
            else:
                line, = self.ax.plot(0.0, 0.0, marker='o', linestyle='None',
                                     color=colors[i])
            self.points.append(line)

    def update(self, path_list=None, traj_list=None, pts_list=None,
               xlim=None, ylim=None, dt=0.1):

        # Population
        if path_list is not None:
            for i, path in enumerate(path_list):
                self.lines[i].set_data(path[:, 0], path[:, 1])

        # Trajectory
        if traj_list is not None:
            for i, traj in enumerate(traj_list):
                self.traj_lines[i].set_data(traj[:, 0], traj[:, 1])

        # Trajectory
        if pts_list is not None:
            for i, pts in enumerate(pts_list):
                self.points[i].set_data(pts[:, 0], pts[:, 1])

        if xlim is not None:
            self.ax.set_xlim(xlim)
        if ylim is not None:
            self.ax.set_ylim(ylim)

        # Objects
        # for box, obj in zip(self.object_box, self.world.objects):
        #    points = [pt for pt in obj]
        #    box.set_xy(points)

        if self.jupyter:
            self.fig.canvas.draw()
        else:
            plt.pause(dt)
