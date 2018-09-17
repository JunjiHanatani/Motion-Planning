#!/usr/bin/env python

# In[]:
import numpy as np
from math import pi
import seaborn as sns
import matplotlib.pyplot as plt

from geometry import World
from geometry import ObjectGenerator
from viewer import GraphDrawer
from dijkstra import Dijkstra
from collision_checker import CollisionChecker
import post_process

sns.set()

'''
World setting
'''

# --- world class
world = World()

# --- Set frame
world.frame = np.array(
    [[-pi, -pi],
     [-pi, pi],
     [pi, pi],
     [pi, -pi],
     [-pi, -pi]])

# --- Set start/goal point
world.start = np.array([-pi / 2, -pi / 2]) * 1.9
world.goal = np.array([pi / 2, pi / 2]) * 1.9

# --- Generate objects
og = ObjectGenerator(world)
world.objects = og.example()

# --- CollisionChecker
cc = CollisionChecker(world)

# In[]:
m = 6
n = 6
w = 1.0
h = 1.0
x0 = -3.0
y0 = -3.0

V = [[x0 + i * w, y0 + j * h] for j in range(n) for i in range(m)]

E = []
for i in range(m * n - 1):
    if i % m != (m - 1):
        E.append([(i, i + 1), 1.0])
    if int(i / m) != (n - 1):
        E.append([(i, i + n), 1.0])
obj_cel = [6, 7, 8, 9, 26, 27, 28, 29]
E = [e for e in E if (e[0][0] not in obj_cel) and (e[0][1] not in obj_cel)]

# In[]:
djk = Dijkstra(V, E)
djk.build(0)
djk.query(35)

# In[]:
path = []
for cell in djk.path:
    x = np.random.rand() * w + cell[0]
    y = np.random.rand() * h + cell[1]
    path.append([x, y])
pts = np.array(path)
pts = np.vstack((world.start, pts, world.goal))

# In[]:
pts_smooth = post_process.shortcut(pts, world)

# In[]:
bezier_list = []
for i in range(10):
    pts_resampled = post_process.resampling(pts_smooth, 0.1 * (i + 1))
    pts_bezier = post_process.bezier(pts_resampled, 50)
    bezier_list.append(pts_bezier)

# In[]:
# --- Visualize
gd = GraphDrawer(world)
gd.draw_path([pts, pts_smooth] + bezier_list)
plt.show()
