#!/usr/bin/env python

# In[]:
import numpy as np
from math import pi
import seaborn as sns
import matplotlib.pyplot as plt

from geometry import World
from geometry import ObjectGenerator
from viewer import GraphDrawer
from collision_checker import CollisionChecker
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

world_margin = world.mcopy(rate=1.1)
cc = CollisionChecker(world_margin)

# In[]:
# --- Visualize
gd = GraphDrawer(world_margin)
plt.show()
