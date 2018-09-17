import random
import numpy as np
from math import pi
from operator import attrgetter
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing as mp

import post_process
from collision_checker import CollisionChecker
from geometry import World
from geometry import ObjectGenerator
from viewer import RealtimePlot
from prm import PRM
from dijkstra import Dijkstra
import time

sns.set()
# np.random.seed(1)

'''
Parameter setting
'''
n_ind = 100   # The number of individuals in a population.
n_elite = 10
NGEN = 100   # The number of generation loop.
fitness_thresh = 0.1
verbose = True
# random.seed(64)

'''
World setting
'''

# --- world class
world = World()
world.generate_frame([-pi, pi], [-pi, pi])
world.type = 'cartesian'

# Objects in the cartesian space
og = ObjectGenerator(world)
world.objects = og.generate_object_sample1()

# --- Set start/goal point
world.start = np.array([-pi / 2, -pi / 2]) * 1.9
world.goal = np.array([pi / 2, pi / 2]) * 1.9

# --- Generte Collision Checker
cc = CollisionChecker(world)

'''
Functions
'''


class Individual(np.ndarray):
    """Container of a individual."""
    fitness = None

    def __new__(cls, a):
        return np.asarray(a).view(cls)


def create_pop_prm(m, n):
    prm_list = [PRM(world, 30, 3) for i in range(m)]
    path_list = []
    for prm in prm_list:
        prm.single_query()
        if prm.path_list != []:
            prm.multi_query(n)
        path_list += prm.path_list

    return [Individual(path) for path in path_list]


def create_pop_cd(n_ind):
    m = 6
    n = 6
    w = 1.0
    h = 1.0
    x0 = -3.0
    y0 = -3.0

    # Nodes
    V = [[x0 + i * w, y0 + j * h] for j in range(n) for i in range(m)]

    # Edges
    E = []
    for i in range(m * n - 1):
        if i % m != (m - 1):
            E.append([(i, i + 1), 1.0])
        if int(i / m) != (n - 1):
            E.append([(i, i + n), 1.0])
    obj_cel = [6, 7, 8, 9, 26, 27, 28, 29]
    E = [e for e in E if (e[0][0] not in obj_cel) and (e[0][1] not in obj_cel)]

    # Find shortest path (Dijkstra)
    djk = Dijkstra(V, E)
    djk.build(0)
    djk.query(35)

    # Create Population
    pop = []
    for i in range(n_ind):
        path = [world.start]
        for j in range(1, len(djk.path) - 1):
            x = np.random.rand() * w + djk.path[j][0]
            y = np.random.rand() * h + djk.path[j][1]
            path.append([x, y])
        path.append(world.goal)
        shortened_path = post_process.shortcut(path, world)
        pop.append(Individual(shortened_path))

    return pop


def set_fitness(eval_func, pop):
    """Set fitnesses of each individual in a population."""
    for i, fit in enumerate(map(eval_func, pop)):
        pop[i].fitness = fit


def evalOneMax(gene):
    """Objective function."""
    delta = [0.1, 0.3, 0.5, 0.7]
    score_list = []
    for d in delta:
        pts = smoothing(gene, d)
        score = 0.0
        for i in range(len(pts) - 1):
            dist = np.linalg.norm(pts[i + 1] - pts[i])
            score = score + dist
            # if not cc.path_validation(pts[i], pts[i+1]):
            #    score = score + 10
        # if not cc.collision_check(pts):
        #    score = score + 10
        score_list.append(score)
    return min(score_list)


def eval_for_cx(gene):
    """Objective function."""
    pts = gene.copy()
    score = 0.0
    for i in range(len(pts) - 1):
        dist = np.linalg.norm(pts[i + 1] - pts[i])
        score = score + dist
    return score


def selTournament(pop, n_ind, tournsize):
    """Selection function."""
    chosen = []
    for i in range(n_ind):
        aspirants = [random.choice(pop) for j in range(tournsize)]
        chosen.append(min(aspirants, key=attrgetter("fitness")))
    return chosen


def selElite(pop, n_elite):
    pop_sort = sorted(pop, key=attrgetter("fitness"))
    elites = pop_sort[:n_elite]
    return elites


def calc_intersection_point(A, B, C, D):
    denominator = (B[0] - A[0]) * (C[1] - D[1]) - (B[1] - A[1]) * (C[0] - D[0])
    # If two lines are parallel,
    if abs(denominator) < 1e-6:
        return None, None, None
    AC = A - C
    r = ((D[1] - C[1]) * AC[0] - (D[0] - C[0]) * AC[1]) / denominator
    s = ((B[1] - A[1]) * AC[0] - (B[0] - A[0]) * AC[1]) / denominator
    # If the intersection is out of the edges
    if r < -1e-6 or r > 1.00001 or s < -1e-6 or s > 1.00001:
        return None, r, s
    # Endpoint and startpoint make the intersection.
    if ((np.linalg.norm(r - 1.0) < 1e-6 and np.linalg.norm(s) < 1e-6) or
            (np.linalg.norm(s - 1.0) < 1e-6 and np.linalg.norm(r) < 1e-6)):
        return None, r, s
    point_intersection = A + r * (B - A)
    return point_intersection, r, s


def subcalc(queue, seed, offspring, elites, n):
    np.random.seed(seed)
    crossover = []
    for i in range(10):
        randint1 = np.random.randint(len(offspring))
        randint2 = np.random.randint(len(elites))
        child = cx(offspring[randint1], elites[randint2])
        crossover.append(child)
    queue.put(crossover)


def cx(ind1, ind2):
    """Crossover function for path planning."""
    # --- If the ind1 and the ind2 is the same path, return ind1 and exit.
    if len(ind1) == len(ind2) and all((ind1 == ind2).flatten()):
        return ind1

    # --- Initialize
    best = []
    id1 = [0]
    id2 = [0]
    tmp1 = ind1.copy()
    tmp2 = ind2.copy()
    j = 0

    # --- Search for the intersection
    for i1 in range(len(ind1) - 1):
        for i2 in range(len(ind2) - 1):
            # Calculate an intersection between line AB and line CD.
            pt, r, s = calc_intersection_point(
                ind1[i1], ind1[i1 + 1], ind2[i2], ind2[i2 + 1])
            # If intersection is found,
            if pt is not None:
                if np.linalg.norm(r - 1.0) > 1e-6 and np.linalg.norm(r) > 1e-6:
                    # Add the intersection to the point lists.
                    tmp1 = np.insert(tmp1, i1 + j + 1, pt, axis=0)
                    tmp2 = np.insert(tmp2, i2 + j + 1, pt, axis=0)
                    # Revise the intersection lists.
                    id1.append(i1 + j + 1)
                    id2.append(i2 + j + 1)
                    # j: Num. of the intersection points.
                    j = j + 1
    # Add the last point of the path to the intersection lists.
    id1 = id1 + [len(ind1) + j + 1]
    id2 = id2 + [len(ind2) + j + 1]

    # --- Select the best path based on the path length.
    for i in range(len(id1) - 1):
        if (eval_for_cx(tmp1[id1[i]:id1[i + 1] + 1])
                < eval_for_cx(tmp2[id2[i]: id2[i + 1] + 1])):
            best = best + list(tmp1[id1[i]: id1[i + 1]])
        else:
            best = best + list(tmp2[id2[i]: id2[i + 1]])

    # --- Delete the redundant points on the path.
    new_path = node_reduction(best)

    return Individual(new_path)


def node_reduction(path):
    new_path = [path[0]]
    for i in range(1, len(path) - 1):
        u = path[i + 1] - path[i]
        umag = np.linalg.norm(u)
        if umag > 0.1:
            new_path.append(path[i])
        elif not cc.line_validation(new_path[-1], path[i + 1]):
            new_path.append(path[i])
    new_path.append(path[-1])

    path = new_path.copy()
    new_path = [path[0]]
    for i in range(1, len(path) - 1):
        u = path[i + 1] - path[i]
        v = path[i - 1] - path[i]
        umag = np.linalg.norm(u)
        vmag = np.linalg.norm(v)
        if umag != 0 and vmag != 0:
            cos = np.dot(u, v) / (umag * vmag)
            if abs(cos) < 0.99:
                new_path.append(path[i])
            elif not cc.line_validation(new_path[-1], path[i + 1]):
                new_path.append(path[i])
    new_path.append(path[-1])
    return new_path


def mut_normal(ind, indpb, maxiter):
    """Mutation function."""
    mut = ind.copy()
    for i in range(1, len(ind) - 1):
        if random.random() < indpb:
            var = 0.5
            for j in range(maxiter):
                mut[i] = ind[i] + np.random.normal(0.0, var, 2)
                var = var * 0.5
                if cc.path_validation([mut[i - 1], mut[i], mut[i + 1]]):
                    break
            else:
                mut[i] = ind[i]
    return Individual(mut)


def smoothing(pts, delta=0.1):
    # resampled_path = post_process.resampling(pts, delta)
    # bezier_path = post_process.bezier(resampled_path, 50)
    # return bezier_path
    return pts


'''
Main Routine for Genetic Algorithm
'''
ini_time = time.time()
# --- Initial Population
# initial_pop = create_pop_cd(100)
initial_pop = create_pop_prm(10, 10)
pop = initial_pop.copy()

# --- Evaluate the initial population
set_fitness(evalOneMax, pop)
best_ind = min(pop, key=attrgetter("fitness"))

# Initialize output list
trajectory = np.array([world.start])
history = [[0, best_ind.fitness]]

if verbose:
    # Path & Trajectory
    rp = RealtimePlot(world, n_ind, dt=0.01)
    rp.plot(pop, smoothing(best_ind), trajectory, np.array(history))

    hist_dataset = [[ind.fitness for ind in pop]]

# --- Generation loop starts.
print("Generation loop start.")
print("Generation: 0. Best fitness: " + str(best_ind.fitness))
g = 0
for total_g in range(NGEN):

    '''
    STEP1 : Selection.
    '''
    t0 = time.time()
    # Elite selection
    elites = selElite(pop, n_elite)
    # Tournament Selection
    offspring = selTournament(pop, n_ind - n_elite, tournsize=3)

    '''
    STEP2 : Mutation.
    '''
    t1 = time.time()
    mutant = []
    for ind in offspring:
        if np.random.rand() < 0.7:
            tmp = mut_normal(ind, indpb=1.0, maxiter=3)
            mutant.append(tmp)
        else:
            mutant.append(ind)
    offspring = mutant.copy()

    '''
    Step3 : Crossover.
    '''

    t2 = time.time()
    proc = 8
    n_cross = 10
    queue = mp.Queue()
    ps = [
        mp.Process(target=subcalc, args=(queue, i, offspring, elites, n_cross))
        for i in np.random.randint(100, size=proc)
    ]

    for p in ps:
        p.start()

    crossover = []
    for i in range(proc):
        crossover += queue.get()

    '''
    STEP4: Update next generation.
    '''
    t3 = time.time()
    n_offspring = n_ind - len(elites) - len(crossover)
    pop = (list(elites) + list(crossover) + list(offspring[0:n_offspring]))
    set_fitness(evalOneMax, pop)
    g = g + 1

    '''
    Output
    '''
    t4 = time.time()

    # --- Print best fitness in the population.
    best_ind = min(pop, key=attrgetter("fitness"))
    print("Generation: {0: > 2}. Best fitness: {1: .3f}. \
          Time: {2: .3f}, {3: .3f}, {4: .3f}, {5: .3f}, Total: {6: .3f}"
          .format(total_g + 1, best_ind.fitness,
                  t1 - t0, t2 - t1, t3 - t2, t4 - t3, t4 - t0))

    # Fitness transition
    history.append([total_g + 1, best_ind.fitness])

    # --- Visualization
    if verbose:
        # Path & Trajectory
        rp.plot(pop, smoothing(best_ind), trajectory, np.array(history))

        # Fitness distribution
        if (total_g + 1) % 2 == 0:
            hist_dataset.append([ind.fitness for ind in pop])

    '''
    # STEP5: Termination
    '''
    if best_ind.fitness <= fitness_thresh:
        print('The best fitness reaches the threshold value.')
        break

    if total_g >= 10:
        if np.linalg.norm(history[-10][1] - history[-1][1]) < 1e-2:
            print('The best fitness does not change for 10 steps.')
            break


print("Generation loop ended. The best individual: {}"
      .format(best_ind.fitness))
print("Total time: {}".format(time.time() - ini_time))
rp.fig.clf() if verbose else None

if verbose:
    for i in range(5):
        plt.hist(hist_dataset[i], bins=np.linspace(13, 15, 31), alpha=0.7)
    plt.show()
