import random
import numpy as np
from math import pi
from operator import attrgetter
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing as mp

from collision_checker import CollisionChecker
from geometry import World
from geometry import ObjectGenerator
from viewer import RealtimePlot
from prm import PRM
import time

sns.set()

'''
Parameter setting
'''
n_ind = 100   # The number of individuals in a population.
n_elite = 10
NGEN = 1000   # The number of generation loop.
fitness_thresh = 0.1
verbose = True
# random.seed(64)

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
# og.generate(10, 5)
# og.locate([2.0, 3.0])
# world.objects = og.world.objects

# --- Generte Collision Checker
cc = CollisionChecker(world, 0.1)

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


def set_fitness(eval_func, pop):
    """Set fitnesses of each individual in a population."""
    for i, fit in enumerate(map(eval_func, pop)):
        pop[i].fitness = fit


def evalOneMax(gene):
    """Objective function."""
    pts = gene.copy()
    score = 0.0

    for i in range(len(pts) - 1):
        dist = np.linalg.norm(pts[i + 1] - pts[i])
        score = score + dist
        # if not cc.path_validation(pts[i], pts[i+1]):
        #    score = score + 10
    if not cc.collision_check:
        score = score + 10
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
    if len(ind1) == len(ind2) and all((ind1 == ind2).flatten()):
        return ind1
    best = []
    id1 = []
    id2 = []
    tmp1 = ind1.copy()
    tmp2 = ind2.copy()
    j = 0
    for i1 in range(len(ind1) - 1):
        for i2 in range(len(ind2) - 1):
            # Calculate an intersection between line AB and line CD.
            pt, r, s = calc_intersection_point(
                ind1[i1], ind1[i1 + 1], ind2[i2], ind2[i2 + 1])
            # If intersection is found,
            if pt is not None:
                if np.linalg.norm(r - 1.0) > 1e-6 and np.linalg.norm(r) > 1e-6:
                    tmp1 = np.insert(tmp1, i1 + j + 1, pt, axis=0)
                    tmp2 = np.insert(tmp2, i2 + j + 1, pt, axis=0)
                    id1.append(i1 + j + 1)
                    id2.append(i2 + j + 1)
                    j = j + 1
    id1 = [0] + id1 + [len(ind1) + j + 1]
    id2 = [0] + id2 + [len(ind2) + j + 1]
    for i in range(len(id1) - 1):
        if (evalOneMax(tmp1[id1[i]:id1[i + 1] + 1])
                < evalOneMax(tmp2[id2[i]: id2[i + 1] + 1])):
            best = best + list(tmp1[id1[i]: id1[i + 1]])
        else:
            best = best + list(tmp2[id2[i]: id2[i + 1]])
    new_path = node_reduction(best)
    return Individual(new_path)


def node_reduction(path):

    new_path = [path[0]]
    for i in range(1, len(path) - 1):
        u = path[i + 1] - path[i]
        umag = np.linalg.norm(u)
        if umag > 0.2:
            new_path.append(path[i])
        elif not cc.collision_check([new_path[-1], path[i + 1]]):
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
            elif not cc.collision_check([new_path[-1], path[i + 1]]):
                new_path.append(path[i])
    new_path.append(path[-1])
    return new_path


def mut_normal(ind, indpb):
    """Mutation function."""
    mut = ind.copy()
    for i in range(1, len(ind) - 1):
        if random.random() < indpb:
            var = 0.5
            while True:
                mut[i] = ind[i] + np.random.normal(0.0, var, 2)
                var = var * 0.5
                if cc.collision_check([mut[i - 1], mut[i], mut[i + 1]]):
                    break
    return Individual(mut)


'''
Main Routine for Genetic Algorithm
'''

# --- Initial Population
initial_pop = create_pop_prm(10, 10)
pop = initial_pop.copy()

# --- Evaluate the initial population
set_fitness(evalOneMax, pop)
best_ind = min(pop, key=attrgetter("fitness"))

# Initialize output list
trajectory = np.array([world.start])
history = [[0, best_ind.fitness]]

if verbose:
    rp = RealtimePlot(world, pop, best_ind, trajectory, dt=0.01)
    hist_dataset = [[ind.fitness for ind in pop]]

# --- Generation loop starts.
print("Generation loop start.")
print("Generation: 0. Best fitness: " + str(best_ind.fitness))
clk0 = 0.0
g = 0
for total_g in range(NGEN):

    '''
    STEP1 : Selection.
    '''

    t0 = time.time()
    # Remove collision path
    if g == 0:
        new_pop = []
        for ind in pop:
            new_path = cc.collision_avoidance(ind)
            if new_path is not None:
                new_pop.append(Individual(new_path))
        pop = new_pop.copy()
        set_fitness(evalOneMax, pop)

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
        if np.random.rand() < 0.5:
            tmp = mut_normal(ind, indpb=0.3)
            mutant.append(tmp)
        else:
            mutant.append(ind)
    offspring = mutant.copy()

    if g == 0:
        mutant = create_pop_prm(2, 5)
    else:
        mutant = []
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
    n_offspring = n_ind - len(elites) - len(crossover) - len(mutant)
    pop = (list(elites) + list(crossover)
           + list(mutant) + list(offspring[0:n_offspring]))
    set_fitness(evalOneMax, pop)
    g = g + 1

    # Update World
    clk1 = time.time()
    if clk1 - clk0 > 10000:
        # Update objects.
        world.objects[0] = world.objects[0] + np.array([0.01, 0.0])
        world.objects[1] = world.objects[1] - np.array([0.01, 0.0])

        # Update current robot position.
        vec = best_ind[1] - best_ind[0]
        world.start = vec / np.linalg.norm(vec) * 0.1 + world.start
        for i in range(len(pop)):
            pop[i][0] = world.start

        trajectory = np.vstack((trajectory, world.start))

        # Initialize
        clk0 = time.time()
        g = 0

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
        rp.plot(pop, best_ind, trajectory, np.array(history))

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

# rp.fig.clf() if verbose else None

for i in range(5):
    plt.hist(hist_dataset[i], bins=np.linspace(13, 15, 31), alpha=0.7)
plt.show()
