# In[]:
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
from cell_decomposition import CellDecomposition
import time

sns.set()
# np.random.seed(1)


class Individual(np.ndarray):
    """Container of a individual."""
    fitness = None

    def __new__(cls, a):
        return np.asarray(a).view(cls)


class GeneticAlgorithm():

    def __init__(self, world=None, NGEN=1000, n_ind=100, n_elite=10,
                 fitness_thresh=0.1, margin_on=False, verbose=True):
        self.trajectory = [world.start]
        self.history = []
        self.pop = []
        self.best_ind = []
        self.gtot = 0

        self.n_ind = n_ind   # The number of individuals in a population.
        self.n_elite = n_elite
        self.NGEN = NGEN   # The number of generation loop.
        self.fitness_thresh = fitness_thresh
        self.verbose = verbose

        # --- Generate Collision Checker
        self.world = world
        if margin_on:
            self.world_margin = world.mcopy(rate=1.05)
        else:
            self.world_margin = world
        self.cc = CollisionChecker(world)
        self.ccm = CollisionChecker(self.world_margin)

    def create_pop_prm(self, m, n):
        prm_list = [PRM(self.world_margin, 30, 3) for i in range(m)]
        path_list = []
        for prm in prm_list:
            prm.single_query()
            if prm.path_list != []:
                prm.multi_query(n)
            path_list += prm.path_list
        return [Individual(path) for path in path_list]

    def create_pop_cd(self):
        cd = CellDecomposition(self.world_margin)
        path_list = cd.main(self.n_ind, shortcut=True)
        self.pop = [Individual(path) for path in path_list]
        return self.pop

    def set_fitness(self, eval_func):
        """Set fitnesses of each individual in a population."""
        for i, fit in enumerate(map(eval_func, self.pop)):
            self.pop[i].fitness = fit

    def evalOneMax(self, gene):
        """Objective function."""
        delta = [0.1]
        score_list = []
        for d in delta:
            pts = self.smoothing(gene, d)
            score = 0.0
            for i in range(len(pts) - 1):
                dist = np.linalg.norm(pts[i + 1] - pts[i])
                score = score + dist
                # if not cc.path_validation(pts[i], pts[i+1]):
                #    score = score + 10
            if not self.ccm.path_validation(pts):
                score = score + 10
            score_list.append(score)
        return min(score_list)

    def eval_for_cx(self, gene):
        """Objective function."""
        pts = gene.copy()
        score = 0.0
        for i in range(len(pts) - 1):
            dist = np.linalg.norm(pts[i + 1] - pts[i])
            score = score + dist
        return score

    def selTournament(self, tournsize):
        """Selection function."""
        chosen = []
        for i in range(self.n_ind - self.n_elite):
            aspirants = [random.choice(self.pop) for j in range(tournsize)]
            chosen.append(min(aspirants, key=attrgetter("fitness")))
        return chosen

    def selElite(self):
        pop_sort = sorted(self.pop, key=attrgetter("fitness"))
        elites = pop_sort[:self.n_elite]
        return elites

    def calc_intersection_point(self, A, B, C, D):
        denominator = (B[0] - A[0]) * (C[1] - D[1]) - \
            (B[1] - A[1]) * (C[0] - D[0])
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

    def subcalc(self, queue, seed, offspring, elites, n):
        np.random.seed(seed)
        crossover = []
        for i in range(10):
            randint1 = np.random.randint(len(offspring))
            randint2 = np.random.randint(len(elites))
            child = self.cx(offspring[randint1], elites[randint2])
            crossover.append(child)
        queue.put(crossover)

    def cx(self, ind1, ind2):
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
                pt, r, s = self.calc_intersection_point(
                    ind1[i1], ind1[i1 + 1], ind2[i2], ind2[i2 + 1])
                # If intersection is found,
                if pt is not None:
                    if np.linalg.norm(r - 1.0) > 1e-6 and \
                            np.linalg.norm(r) > 1e-6:
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
            if (self.eval_for_cx(tmp1[id1[i]:id1[i + 1] + 1])
                    < self.eval_for_cx(tmp2[id2[i]: id2[i + 1] + 1])):
                best = best + list(tmp1[id1[i]: id1[i + 1]])
            else:
                best = best + list(tmp2[id2[i]: id2[i + 1]])

        return Individual(best)

    def node_reduction(self, path):
        path = np.array(path)
        new_path = [path[0]]
        for i in range(1, len(path) - 1):
            u = path[i + 1] - path[i]
            umag = np.linalg.norm(u)
            if umag > 0.1:
                new_path.append(path[i])
            elif not self.ccm.line_validation(new_path[-1], path[i + 1]):
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
                elif not self.ccm.line_validation(new_path[-1], path[i + 1]):
                    new_path.append(path[i])
        new_path.append(path[-1])
        new_path = np.array(new_path)
        return new_path

    def mut_normal(self, ind, indpb, maxiter):
        """Mutation function."""
        mut = ind.copy()
        for i in range(1, len(ind) - 1):
            if random.random() < indpb:
                var = 0.5
                for j in range(maxiter):
                    mut[i] = ind[i] + np.random.normal(0.0, var, 2)
                    var = var * 0.5
                    if self.ccm.path_validation(
                            [mut[i - 1], mut[i], mut[i + 1]]):
                        break
                else:
                    mut[i] = ind[i]
        return Individual(mut)

    def smoothing(self, pts, delta=0.1):
        # resampled_path = post_process.resampling(pts, delta)
        # bezier_path = post_process.bezier(resampled_path, 50)
        # return bezier_path
        return pts

    def main(self, duration):
        '''
        Main Routine for Genetic Algorithm
        '''
        ini_time = time.time()

        # --- Evaluate the initial population
        self.set_fitness(self.evalOneMax)
        self.best_ind = min(self.pop, key=attrgetter("fitness"))

        self.history.append([self.gtot, self.best_ind.fitness])
        # self.trajectory.append([world.start[0], world.start[1]])

        np.vstack((self.trajectory, self.world.start))

        # if verbose:
        #    rp.plot(self.pop, self.best_ind, self.trajectory, self.history)

        # --- Generation loop starts.
        print('\n[Genetic Algorithm]')
        print("Generation loop start.")
        print("Generation: 0. Best fitness: {}\n"
              .format(str(self.best_ind.fitness)))
        for g in range(self.NGEN):

            '''
            STEP1 : Selection.
            '''
            t0 = time.time()
            # Elite selection
            elites = self.selElite()
            # Tournament Selection
            offspring = self.selTournament(tournsize=3)

            '''
            STEP2 : Mutation.
            '''
            t1 = time.time()
            mutant = []
            for ind in offspring:
                if np.random.rand() < 0.5:
                    tmp = self.mut_normal(ind, indpb=0.3, maxiter=3)
                    mutant.append(tmp)
                else:
                    mutant.append(ind)
            offspring = mutant.copy()
            # mutant = self.create_pop_cd(10)
            mutant = self.create_pop_prm(2, 5)

            '''
            Step3 : Crossover.
            '''
            t2 = time.time()
            proc = 8
            n_cross = 10
            queue = mp.Queue()
            ps = [
                mp.Process(target=self.subcalc, args=(
                    queue, i, offspring, elites, n_cross))
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
            n_offspring = self.n_ind - \
                (len(elites) + len(crossover) + len(mutant))
            self.pop = (list(elites) + list(crossover) +
                        list(offspring[0:n_offspring])) + list(mutant)

            # --- Delete the redundant points on the path.
            self.pop = [Individual(self.node_reduction(path))
                        for path in self.pop]

            self.set_fitness(self.evalOneMax)

            '''
            Output
            '''
            t4 = time.time()

            # --- Print best fitness in the population.
            self.best_ind = min(self.pop, key=attrgetter("fitness"))
            print("Generation: {: > 2}".format(g))
            print("Best fitness: {: .3f}".format(self.best_ind.fitness))
            print("Time: {0:.3f}, {1:.3f}, {2:.3f}, {3:.3f}, Total: {4:.3f} \n"
                  .format(t1 - t0, t2 - t1, t3 - t2, t4 - t3, t4 - t0))

            # Fitness transition
            self.history.append([self.gtot, self.best_ind.fitness])

            '''
            # STEP5: Termination
            '''
            if time.time() - ini_time > duration:
                # --- Visualization
                if self.verbose:
                    rp.plot(self.pop, self.best_ind,
                            self.trajectory, self.history)

                break

            if self.best_ind.fitness <= self.fitness_thresh:
                print('The best fitness reaches the threshold value.')
                break

            if g >= 5:
                diff = np.abs(self.history[-5][1] - self.history[-1][1])
                if diff < 1e-2:
                    print('The best fitness does not change for 10 steps.')
                    break

            self.gtot += 1


if __name__ == '__main__':

    '''
    World setting
    '''

    # World
    world = World()
    world.generate_frame([-pi, pi], [-pi, pi])
    world.type = 'cartesian'

    # Objects in the cartesian space
    og = ObjectGenerator(world)
    og.generate_object_sample1()

    # --- Set start/goal point
    world.start = np.array([-pi / 2, -pi / 2]) * 1.9
    world.goal = np.array([pi / 2, pi / 2]) * 1.9

    '''
    Functions
    '''
    NGEN = 1000
    n_ind = 100
    n_elite = 10
    fitness_thresh = 0.1
    verbose = True
    ga = GeneticAlgorithm(world, NGEN, n_ind, n_elite, fitness_thresh, verbose)
    rp = RealtimePlot(world, 100, dt=0.01)

    # --- Initial Population
    # initial_pop = create_pop_cd(100)
    initial_pop = ga.create_pop_prm(10, 10)
    ga.pop = initial_pop.copy()

    for i in range(100):

        ga.main(1.0)

        # Update objects.
        if i % 10 < 5:
            vel = 0.05
        else:
            vel = -0.05
        dl = np.array([[vel, 0.0], [-vel, 0.0]])

        world.update_objects(dl)
        ga.world_margin.update_objects(dl)

        # Update current robot position.
        dl = 0.1
        world.update_start(ga.best_ind, dl)
        ga.world_margin.update_start(ga.best_ind, dl)
        for i in range(len(ga.pop)):
            ga.pop[i][0] = world.start

        # Create trajectory
        ga.trajectory = np.vstack((ga.trajectory, world.start))

        if ga.best_ind.fitness <= ga.fitness_thresh:
            print('Goal.')
            break

    if verbose:
        rp.fig.clf()
        plt.show()
