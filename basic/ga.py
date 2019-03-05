import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import copy
import pandas as pd
import csv

def out(gen, pop):
    path_list = [ind.path for ind in pop]
    columns = ["Pt" + str(i) for i in range(len(pop[0].path))]
    index = ["ID:" + str(ind.id) for ind in pop]
    df_path = pd.DataFrame(path_list, columns=columns, index=index)
    df_path.to_csv("path_" + str(gen) + ".csv")

    property_list = [[ind.id, ind.fitness, ind.parent] for ind in pop]
    columns = ["ID", "Fitness", "Parent"]
    df_property = pd.DataFrame(property_list, columns=columns)
    df_property.to_csv("property_" + str(gen) + ".csv", index=False)

def log(gen, best_ind, filename):

    if gen==0:
        with open(filename, "w") as f:    
            f.write("Generation, Fitness, path\n")
            f.close()

    with open(filename, "a") as f:    
        f.write(str(gen) + ", " + str(best_ind.fitness) + ", " + str(best_ind.path) + "\n")
        f.close()

def mutVisualization(gen, childID):

    df_prop_parent = pd.read_csv("property_0.csv")
    df_prop_child = pd.read_csv("property_1.csv")
    df_path_parent = pd.read_csv("path_0.csv")
    df_path_child = pd.read_csv("path_1.csv")

    parentID = int(df_prop_child[df_prop_child["ID"]==childID]["Parent"])
    parent_path = np.array(df_path_parent.iloc[parentID, 1:7], dtype=np.float64)
    child_path = np.array(df_path_child.iloc[childID, 1:7], dtype=np.float64)

    fig = plt.figure(figsize=[15, 3])
    ax = fig.add_subplot(111)
    data = np.vstack((parent_path.reshape(1, 6), child_path.reshape(1, 6)))
    sns.heatmap(data, annot=True, fmt="1.2f", cmap="RdBu_r", cbar=True, yticklabels=["Parent", "Child"], 
                vmin=0.0, vmax=1.0, linewidths=1.0)
    plt.show()

class Individual():
    def __init__(self):
        self.path = None
        self.id = None
        self.parent = None

pop = []
for i in range(5):
    ind = Individual()
    ind.id = i
    ind.path = np.random.rand(6)
    ind.fitness = np.sum(ind.path)
    pop.append(ind)

for gen in range(2):
    new_pop = []

    for i, ind in enumerate(pop):
        child = Individual()
        child.id = i
        child.path = copy.deepcopy(ind.path)
        child.path[np.random.randint(6)] += np.random.normal(0.0, 0.1)
        child.fitness = np.sum(child.path)
        child.parent = ind.id
        new_pop.append(child)

    out(gen, pop)
    log(gen, ind, "log.log")

    pop = new_pop

mutVisualization(1, 2)
