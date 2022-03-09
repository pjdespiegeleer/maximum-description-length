from typing import List

import numpy as np
from cpmpy import *
import pickle


def knapsack(n: int = 20, r: int = 500, per: int = 90, save: bool = True):
    np.random.seed(1)
    values = np.random.randint(1, r, n)
    weights = np.random.randint(1, r, n)
    capacity = int(max([0.5 * sum(weights), r]))

    # Construct the model.
    x = boolvar(shape=n, name="x")
    model = Model(
        sum(x*weights) <= capacity,
        maximize=
        sum(x*values)
    )
    model.solve()
    # print(f"Capacity: {capacity}, used: {sum(x.value()*weights)}")
    items = np.where(x.value())[0]
    # print("In items:", items)
    max_value = sum(values[items])
    # print(max_value)

    model = Model()
    model += (sum(x*weights) <= capacity)
    model += (sum(x*values) >= int(max_value*per/100))

    item_list = []
    bool_list = []
    num = model.solveAll(display=lambda: bool_list.append(x.value()))
    print("Number of solutions:", num)

    fitness_list = []
    for b in bool_list:
        item_list.append(np.where(b)[0])
        fitness_list.append(sum(b * values))

    set_list = []
    for l in item_list:
        set_list += [frozenset(l)]

    if save:
        base_string = "knapsack_solutions/knapsack_n"+str(n)+"_R"+str(r)+"_per"+str(per)
        with open(base_string + '_setlist.pickle', 'wb') as handle:
            pickle.dump(set_list, handle)
            print("Saved set list")
        with open(base_string + '_boollist.pickle', 'wb') as handle:
            pickle.dump(bool_list, handle)
            print("Saved bool list")
        with open(base_string + '_fitnesslist.pickle', 'wb') as handle:
            pickle.dump(fitness_list, handle)
            print("Saved fitness list")

    return bool_list, set_list, fitness_list
