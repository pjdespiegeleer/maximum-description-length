from typing import List

import numpy as np
from cpmpy import *
import pickle


def knapsack(n: int = 20, r: int = 500, per: int = 90, save: bool = True, seed: int = 1, cap=None):
    np.random.seed(seed)
    values = np.random.randint(1, r, n)
    weights = np.random.randint(1, r, n)
    capacity = int(max([0.2 * sum(weights), r]))

    # Construct the model.
    x = boolvar(shape=n, name="x")
    model = Model(
        sum(x*weights) <= capacity,
        maximize=
        sum(x*values)
    )
    model.solve()
    items = np.where(x.value())[0]
    max_value = sum(values[items])

    model = Model()
    model += (sum(x*weights) <= capacity)
    model += (sum(x*values) >= int(max_value*per/100))

    item_list = []
    bool_list = []
    if cap is None:
        num = model.solveAll(display=lambda: bool_list.append(x.value()))
    else:
        num = model.solveAll(display=lambda: bool_list.append(x.value()), solution_limit=cap)

    print("Number of solutions:", num)

    fitness_list = []
    for b in bool_list:
        item_list.append(np.where(b)[0])
        fitness_list.append(sum(b * values))

    set_list = []
    for l in item_list:
        set_list += [frozenset(l)]

    if save:
        if cap is None:
            base_string = "knapsack_solutions/knapsack_randomseed_"+str(seed)+"_n"+str(n)+"_R"+str(r)+"_per"+str(per)
        else:
            base_string = "knapsack_solutions/knapsack_randomseed_"+str(seed)+"_n"+str(n)+"_R"+str(r)+"_per"+str(per)\
                          +"_cap"+str(cap)
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


def integer_knapsack(n: int = 20, r: int = 100, per: int = 90, max_c: int = 1, save: bool = True, seed: int = 1, cap=None):
    np.random.seed(seed)
    values = np.random.randint(1, r, n)
    weights = np.random.randint(1, r, n)
    capacity = int(max([0.2 * sum(weights), r]))

    # Construct the model.
    x = intvar(lb=0, ub=max_c, shape=n, name="x")
    model = Model(
        sum(x*weights) <= capacity,
        maximize=
        sum(x*values)
    )
    model.solve()
    items = np.where(x.value())[0]
    max_value = sum(x.value()*values)

    model = Model()
    model += (sum(x*weights) <= capacity)
    model += (sum(x*values) >= int(max_value*per/100))

    integer_list = []
    if cap is None:
        num = model.solveAll(display=lambda: integer_list.append(x.value()))
    else:
        num = model.solveAll(display=lambda: integer_list.append(x.value()), solution_limit=cap)

    print("Number of solutions:", num)

    fitness_list = []
    bool_list = []
    for x in integer_list:
        bool_list.append(integer_list_to_bool_list(lst=x, max_c=max_c))
        fitness_list.append(sum(x * values))

    set_list = []
    for l in item_list:
        set_list += [frozenset(l)]

    if save:
        if cap is None:
            base_string = "knapsack_solutions/knapsack_randomseed_"+str(seed)+"_n"+str(n)+"_R"+str(r)+"_per"+str(per)
        else:
            base_string = "knapsack_solutions/knapsack_randomseed_"+str(seed)+"_n"+str(n)+"_R"+str(r)+"_per"+str(per) \
                          +"_cap"+str(cap)
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


def integer_list_to_bool_list(lst: List[int], max_c=None):
    if max_c is None:
        max_c = np.max(lst)
    bool_list = []
    for i, x in enumerate(lst):
        bools = [int(x <= j) for j in range(max_c+1)]
        bool_list += bools
    return bool_list
