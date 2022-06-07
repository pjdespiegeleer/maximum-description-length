from typing import List

import numpy as np
from cpmpy import *
import pickle
import pandas as pd
from evaluation import generate_entropy_table
from madl import max_description_length
from hamming_diversity import greedy_hamming


def knapsack(n: int = 20, r: int = 500, per: int = 90, save: bool = True, seed: int = 1, cap=None, capacity=None):
    np.random.seed(seed)
    values = np.random.randint(1, r, n)
    weights = np.random.randint(1, r, n)
    if capacity is None:
        capacity = int(max([0.2 * sum(weights), r]))
    print("R = " + str(capacity))
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


    bool_list = []
    num = model.solveAll(display=lambda: bool_list.append(x.value()))
    fitness_list = []
    for b in bool_list:
        fitness_list.append(sum(b * values))

    print("Number of solutions:", num)

    if cap is not None:
        index_list = range(len(bool_list))
        index_list_sorted = sorted(index_list, key=lambda i: fitness_list[i], reverse=True)
        index_list_cap = index_list_sorted[:cap]
        bool_list_np = np.array(bool_list)
        bool_list = bool_list_np[index_list_cap]

    bool_list = list(bool_list)

    fitness_list = []
    item_list = []
    for b in bool_list:
        item_list.append(np.where(b)[0])
        fitness_list.append(sum(b * values))

    set_list = []
    for l in item_list:
        set_list += [frozenset(l)]

    if save:
        if cap is None:
            base_string = "experiments/knapsack/knapsack_solutions/knapsack_randomseed_"+str(seed)+"_n"+str(n)+"_R"+str(r)+"_per"+str(per)+"_capacity"+str(capacity)
        else:
            base_string = "experiments/knapsack/knapsack_solutions/knapsack_randomseed_"+str(seed)+"_n"+str(n)+"_R"+str(r)+"_per"+str(per) \
                          +"_capacity"+str(capacity)+"_cap"+str(cap)
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


def knapsack_with_limit(n: int = 30, r: int = 500, per: int = 90, lim: int = 10, save: bool = True, seed: int = 1, cap=None, capacity=None):
    np.random.seed(seed)
    values = np.random.randint(1, r, n)
    weights = np.random.randint(1, r, n)
    if capacity is None:
        capacity = int(max([0.2 * sum(weights), r]))
    print("R = " + str(capacity))
    # Construct the model.
    x = boolvar(shape=n, name="x")
    model = Model(
        sum(x*weights) <= capacity,
        sum(x) == lim,
        maximize=
        sum(x*values)
    )
    model.solve()
    items = np.where(x.value())[0]
    max_value = sum(values[items])

    model = Model()
    model += (sum(x*weights) <= capacity)
    model += (sum(x) == lim)
    model += (sum(x*values) >= int(max_value*per/100))


    bool_list = []
    num = model.solveAll(display=lambda: bool_list.append(x.value()))
    fitness_list = []
    for b in bool_list:
        fitness_list.append(sum(b * values))

    print("Number of solutions:", num)

    if cap is not None:
        index_list = range(len(bool_list))
        index_list_sorted = sorted(index_list, key=lambda i: fitness_list[i], reverse=True)
        index_list_cap = index_list_sorted[:cap]
        bool_list_np = np.array(bool_list)
        bool_list = bool_list_np[index_list_cap]

    bool_list = list(bool_list)

    fitness_list = []
    item_list = []
    for b in bool_list:
        item_list.append(np.where(b)[0])
        fitness_list.append(sum(b * values))

    set_list = []
    for l in item_list:
        set_list += [frozenset(l)]

    if save:
        if cap is None:
            base_string = "experiments/knapsack/knapsack_solutions/knapsack_randomseed_"+str(seed)+"_n"+str(n)+"_R"+str(r)+"_lim"+str(lim)+"_per"+str(per)+"_capacity"+str(capacity)
        else:
            base_string = "experiments/knapsack/knapsack_solutions/knapsack_randomseed_"+str(seed)+"_n"+str(n)+"_R"+str(r)+"_lim"+str(lim)+"_per"+str(per) \
                          +"_capacity"+str(capacity)+"_cap"+str(cap)
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


def knapsack_between_limits(n: int = 30, r: int = 500, per: int = 90, low_lim: int = 10, high_lim: int = 10, save: bool = True, seed: int = 1, cap=None, capacity=None):
    np.random.seed(seed)
    values = np.random.randint(1, r, n)
    weights = np.random.randint(1, r, n)
    if capacity is None:
        capacity = int(max([0.2 * sum(weights), r]))
    print("R = " + str(capacity))
    # Construct the model.
    x = boolvar(shape=n, name="x")
    model = Model(
        sum(x*weights) <= capacity,
        sum(x) <= high_lim,
        sum(x) >= low_lim,
        maximize=
        sum(x*values)
    )
    model.solve()
    items = np.where(x.value())[0]
    max_value = sum(values[items])

    model = Model()
    model += (sum(x*weights) <= capacity)
    model += (sum(x) >= low_lim)
    model += (sum(x) <= high_lim)
    model += (sum(x*values) >= int(max_value*per/100))


    bool_list = []
    num = model.solveAll(display=lambda: bool_list.append(x.value()))
    fitness_list = []
    for b in bool_list:
        fitness_list.append(sum(b * values))

    print("Number of solutions:", num)

    if cap is not None:
        index_list = range(len(bool_list))
        index_list_sorted = sorted(index_list, key=lambda i: fitness_list[i], reverse=True)
        index_list_cap = index_list_sorted[:cap]
        bool_list_np = np.array(bool_list)
        bool_list = bool_list_np[index_list_cap]

    bool_list = list(bool_list)

    fitness_list = []
    item_list = []
    for b in bool_list:
        item_list.append(np.where(b)[0])
        fitness_list.append(sum(b * values))

    set_list = []
    for l in item_list:
        set_list += [frozenset(l)]

    if save:
        if cap is None:
            base_string = "experiments/knapsack/knapsack_solutions/knapsack_randomseed_"+str(seed)+"_n"+str(n)+"_R"+str(r)+"_lowlim"+str(low_lim)+"_highlim"+str(high_lim)+"_per"+str(per)+"_capacity"+str(capacity)
        else:
            base_string = "experiments/knapsack/knapsack_solutions/knapsack_randomseed_"+str(seed)+"_n"+str(n)+"_R"+str(r)+"_lowlim"+str(low_lim)+"_highlim"+str(high_lim)+"_per"+str(per) \
                          +"_capacity"+str(capacity)+"_cap"+str(cap)
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


def generate_entropy_table(sol_set: List[frozenset], n: int):
    table = []
    for i in range(n):
        count = 0
        for s in sol_set:
            if i in s:
                count += 1
        table += [count / len(sol_set), ]
    return table


if __name__ == "__main__":
    # lim_list = [15, 15, 10]
    # n_list = [25, 25, 25]
    # per_list = [88, 88, 91]
    # r_list = [50, 50, 50]
    # capacity_list = [200, 200, 200]
    lim_list = [10]
    n_list = [25]
    per_list = [91]
    r_list = [50]
    capacity_list = [200]
    for n, per, r, capacity, lim in zip(n_list, per_list, r_list, capacity_list, lim_list):
        # if lim < 12:
        #     continue
        print("Z = "+str(lim))
        b, s, f = knapsack_with_limit(n=n, r=r, per=per, lim=lim, save=False, seed=1, capacity=capacity, cap=1000)
        mdl = max_description_length(db=s, k=10)
        # break
        a = np.array(b)
        c = a.sum(axis=0)/len(a)
        d = [x for x in c if x > 0]
        print(len(d))
        print("Average = " + str(np.average(a.sum(axis=1))))
        # print("STD = " + str(np.std(a.sum(axis=1))))



