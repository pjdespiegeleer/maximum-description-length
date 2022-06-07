from typing import List

import numpy as np
from cpmpy import *
import pickle
import pandas as pd

from hamming_diversity import greedy_hamming
from madl import max_description_length


def warehouse_location(nw: int = 3, ns: int = 5, r:int = 100, per: int = 110, save: bool = True, seed: int = 0, cap=None):
    np.random.seed(seed)
    warehouse_costs = np.random.randint(size=nw, low=1, high=int(r/4))
    transport_costs = np.random.randint(size=(ns, nw), low=1, high=r)
    warehouse_capacity = np.random.randint(size=nw, low=1, high=ns)

    x = intvar(lb=0, ub=1, shape=(ns, nw), name="x")
    model = Model()
    for i in range(ns):
        model += (sum(x[i, :]) == 1)
    for j in range(nw):
        model += (sum(x[:, j]) <= warehouse_capacity[j], )
    obj = sum(transport_costs * x) + sum(warehouse_costs * [sum(x[:, i]) > 0 for i in range(nw)])
    model.minimize(obj)
    model.solve()
    min_obj = sum(transport_costs * x.value()) + sum(warehouse_costs * [sum(x.value()[:, i]) > 0 for i in range(nw)])

    model = Model()
    for i in range(ns):
        model += (sum(x[i, :]) == 1)
    for j in range(nw):
        model += (sum(x[:, j]) <= warehouse_capacity[j], )
    obj = sum(transport_costs * x) + sum(warehouse_costs * [sum(x[:, i]) > 0 for i in range(nw)])
    model += (obj <= int(min_obj * per/100), )
    bool_list = []
    num = model.solveAll(display=lambda: bool_list.append(x.value()))
    print("Number of solutions:", num)
    fitness_list = []
    for b in bool_list:
        obj = sum(transport_costs * b) + sum(warehouse_costs * [sum(b[:, i]) > 0 for i in range(nw)])
        fitness_list.append(1/obj)

    if cap is not None:
        index_list = range(len(bool_list))
        index_list_sorted = sorted(index_list, key=lambda i: fitness_list[i], reverse=True)
        index_list_cap = index_list_sorted[:cap]
        bool_list_np = np.array(bool_list)
        bool_list = bool_list_np[index_list_cap]

    bool_list = list(bool_list)

    fitness_list = []
    item_list = []
    for i, b in enumerate(bool_list):
        # item_list.append(np.where(np.transpose(b).flatten())[0])
        item_list.append(np.where(b.flatten())[0])
        fitness = sum(transport_costs * b) + sum(warehouse_costs * [sum(b[:, i]) > 0 for i in range(nw)])
        fitness_list.append(1/fitness)
        bool_list[i] = b.flatten()
        # bool_list[i] = np.transpose(b).flatten()

    set_list = []
    for l in item_list:
        set_list += [frozenset(l)]

    if save:
        if cap is None:
            base_string = "experiments/warehouse/warehouse_solutions/warehouse_randomseed_"+str(seed)+"_nw"+str(nw)+"_ns"+str(ns)+"_r"+str(r)+"_per"+str(per)
        else:
            base_string = "experiments/warehouse/warehouse_solutions/warehouse_randomseed_"+str(seed)+"_nw"+str(nw)+"_ns"+str(ns)+"_r"+str(r)+"_per"+str(per) \
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
    nw_list = [10, 10, 10]
    ns_list = [12, 12, 12]
    r_list = [29, 28, 27]
    per_list = [120, 120, 120]
    for nw, ns, per, r in zip(nw_list, ns_list, per_list, r_list):
        print("NS = "+str(ns))
        b, s, o = warehouse_location(save=False, ns=ns, nw=nw, r=r, per=per, seed=0, cap=1000)
        print(len(s))
        d_ori = generate_entropy_table(sol_set=s, n=nw*ns)
        d0 = [x for x in d_ori if x > 0]
        print("N_eff = " + str(len(d0)))
        # madl_set = max_description_length(db=s, k=10)
        # h_set = greedy_hamming(sol=b, k=10)
        # h_list = []
        # for x in h_set:
        #     item = np.where(x)[0]
        #     h_list += [item, ]
        # # d = np.array([x for x in c if x>0])
        # d_madl = generate_entropy_table(sol_set=madl_set, n=80)
        # d_hamming = generate_entropy_table(sol_set=h_list, n=80)
        # df = pd.DataFrame({"Original": d_ori, "MaDL": d_madl, "Hamming": d_hamming})
        # df.to_csv("warehouse_plot_madl_hamming_alt.csv")
        # break


