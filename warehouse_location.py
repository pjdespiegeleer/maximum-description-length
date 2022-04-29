import numpy as np
from cpmpy import *
import pickle


def warehouse_location(nw: int = 3, ns: int = 5, per: int = 110, save: bool = True, seed: int = 0, cap=None):
    np.random.seed(seed)
    warehouse_costs = np.random.randint(size=nw, low=1, high=nw**2)
    transport_costs = np.random.randint(size=(ns, nw), low=1, high=ns**2)
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

    if cap is not None:
        random_perm = np.random.permutation(bool_list)
        bool_list = random_perm[:cap]

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
            base_string = "experiments/warehouse/warehouse_solutions/warehouse_randomseed_"+str(seed)+"_nw"+str(nw)+"_ns"+str(ns)+"_per"+str(per)
        else:
            base_string = "experiments/warehouse/warehouse_solutions/warehouse_randomseed_"+str(seed)+"_nw"+str(nw)+"_ns"+str(ns)+"_per"+str(per) \
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


if __name__ == "__main__":
    nw = 8
    ns = 10
    per_list = [115, 130, 138, 143, 150, 155]
    for per in per_list:
        print("Percentage = "+str(per))
        warehouse_location(save=False, ns=ns, nw=nw, per=per, seed=0)


