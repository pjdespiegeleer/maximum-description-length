import numpy as np
from cpmpy import *


def warehouse_location(nw: int = 3, ns: int = 5, per: int = 110, save: bool = True, seed: int = 1, cap=None):
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
    num = model.solveAll(display=lambda: bool_list.append(x.value().flatten()))
    print("Number of solutions:", num)
    return bool_list





if __name__ == "__main__":
    warehouse_location()


