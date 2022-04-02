import numpy as np
from cpmpy import *


def auction(n: int = 500, m: int = 100, per: int = 50, save: bool = True, seed: int = 1, cap=None):
    np.random.seed(1)
    bid_items = (np.random.rand(n, m) > 0.8).astype(int)
    bid_values = np.random.randint(low=1, high=m**2, size=(n, ))
    x = intvar(lb=0, ub=1, shape=n)
    model = Model()
    for i in range(m):
        model += (sum(x * bid_items[:, i]) <= 1, )
    obj = sum(x * bid_values)
    model.maximize(obj)
    model.solve()
    max_obj = sum(x.value() * bid_values)

    model = Model()
    for i in range(m):
        model += (sum(x * bid_items[:, i]) <= 1, )
    obj = sum(x * bid_values)
    model += (obj >= int(max_obj * per/100))
    bool_list = []
    num = model.solveAll(display=lambda: bool_list.append(x.value()))
    print("Number of solutions:", num)
    return bool_list




if __name__ == "__main__":
    auction()
