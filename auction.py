import numpy as np
from cpmpy import *
import pickle


def auction(n: int = 500, m: int = 100, r: int = 80, per: int = 50, save: bool = True, seed: int = 0, cap=None):
    np.random.seed(seed)
    bid_items = (np.random.rand(n, m) > r/100).astype(int)
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

    fitness_list = []
    for b in bool_list:
        fitness_list.append(sum(b * bid_values))

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
        fitness_list.append(sum(b * bid_values))

    set_list = []
    for l in item_list:
        set_list += [frozenset(l)]

    if save:
        if cap is None:
            base_string = "experiments/auction/auction_solutions/auction_randomseed_"+str(seed)+"_n"+str(n)+"_m"+str(m)+"_r"+str(r)+"_per"+str(per)
        else:
            base_string = "experiments/auction/auction_solutions/auction_randomseed_"+str(seed)+"_n"+str(n)+"_m"+str(m)+"_r"+str(r)+"_per"+str(per) \
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

def auction_with_limit(n: int = 500, m: int = 100, r: int = 80, lim: int = 10, per: int = 50, save: bool = True, seed: int = 0, cap=None):
    np.random.seed(seed)
    bid_items = (np.random.rand(n, m) > r/100).astype(int)
    bid_values = np.random.randint(low=1, high=m**2, size=(n, ))
    x = intvar(lb=0, ub=1, shape=n)
    model = Model()
    model += (sum(x) == lim, )
    for i in range(m):
        model += (sum(x * bid_items[:, i]) <= 1, )
    obj = sum(x * bid_values)
    model.maximize(obj)
    model.solve()
    max_obj = sum(x.value() * bid_values)

    model = Model()
    model += (sum(x) == lim, )
    for i in range(m):
        model += (sum(x * bid_items[:, i]) <= 1, )
    obj = sum(x * bid_values)
    model += (obj >= int(max_obj * per/100))
    bool_list = []
    num = model.solveAll(display=lambda: bool_list.append(x.value()))
    print("Number of solutions:", num)

    fitness_list = []
    for b in bool_list:
        fitness_list.append(sum(b * bid_values))

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
        fitness_list.append(sum(b * bid_values))

    set_list = []
    for l in item_list:
        set_list += [frozenset(l)]

    if save:
        if cap is None:
            base_string = "experiments/auction/auction_solutions/auction_randomseed_"+str(seed)+"_n"+str(n)+"_m"+str(m)+"_r"+str(r)+"_lim"+str(lim)+"_per"+str(per)
        else:
            base_string = "experiments/auction/auction_solutions/auction_randomseed_"+str(seed)+"_n"+str(n)+"_m"+str(m)+"_r"+str(r)+"_lim"+str(lim)+"_per"+str(per) \
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
    n = 25
    r = 90
    lim = 10
    per_list = [85, 83]
    for per in per_list:
        print("N = " + str(n))
        b, s, o = auction_with_limit(save=False, lim=lim, m=10, n=n, r=r, per=per, seed=0)
        a = np.array(b)
        c = a.sum(axis=0) / len(a)
        d = [x for x in c if x > 0]
        print("$N_{eff}$ = " + str(len(d)))


