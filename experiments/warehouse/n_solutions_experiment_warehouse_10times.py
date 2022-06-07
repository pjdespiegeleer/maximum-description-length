from generate_diversity_sets import generate_diversity_sets
from hamming_diversity import greedy_hamming
from knapsack import knapsack
from coding_table import CodingTable
import numpy as np
from tqdm import tqdm
import pickle
from madl import max_description_length
from warehouse_location import warehouse_location

seed = 0
np.random.seed(seed)
ns = 6
nw = 5
per = 205
r = 80
bool_list, set_list, value_list = warehouse_location(ns=ns, nw=nw, r=r, per=per, save=True, seed=seed)
k = 20
cap_list = [10000, 5000, 1000]
for i, cap in enumerate(cap_list):
    print("Number of required solutions = "+str(cap))
    index_list = np.random.permutation(range(len(bool_list)))
    bool_list_cap = [bool_list[j] for j in index_list[:cap]]
    set_list_cap = [set_list[j] for j in index_list[:cap]]
    value_list_cap = [value_list[j] for j in index_list[:cap]]

    base_string = "experiments/warehouse/warehouse_solutions/warehouse_randomseed_"+str(1)+"_nw"+str(nw)+"_ns"+str(ns)+"_r"+str(r)+"_per"+str(per) \
                  +"_cap"+str(cap)
    with open(base_string + '_boollist.pickle', 'wb') as handle:
        pickle.dump(bool_list_cap, handle)
    with open(base_string + '_setlist.pickle', 'wb') as handle:
        pickle.dump(set_list_cap, handle)
    for z in tqdm(range(1, 11)):
        seed = z
        np.random.seed(seed)
        optimal_index = np.random.randint(0, len(bool_list_cap))

        base_path = "experiments/warehouse/n_solutions/diversity_sets/cco_randomseed_"+str(seed)+"_ns"+str(ns)+"_nw"+str(nw)+"_k"+str(k)+"_per"+str(per)+"_cap"+str(cap)+"_"
        mdl, hd = generate_diversity_sets(set_list=set_list_cap, bool_list=bool_list_cap, k=k, optimal_index=optimal_index,
                                          base_path=base_path, save=True, fitness_list=value_list_cap, tie_breaker=False)







