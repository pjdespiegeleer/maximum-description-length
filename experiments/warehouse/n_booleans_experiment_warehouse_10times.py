from generate_diversity_sets import generate_diversity_sets
from hamming_diversity import greedy_hamming
from knapsack import knapsack
from coding_table import CodingTable
import numpy as np
from tqdm import tqdm
import pickle
from madl import max_description_length
from warehouse_location import warehouse_location

nw_list = [5, 6, 4]
ns = 6
r_list = [30, 30, 30]
per_list = [170, 170, 300]
for nw, per, r in zip(nw_list, per_list, r_list):
    print("Warehouses = " + str(nw))
    seed = 0
    np.random.seed(seed)
    cap = 1000
    bool_list, set_list, value_list = warehouse_location(ns=ns, nw=nw, r=r, per=per, save=True, seed=seed, cap=cap)
    k = 20
    for i in tqdm(range(1, 11)):
        seed = i
        np.random.seed(seed)
        optimal_index = np.random.randint(0, len(bool_list))

        base_path = "experiments/warehouse/n_booleans/diversity_sets/cco_randomseed_"+str(seed)+"_ns"+str(ns)+"_nw"+str(nw)+"_r"+str(r)+"_k"+str(k)+"_per"+str(per)+"_"
        mdl, hd = generate_diversity_sets(set_list=set_list, bool_list=bool_list, k=k, optimal_index=optimal_index,
                                          base_path=base_path, save=True, fitness_list=value_list, tie_breaker=False)







