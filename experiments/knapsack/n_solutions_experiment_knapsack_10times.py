from generate_diversity_sets import generate_diversity_sets
from hamming_diversity import greedy_hamming
from knapsack import knapsack, knapsack_with_limit
from coding_table import CodingTable
import numpy as np
from tqdm import tqdm
import pickle
from madl import max_description_length
from os.path import exists
from pathlib import Path
seed = 0
np.random.seed(seed)
n = 30
per = 86
r = 50
lim = 10
capacity = 200
bool_list, set_list, value_list = knapsack_with_limit(n=n, r=r, per=per, lim=lim, save=False, seed=seed, capacity=capacity)
k = 20
cap_list = [10000, 5000, 1000]
for i, cap in enumerate(cap_list):
    print("Number of required solutions = "+str(cap))
    index_list = np.random.permutation(range(len(bool_list)))
    bool_list_cap = [bool_list[j] for j in index_list[:cap]]
    set_list_cap = [set_list[j] for j in index_list[:cap]]
    value_list_cap = [value_list[j] for j in index_list[:cap]]
    #
    # a = np.array(bool_list_cap)
    # c = a.sum(axis=0)/len(a)
    # d = [x for x in c if x > 0]
    # print(len(d))
    base_string = "experiments/knapsack/knapsack_solutions/knapsack_randomseed_"+str(1)+"_n"+str(n)+"_r"+str(r)+"_lim"+str(lim)+"_per"+str(per) \
                  +"_cap"+str(cap)
    with open(base_string + '_boollist.pickle', 'wb') as handle:
        pickle.dump(bool_list_cap, handle)
    with open(base_string + '_setlist.pickle', 'wb') as handle:
        pickle.dump(set_list_cap, handle)
    for z in tqdm(range(1, 11)):
        seed = z
        np.random.seed(seed)
        optimal_index = np.random.randint(0, len(bool_list_cap))

        base_path = "experiments/knapsack/n_solutions/diversity_sets/cco_randomseed_"+str(seed)+"_n"+str(n)+"_r"+str(r)+"_lim"+str(lim)+"_k"+str(k)+"_per"+str(per)+"_cap"+str(cap)+"_"
        mdl, hd = generate_diversity_sets(set_list=set_list_cap, bool_list=bool_list_cap, k=k, optimal_index=optimal_index,
                                          base_path=base_path, save=True, fitness_list=value_list_cap, tie_breaker=False)







