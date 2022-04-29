from generate_diversity_sets import generate_diversity_sets
from hamming_diversity import greedy_hamming
from knapsack import knapsack
from coding_table import CodingTable
import numpy as np
from tqdm import tqdm
import pickle
from madl import max_description_length
seed = 0
np.random.seed(seed)
n = 30
per = 76
r = 100
capacity = 200
bool_list, set_list, value_list = knapsack(n=n, r=r, per=per, save=True, seed=seed, capacity=capacity)
k = 20
cap_list = [10000, 1000, 100]
# cap_list = [100]
for i, cap in enumerate(cap_list):
    print("Number of required solutions = "+str(cap))
    index_list = np.random.permutation(range(len(bool_list)))
    bool_list_cap = [bool_list[j] for j in index_list[:cap]]
    set_list_cap = [set_list[j] for j in index_list[:cap]]
    value_list_cap = [value_list[j] for j in index_list[:cap]]
    for z in tqdm(range(1, 11)):
        seed = z
        np.random.seed(seed)
        optimal_index = np.random.randint(0, len(bool_list_cap))

        base_path = "experiments/knapsack/n_solutions/diversity_sets/randomseed_"+str(seed)+"_n"+str(n)+"_k"+str(k)+"_per"+str(per)+"_cap"+str(cap)+"_"
        mdl, hd = generate_diversity_sets(set_list=set_list_cap, bool_list=bool_list_cap, k=k, optimal_index=optimal_index,
                                          base_path=base_path, save=True, fitness_list=value_list_cap, tie_breaker=False)







