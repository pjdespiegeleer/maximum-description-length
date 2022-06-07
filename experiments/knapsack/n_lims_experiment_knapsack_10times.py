from generate_diversity_sets import generate_diversity_sets
from hamming_diversity import greedy_hamming
from knapsack import knapsack, knapsack_with_limit
from coding_table import CodingTable
import numpy as np
from tqdm import tqdm
import pickle
from madl import max_description_length
# n_list = [35, 33, 30, 28, 25, 22, 20]
# per_list = [94, 93, 89, 87, 86, 75, 75]
# n_list = [40, 35, 30, 25, 20]
# per_list = [90, 93, 86, 84, 75]
lim_list = [8, 10, 12]
n_list = [25, 25, 25]
per_list = [87, 91, 92]
r_list = [50, 50, 50]
capacity_list = [200, 200, 200]
# lim_list = [5, 15]
# n_list = [25, 25]
# per_list = [75, 87]
# r_list = [50, 50]
# capacity_list = [200, 200]
for n, per, r, capacity, lim in zip(n_list, per_list, r_list, capacity_list, lim_list):
    print("knapsack items = " + str(n))
    seed = 1
    np.random.seed(seed)
    cap = 1000
    bool_list, set_list, value_list = knapsack_with_limit(n=n, r=r, per=per, lim=lim, save=True, seed=seed, cap=cap, capacity=capacity)
    k = 20
    for i in tqdm(range(1, 11)):
        seed = i
        np.random.seed(seed)
        optimal_index = np.random.randint(0, len(bool_list))

        base_path = "experiments/knapsack/n_lims/diversity_sets/cco_randomseed_"+str(seed)+"_n"+str(n)+"_r"+str(r)+"_lim"+str(lim)+"_k"+str(k)+"_per"+str(per)+"_"
        mdl, hd = generate_diversity_sets(set_list=set_list, bool_list=bool_list, k=k, optimal_index=optimal_index,
                                      base_path=base_path, save=True, fitness_list=value_list, tie_breaker=False)







