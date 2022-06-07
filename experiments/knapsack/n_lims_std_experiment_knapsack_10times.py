from generate_diversity_sets import generate_diversity_sets
from hamming_diversity import greedy_hamming
from knapsack import knapsack, knapsack_with_limit, knapsack_between_limits
from coding_table import CodingTable
import numpy as np
from tqdm import tqdm
import pickle
from madl import max_description_length
# n_list = [35, 33, 30, 28, 25, 22, 20]
# per_list = [94, 93, 89, 87, 86, 75, 75]
# n_list = [40, 35, 30, 25, 20]
# per_list = [90, 93, 86, 84, 75]
# low_lim = 9
low_lim = 11
high_lim = 11
# n_list = [25, 25, 25]
# per_list = [94, 93, 92]
# r_list = [30, 40, 50]
# capacity_list = [200, 200, 200]
n_list = [25]
per_list = [94]
r_list = [30]
capacity_list = [200]

for n, per, r, capacity in zip(n_list, per_list, r_list, capacity_list):
    print("knapsack items = " + str(n))
    seed = 1
    np.random.seed(seed)
    cap = 1000
    bool_list, set_list, value_list = knapsack_between_limits(n=n, r=r, per=per, low_lim=low_lim, high_lim=high_lim, save=True, seed=seed, cap=cap, capacity=capacity)
    k = 20
    for i in tqdm(range(1, 11)):
        seed = i
        np.random.seed(seed)
        optimal_index = np.random.randint(0, len(bool_list))

        base_path = "experiments/knapsack/n_lims_std/diversity_sets/randomseed_"+str(seed)+"_n"+str(n)+"_r"+str(r)+"_lowlim"+str(low_lim)+"_highlim"+str(high_lim)+"_k"+str(k)+"_per"+str(per)+"_"
        mdl, hd = generate_diversity_sets(set_list=set_list, bool_list=bool_list, k=k, optimal_index=optimal_index,
                                      base_path=base_path, save=True, fitness_list=value_list, tie_breaker=False)







