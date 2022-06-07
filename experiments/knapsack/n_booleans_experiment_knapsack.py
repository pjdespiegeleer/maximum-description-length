from generate_diversity_sets import generate_diversity_sets
from hamming_diversity import greedy_hamming
from knapsack import knapsack
from coding_table import CodingTable
import numpy as np
import pickle
from madl import max_description_length
# n_list = [40, 35, 30, 25, 20]
# per_list = [96, 94, 89, 86, 75]
# n_list = [30, 25, 20]
# per_list = [86, 82, 75]
n_list = [80, 80, 80]
per_list = [91, 90, 89]
r_list = [50, 120, 150]
capacity_list = [65, 100, 210]
for n, per, r, capacity in zip(n_list, per_list, r_list, capacity_list):
    print("knapsack items = "+str(n))
    seed = 1
    np.random.seed(seed)
    # n = n
    # per = per
    # r = 100
    cap = 500
    bool_list, set_list, value_list = knapsack(n=n, r=r, per=per, save=True, seed=seed, cap=cap, capacity=capacity)
    k = 20
    optimal_index = np.argmax(value_list)

    base_path = "experiments/knapsack/n_booleans/diversity_sets/randomseed_"+str(seed)+"_n"+str(n)+"_R"+str(r)+"_k"+str(k)+"_per"+str(per)+"_"
    mdl, hd = generate_diversity_sets(set_list=set_list, bool_list=bool_list, k=k, optimal_index=optimal_index,
                                      base_path=base_path, save=True, fitness_list=value_list, tie_breaker=False)







