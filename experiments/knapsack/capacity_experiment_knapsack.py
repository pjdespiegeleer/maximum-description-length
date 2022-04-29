from generate_diversity_sets import generate_diversity_sets
from hamming_diversity import greedy_hamming
from knapsack import knapsack
from coding_table import CodingTable
import numpy as np
import pickle
from madl import max_description_length
n_list = [30, 50, 70, 80]
capacity_list = [20, 30, 15, 14]
per_list = [90, 90, 90, 90]
for n, per, capacity in zip(n_list, per_list, capacity_list):
    print("Capacity = "+str(capacity))
    seed = 0
    np.random.seed(seed)
    n = n
    per = per
    r = 6
    cap = 500
    bool_list, set_list, value_list = knapsack(capacity=capacity, n=n, r=r, per=per, save=True, seed=seed, cap=cap)
    k = 3
    optimal_index = np.argmax(value_list)

    base_path = "experiments/knapsack/capacity/diversity_sets/randomseed_"+str(seed)+"_n"+str(n)+"r_"+str(r)+"_capacity"+str(capacity)+"_k"+str(k)+"_per"+str(per)+"_"
    mdl, hd = generate_diversity_sets(set_list=set_list, bool_list=bool_list, k=k, optimal_index=optimal_index,
                                      base_path=base_path, save=True, fitness_list=value_list, tie_breaker=False)







