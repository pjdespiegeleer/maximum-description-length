from generate_diversity_sets import generate_diversity_sets
from hamming_diversity import greedy_hamming
from knapsack import knapsack
from coding_table import CodingTable
import numpy as np
import pickle
from madl import max_description_length
# n_list = [35, 33, 30, 28, 25, 22, 20]
# per_list = [94, 93, 89, 87, 86, 75, 75]
n_list = [40, 35, 30, 25, 20]
per_list = [96, 94, 89, 86, 75]
for n, per in zip(n_list, per_list):
    print("knapsack items = "+str(n))
    seed = 0
    np.random.seed(seed)
    n = n
    per = per
    r = 100
    cap = 500
    bool_list, set_list, value_list = knapsack(n=n, r=r, per=per, save=True, seed=seed, cap=cap)
    k = 20
    optimal_index = np.argmax(value_list)

    base_path = "experiments/knapsack/n_booleans/diversity_sets/randomseed_"+str(seed)+"_n"+str(n)+"_R"+"_k"+str(k)+"_per"+str(per)+"_"
    mdl, hd = generate_diversity_sets(set_list=set_list, bool_list=bool_list, k=k, optimal_index=optimal_index,
                                      base_path=base_path, save=True, fitness_list=value_list, tie_breaker=False)







