from evaluation import evaluate_full_set
from generate_diversity_sets import generate_diversity_sets
from hamming_diversity import greedy_hamming
from knapsack import knapsack
from coding_table import CodingTable
import numpy as np
import pickle
from madl import max_description_length, max_description_length_st
n_list = [35, 33, 30, 28, 25, 22, 20, 18, 15]
per_list = [95, 90, 88, 85, 83, 81]
for per in per_list:
    print("Threshold = "+str(per)+"%")
    seed = 0
    np.random.seed(seed)
    n = 30
    r = 100
    # per = per
    # cap = 1000
    bool_list, set_list, value_list = knapsack(n=n, r=r, per=per, save=True, seed=seed)
    k = 20
    optimal_index = np.argmax(value_list)

    base_path = "diversity_sets/randomseed_"+str(seed)+"_n"+str(n)+"_R"+"_k"+str(k)+"_per"+str(per)+"_"
    mdl, hd = generate_diversity_sets(set_list=set_list, bool_list=bool_list, k=k, optimal_index=optimal_index,
                                      base_path=base_path, save=True, fitness_list=value_list, tie_breaker=False)







