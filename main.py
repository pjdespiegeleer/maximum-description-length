from evaluation import evaluate_full_set
from generate_diversity_sets import generate_diversity_sets
from hamming_diversity import greedy_hamming
from knapsack import knapsack
from coding_table import CodingTable
import numpy as np
import pickle
from madl import max_description_length, max_description_length_st

n = 20
r = 100
per = 80
bool_list, set_list, value_list = knapsack(n=n, r=r, per=per, save=True)
k = int(len(bool_list) / 100)
optimal_index = np.argmax(value_list)
print(optimal_index)
base_path = "diversity_sets/n"+str(n)+"_R"+str(r)+"_per"+str(per)+"_k"+str(k)+"_"

mdl, hd = generate_diversity_sets(set_list=set_list, bool_list=bool_list, k=k, optimal_index=optimal_index,
                                  base_path=base_path, save=True, fitness_list=value_list, tie_breaker=False)







