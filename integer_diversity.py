import numpy as np

from generate_diversity_sets import generate_diversity_sets
from knapsack import integer_knapsack, integer_list_to_bool_list


def manhattan_distance(lst1, lst2):
    return np.sum(np.abs(np.asarray(lst1)-np.asarray(lst2)))


def hamming_distance(lst1, lst2):
    return np.sum(np.logical_xor(lst1, lst2))

seed = 0
np.random.seed(seed)
n = 10
r = 500
max_c = 4
lst = [np.random.randint(low=0, high=max_c-1, size=4) for _ in range(200)]
bool_list = []
set_list = []
for i, x in enumerate(lst):
    bool_x = integer_list_to_bool_list(lst=x, max_c=max_c)
    bool_list += [bool_x, ]
    set_list += [frozenset(np.where(bool_x)[0]), ]
k = 20
optimal_index = 0
import pickle
with open('knapsack_solutions/integer_diversity/test_setlist_1000.pickle', 'wb') as handle:
    pickle.dump(set_list, handle)
    print("Saved set list")
# print(set_list)
# print(bool_list)
base_path = "diversity_sets/integer_diversity/random_test_1000_"
mdl, hd = generate_diversity_sets(set_list=set_list, bool_list=bool_list, k=k, optimal_index=optimal_index,
                                  base_path=base_path, save=True, tie_breaker=False)

# bool_list = integer_list_to_bool_list(lst=lst, max_c=4)
# base_path = "diversity_sets/randomseed_"+str(seed)+"_n"+str(n)+"_R"+"_k"+str(k)+"_per"+str(per)+"_"
# mdl, hd = generate_diversity_sets(set_list=set_list, bool_list=bool_list, k=k, optimal_index=optimal_index,
#                                   base_path=base_path, save=True, fitness_list=value_list, tie_breaker=False)