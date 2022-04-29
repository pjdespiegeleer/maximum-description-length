from auction import auction
from generate_diversity_sets import generate_diversity_sets
import numpy as np
n_list = [35, 33, 30, 28, 25, 22, 20, 16]
per_list = [90, 90, 85, 88, 85, 75, 75, 65]
for n, per in zip(n_list, per_list):
    print("N = "+str(n))
    seed = 0
    np.random.seed(seed)
    # n = 30
    m = 10
    per = per
    cap = 500
    bool_list, set_list, value_list = auction(n=n, m=m, per=per, save=True, seed=seed, cap=cap)
    k = 20
    optimal_index = np.argmax(value_list)

    base_path = "experiments/auction/n_booleans/diversity_sets/randomseed_"+str(seed)+"_n"+str(n)+"_m"+str(m)+"_k"+str(k)+"_per"+str(per)+"_"
    mdl, hd = generate_diversity_sets(set_list=set_list, bool_list=bool_list, k=k, optimal_index=optimal_index,
                                      base_path=base_path, save=True, fitness_list=value_list, tie_breaker=False)







