from auction import auction, auction_with_limit
from generate_diversity_sets import generate_diversity_sets
import numpy as np
from tqdm import tqdm
n_list = [25, 25, 25]
per_list = [86, 89, 86]
r_list = [95, 95, 95]
lim_list = [8, 10, 12]
for n, r, per, lim in zip(n_list, r_list, per_list, lim_list):
    print("N = "+str(n))
    seed = 0
    np.random.seed(seed)
    m = 20
    per = per
    cap = 1000
    bool_list, set_list, value_list = auction_with_limit(n=n, m=m, r=r, lim=lim, per=per, save=True, seed=seed, cap=cap)
    k = 20
    for i in tqdm(range(1, 11)):
        seed = i
        np.random.seed(seed)
        optimal_index = np.random.randint(0, len(bool_list))

        base_path = "experiments/auction/n_bids/diversity_sets/cco_randomseed_"+str(seed)+"_n"+str(n)+"_m"+str(m)+"_r"+str(r)+"_k"+str(k)+"_lim"+str(lim)+"_per"+str(per)+"_"
        mdl, hd = generate_diversity_sets(set_list=set_list, bool_list=bool_list, k=k, optimal_index=optimal_index,
                                      base_path=base_path, save=True, fitness_list=value_list, tie_breaker=False)







