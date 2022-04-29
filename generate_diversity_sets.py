import numpy as np

from madl import max_description_length
from hamming_diversity import greedy_hamming
import pickle


def generate_diversity_sets(set_list, bool_list, k=None, optimal_index: int = 0, save=True, base_path="",
                            fitness_list=[], tie_breaker=False):
    if k is None:
        k = int(len(set_list) / 100)

    mdl = max_description_length(db=set_list, k=k, optimal_index=optimal_index,
                                 fitness_list=fitness_list, tie_breaker=tie_breaker)
    mdl_list = []
    for x in mdl:
        mdl_list += [list(x), ]

    if save:
        with open(base_path+'mdl.pickle', 'wb') as handle:
            pickle.dump(mdl, handle)

    hd = greedy_hamming(sol=bool_list, k=k, optimal_index=optimal_index,
                        fitness_list=fitness_list, tie_breaker=tie_breaker)
    hd_list = []
    for x in hd:
        item = np.where(x)[0]
        hd_list += [item, ]

    if save:
        with open(base_path + 'hamming.pickle', 'wb') as handle:
            pickle.dump(hd, handle)

    return mdl, hd
