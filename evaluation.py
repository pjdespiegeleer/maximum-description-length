from typing import List
from scipy.stats import entropy
import numpy as np


def kl_divergence(original: List[List[int]], sol_set: List[List[int]], n=int):
    # generate entropy-tables for both the original set & the chosen diversity set
    p = generate_entropy_table(sol_set=sol_set, n=n)
    q = generate_entropy_table(sol_set=original, n=n)
    total_kl = entropy(pk=p, qk=q)
    return total_kl


def generate_entropy_table(sol_set, n):
    table = []
    for i in range(n):
        count = 0
        for s in sol_set:
            if i in s:
                count += 1
        table += [count / len(sol_set), ]
    return table


def fitness_based_tie_breaker(value_list, fitness_list):
    max_indices = np.argwhere(value_list == np.amax(value_list)).flatten().tolist()
    best_fitness = np.argmax([fitness_list[i] for i in max_indices])
    best_index = max_indices[best_fitness]
    return best_index


if __name__ == "__main__":
    f = [1,2,3,10,4,6,7,8]
    v = [1,2,4,4,4,1,1,1]
    print(fitness_based_tie_breaker(v, f))