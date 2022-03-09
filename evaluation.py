from typing import List
import numpy as np


def evaluate_diversity(sol_set: List[List[int]], n: int, average=False):
    s = [[(k in sol_set[i]) for k in range(n)] for i in range(len(sol_set))]
    total_sum = 0
    count = 0
    for i, a in enumerate(s):
        for j, b in enumerate(s[i+1:]):
            total_sum += np.logical_xor(a,b).sum()
            count += 1
    if average:
        return total_sum / count
    return total_sum


def generate_entropy_table(sol_set: List[List[int]], n: int):
    table = {}
    for i in range(n):
        count = 0
        for s in sol_set:
            if i in s:
                count += 1
        table[i] = -np.log2(count / len(sol_set))
    return table


def evaluate_diversity_entropy(sol_set: List[List[int]], n: int, average=False):
    total_entropy = 0
    t = generate_entropy_table(sol_set=sol_set, n=n)

    for s in sol_set:
        H = 0
        for x in s:
            H += t[x]
        total_entropy += H

    if average:
        return total_entropy / len(sol_set)

    return total_entropy


def evaluate_full_set(sol_list: List[List[int]], n: int):
    result_list = []
    div_length = range(2, len(sol_list)+1)
    for x in div_length:
        e = evaluate_diversity_entropy(sol_set=sol_list[:x], n=n, average=True)
        result_list += [e, ]
    return result_list


def fitness_based_tie_breaker(value_list, fitness_list):
    max_indices = np.argwhere(value_list == np.amax(value_list)).flatten().tolist()
    best_fitness = np.argmax([fitness_list[i] for i in max_indices])
    best_index = max_indices[best_fitness]
    return best_index


if __name__ == "__main__":
    f = [1,2,3,10,4,6,7,8]
    v = [1,2,4,4,4,1,1,1]
    print(fitness_based_tie_breaker(v, f))