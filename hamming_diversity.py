import numpy as np

from evaluation import fitness_based_tie_breaker


def find_most_diverse_no_cp(sol, div_set, fitness_list=[], tie_breaker=True):
    div_array = np.array([
        np.array([np.logical_xor(sol[i], a).sum() for a in div_set]).sum() for i in range(len(sol))
    ])
    if tie_breaker:
        index = fitness_based_tie_breaker(value_list=div_array, fitness_list=fitness_list)
    else:
        index = np.argmax(div_array)
    return index


def find_two_most_diverse(sol):
    max_div = 0
    div_set = []
    for i in range(len(sol)):
        for j in range(i+1, len(sol)):
            s = np.logical_xor(sol[i], sol[j]).sum()
            if s > max_div:
                max_div = s
                div_set = [sol[i], sol[j]]

    print(max_div)
    return div_set


def greedy_hamming(sol, k: int = 2, optimal_index: int = 0, fitness_list=[], tie_breaker=False):
    assert(k >= 2)
    sol = sol.copy()
    fitness_list = fitness_list.copy()
    first_sol = sol.pop(optimal_index)
    if tie_breaker:
        fitness_list.pop(optimal_index)
    div_array = [np.logical_xor(first_sol, sol[i]).sum() for i in range(len(sol))]
    if tie_breaker:
        second_index = fitness_based_tie_breaker(value_list=div_array, fitness_list=fitness_list)
        fitness_list.pop(second_index)
    else:
        second_index = np.argmax(div_array)
    div_set = [first_sol, sol.pop(second_index)]

    assert(len(div_set) == 2)

    while len(div_set) < k:
        print("Already found ", str(len(div_set)), " solutions.")
        index = find_most_diverse_no_cp(sol=sol, div_set=div_set, tie_breaker=tie_breaker, fitness_list=fitness_list)

        div_set += [sol.pop(index), ]
        if tie_breaker:
            fitness_list.pop(index)
    return div_set

