import time
from coding_table import CodingTable, get_standard_table
import numpy as np
from typing import List

# from evaluation import fitness_based_tie_breaker


def fitness_based_tie_breaker(value_list, fitness_list):
    max_indices = np.argwhere(value_list == np.amax(value_list)).flatten().tolist()
    best_fitness = np.argmax([fitness_list[i] for i in max_indices])
    best_index = max_indices[best_fitness]
    return best_index


def max_description_length(db: List[frozenset], k: int, optimal_index: int = 0, fitness_list=[], tie_breaker=False) -> List[frozenset]:
    if k > len(db):
        return db
    db = db.copy()
    fitness_list = fitness_list.copy()

    # generate standard table = coding table made up of singletons
    standard_table = get_standard_table(db=db)

    # First solution in the diversity set is the one with the best fitness value
    sol_set = [db[optimal_index]]
    db.pop(optimal_index)

    # If ties are decided through a tie-breaker, we need to keep the fitness list up to date as well
    if tie_breaker:
        fitness_list.pop(optimal_index)
    # A coding table is generated through the KRIMP algorithm
    ct = CodingTable(db=sol_set, st=standard_table)

    # Iterively add solutions to the diversity set
    while len(sol_set) < k:
        begin = time.time()
        print("Iteration = " + str(len(sol_set)))

        # The coding table from the previous step is used to calculate the encoding length of a new collection:
        # the previously found diverse solutions + one of the solutions that is not chosen yet
        div_array = [ct.get_total_length(db=sol_set + [s]) for s in db]

        # The solutions that has the highest encoding length with the coding table, is chosen as the newest member of
        # the diverse set
        # If a tie-breaker is used, the fitnesses of all the solutions with the largest encoding length is used to decide
        # who is added.
        if tie_breaker:
            j = fitness_based_tie_breaker(value_list=div_array, fitness_list=fitness_list)
            fitness_list.pop(j)
        else:
            j = np.argmax(div_array)
        # print("Evaluation Time = " + str(time.time()-begin))
        # begin2 = time.time()
        # A new coding table is calculated, again using KRIMP, but now for the old set + the new solution
        sol_set += [db[j], ]
        ct = CodingTable(db=sol_set, st=standard_table)
        print("Coding Table Time = " + str(time.time()-begin))
    return sol_set


if __name__ == "__main__":
    knapsack_size = 20
    total_solution_size = 1000
    diverse_solution_size = 20
    bool_list = np.random.random((total_solution_size, knapsack_size)).round().astype(bool)
    db = []
    for bool_x in bool_list:
        db += [frozenset(np.where(bool_x)[0]), ]

    madl = max_description_length(db=db, k=diverse_solution_size, optimal_index=0)
