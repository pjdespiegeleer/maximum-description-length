import time
from coding_table import CodingTable, get_standard_table
import numpy as np
from typing import List

from evaluation import fitness_based_tie_breaker


def max_description_length(db: List[frozenset], k: int, optimal_index: int = 0, fitness_list=[], tie_breaker=True) -> List[frozenset]:
    db = db.copy()
    fitness_list = fitness_list.copy()
    standard_table = get_standard_table(db=db)
    print(standard_table)
    if k > len(db):
        return db
    sol_set = [db[optimal_index]]
    db.pop(optimal_index)
    fitness_list.pop(optimal_index)
    ct = CodingTable(db=sol_set, st=standard_table)
    print("Iteration 1: " + str(ct.coding_table))
    while len(sol_set) < k:
        begin = time.time()
        print(len(sol_set))
        div_array = [ct.get_total_length(db=sol_set + [s]) for s in db]
        if tie_breaker:
            j = fitness_based_tie_breaker(value_list=div_array, fitness_list=fitness_list)
        else:
            j = np.argmax(div_array)
        sol_set += [db[j], ]
        db.pop(j)
        fitness_list.pop(optimal_index)
        ct = CodingTable(db=sol_set, st=standard_table)
        print("Iteration " + str(len(sol_set)) + ": " + str(ct.coding_table))
        print(time.time()-begin)
    return sol_set


def max_description_length_st(db: List[frozenset], k: int, optimal_index: int = 0) -> List[frozenset]:
    db = db.copy()
    standard_table = get_standard_table(db=db)
    if k > len(db):
        return db
    sol_set = [db.pop(optimal_index)]

    ct = CodingTable(db=sol_set, ct=standard_table, st=standard_table)
    while len(sol_set) < k:
        print(len(sol_set))
        j = np.argmax([ct.get_total_length(db=sol_set + [s]) for s in db])
        sol_set += [db.pop(j), ]

    return sol_set
