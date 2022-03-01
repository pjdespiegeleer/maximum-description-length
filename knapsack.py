from typing import List

import numpy as np
from itertools import combinations
from cpmpy import *
from main import max_description_length, krimp_diversity
import pickle


def hamming_diversity(e1: List[bool], e2: List[bool]) -> int:
    d = 0
    for i in range(len(e1)):
        if e1[i] != e2[i]:
            d += 1
    return d


# Problem data
n = 20
np.random.seed(1)
R = 100
values = np.random.randint(1, R, n)
weights = np.random.randint(1, R, n)
capacity = int(max([0.5 * sum(weights), R]))


# Construct the model.
x = boolvar(shape=n, name="x")

model = Model(
    sum(x*weights) <= capacity,
    maximize=
    sum(x*values)
)
model.solve()
print(f"Capacity: {capacity}, used: {sum(x.value()*weights)}")
items = np.where(x.value())[0]
print("In items:", items)
max_value = sum(values[items])
print(max_value)

model = Model()
model += (sum(x*weights) <= capacity)
model += (sum(x*values) >= int(max_value*0.95))
count = 0
item_list = []
bool_list = []
# TODO model.solveAll() --> search in documentation
while model.solve():
    count += 1
    # print("Value:", model.solve()) # solve returns objective value
    print(f"Capacity: {capacity}, used: {sum(x.value()*weights)}")
    bool_list += [x.value(),]
    items = np.where(x.value())[0]
    item_list += [items,]
    print("In items:", items)
    print("Total value:  ", sum(values[items]))
    model += ~all(x == x.value())
print(bool_list)
print(item_list)
set_list = []
for l in item_list:
    set_list += [frozenset(l)]
# print(set_list)
# print(krimp(set_list))


def find_diverse_set(sol, div_size: int = 2):
    model = Model()
    x = boolvar(shape=len(sol), name="x")
    model += (div_size == sum(x))
    obj = sum([
        (x[i+1+j] * x[i]) * (np.logical_xor(a, b).sum() + 1)
        for i, a in enumerate(sol[:-1])
        for j, b in enumerate(sol[i+1:])
    ])
    model.maximize(obj)
    model.solve()
    return x.value()

import time
k = int(max([2, int(len(set_list) / 4)]))
begin = time.time()
l = find_diverse_set(sol=bool_list, div_size=k)
# print("Most diverse set :", str(l))
print("Most diverse solutions :")
final_list = []
for i, x in enumerate(l):
    if x:
        final_list += [item_list[i], ]
        print(item_list[i])
with open('hamming.pickle', 'wb') as handle:
    pickle.dump(final_list, handle)
print("Elapsed Time: ", str(time.time()-begin))
print("-----------------------")
begin = time.time()
print("Maximum description length:")
mdl = max_description_length(db=set_list, k=k)
for x in mdl:
    print(x)
with open('mdl.pickle', 'wb') as handle:
    pickle.dump(mdl, handle)
print("Elapsed Time: ", str(time.time()-begin))
print("-----------------------")
begin = time.time()
print("KRIMP Diversity:")
kd = krimp_diversity(db=set_list, k=k)
for x in kd:
    print(x)
with open('kd.pickle', 'wb') as handle:
    pickle.dump(kd, handle)
print("Elapsed Time: ", str(time.time()-begin))
print("-----------------------")
# d = 0
# for i, k in enumerate(l):
#     for j in range(i+1, len(l)):
#         div = hamming_diversity(bool_list[i], bool_list[j])
#         d += div
#         print("Distance between ", str(item_list[i]), " and ", str(item_list[j]), " = ", str(div))
