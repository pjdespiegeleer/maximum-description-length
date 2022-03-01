from typing import List, Set, Dict
import itertools
import numpy as np


class CodingTable:
    def __init__(self, db: List[frozenset], ct: List[frozenset] = None, st: List[frozenset] = None):
        self.db = db
        if st is None:
            self.standard_table = self.get_standard_table(db=db)
        else:
            self.standard_table = st
        if ct is None:
            self.coding_table = self.krimp(db=db)
        else:
            self.coding_table = ct

    def krimp(self, db: List[frozenset]) -> List[frozenset]:
        ct = self.standard_table
        n_st = len(ct)
        candidates = self.generate_candidates(db=db)
        st_can_order = self.get_standard_candidate_order(db=db, candidates=candidates)
        new_count = 0
        old_length = self.get_total_length(db, ct)
        for i, can_index in enumerate(st_can_order):
            candidate = candidates[can_index]
            if candidate in ct:
                continue

            new_ct = ct[:new_count] + [candidate] + ct[-n_st:]
            new_length = self.get_total_length(db=db, ct=new_ct)
            if new_length < old_length:
                old_length = new_length
                new_count += 1
                ct = new_ct

        return ct

    @staticmethod
    def supp(db: List[frozenset], x: frozenset) -> int:
        count = 0
        for t in db:
            if x.issubset(t):
                count += 1
        return count

    def get_standard_cover_order(self, ct: List[frozenset], db: List[frozenset] = None) -> List[int]:
        if db is None:
            db = self.db

        index_list = range(len(db))
        index_list_sorted = sorted(index_list, key=lambda i: (len(ct[i]), self.supp(db=db, x=ct[i])), reverse=True)
        return index_list_sorted

    def get_standard_candidate_order(self, candidates: List[frozenset], db: List[frozenset] = None) -> List[int]:
        if db is None:
            db = self.db

        index_list = range(len(candidates))
        index_list_sorted = sorted(index_list, key=lambda i: (len(candidates[i]), self.supp(db=db, x=candidates[i])), reverse=True)
        return index_list_sorted

    @staticmethod
    def get_cover(t: frozenset, ct: List[frozenset]) -> List[frozenset]:
        t_copy = set(t.copy())
        cover = []
        for x in ct:
            if x.issubset(t_copy):
                cover += [x]
                t_copy.difference_update(x)
            if len(t_copy) == 0:
                return cover

        return cover

    def get_db_cover(self, ct: List[frozenset], db: List[frozenset] = None) -> Dict[frozenset, List[frozenset]]:
        if db is None:
            db = self.db
        d = {}
        for t in db:
            d[t] = self.get_cover(t=t, ct=ct)
        return d

    def get_code_lengths(self, ct: List[frozenset], db: List[frozenset] = None) -> Dict[frozenset, float]:
        if db is None:
            db = self.db
        l: Dict[frozenset, int] = {}

        total_codes = 0
        for t in db:
            code_list = self.get_cover(t=t, ct=ct)
            for c in code_list:
                if c in l.keys():
                    l[c] += 1
                else:
                    l[c] = 1
            total_codes += len(code_list)

        final_len_dict: Dict[frozenset, float] = {}
        for c in l.keys():
            length = l[c]
            new_length: float = length / total_codes
            final_len_dict[c] = -np.log2(new_length)

        return final_len_dict

    def get_item_encode_length(self, t: frozenset, ct: List[frozenset], l: Dict[frozenset, float]) -> float:
        total_length = 0
        code_list = self.get_cover(t=t, ct=ct)
        for c in code_list:
            total_length += l[c]
        return total_length

    def get_db_encode_length(self, db: List[frozenset], ct: List[frozenset]) -> float:
        l: Dict[frozenset, float] = self.get_code_lengths(ct=ct, db=db)
        total_length: float = 0.0
        for t in db:
            total_length += self.get_item_encode_length(t=t, ct=ct, l=l)
        return total_length

    def get_standard_table(self, db: List[frozenset] = None) -> List[frozenset]:
        if db is None:
            db = self.db
        st: List[frozenset] = []
        for t in db:
            for i in t:
                if frozenset({i}) not in st:
                    st += [frozenset({i})]
        return st

    def get_ct_length(self, db: List[frozenset], ct: List[frozenset]) -> float:
        st = self.standard_table
        l_ct = self.get_code_lengths(db=db, ct=ct)
        l_st = self.get_code_lengths(db=db, ct=st)
        total_length = 0
        for c in ct:
            if c in l_ct.keys():
                total_length += l_ct[c]
                for i in c:
                    total_length += l_st[frozenset({i})]

        return total_length

    def get_total_length(self, ct: List[frozenset] = None, db: List[frozenset] = None) -> float:
        if db is None:
            db = self.db
        if ct is None:
            ct = self.coding_table

        # encoding length for database
        db_length = self.get_db_encode_length(db=db, ct=ct)

        # encoding length for code table
        ct_length = self.get_ct_length(db=db, ct=ct)

        total_length: float = db_length + ct_length
        return total_length

    def generate_candidates(self, db: List[frozenset] = None) -> List[frozenset]:
        if db is None:
            db = self.db
        global_lst = []
        for s in db:
            for j in range(2, len(s)+1):
                comb_list = itertools.combinations(s, j)
                new_list = [frozenset(i) for i in comb_list]
                global_lst += new_list
        return list(set(global_lst))


def get_standard_table(db: List[frozenset]) -> List[frozenset]:
        st: List[frozenset] = []
        for t in db:
            for i in t:
                if frozenset({i}) not in st:
                    st += [frozenset({i})]
        return st


def max_description_length(db: List[frozenset], k: int) -> List[frozenset]:
    db = db.copy()
    standard_table = get_standard_table(db=db)
    if k > len(db):
        return db
    sol_set = [db[0]]
    db.pop(0)
    ct = CodingTable(db=sol_set, st=standard_table)
    while len(sol_set) < k:
        j = np.argmax([ct.get_total_length(db=sol_set + [s]) for s in db])
        sol_set += [db[j], ]
        db.pop(j)
        ct = CodingTable(db=sol_set, st=standard_table)
    return sol_set


def krimp_diversity(db: List[frozenset], k: int) -> List[frozenset]:
    l_array = np.zeros(len(db))
    for i in range(len(db)):
        print("Generating KRIMP Nr. ", str(i))
        new_db = db.copy()
        new_db.pop(i)
        ct = CodingTable(db=new_db)
        for j in range(len(db)):
            d = ct.get_total_length(db=[db[j]])
            l_array[j] += d
    index_array = range(len(db))
    final = sorted(index_array, key=lambda i: l_array[i], reverse=True)
    result = []
    for i in final[0:k]:
        result += [db[i], ]
    return result



# # print(get_total_length(db, ct))
# set1 = frozenset({"A", "B", "C"})
# set2 = frozenset({"A", "B"})
# set3 = frozenset({"A"})
# set4 = frozenset({"B"})
#
# db = [set1, set1, set1, set1, set1, set2, set3, set4]
# ct = krimp(db)
# ct0 = [set1, set3, set4]
#
# print(get_total_length(db, ct))
# print(get_total_length(db, ct0))

db = [frozenset({0, 1, 2, 6, 7}), frozenset({1, 2, 6, 7, 9}), frozenset({1, 2, 3, 6, 7, 9}), frozenset({1, 2, 4, 6, 7, 9}),
      frozenset({0, 1, 2, 4, 7, 9}), frozenset({0, 1, 6, 7, 9}), frozenset({0, 1, 2, 3, 7, 9}), frozenset({0, 1, 2, 6, 9}),
      frozenset({0, 1, 3, 6, 7, 9})]
# ct = krimp(db)
# print(ct)
# print(get_total_length(db, ct))
# ct1 = [frozenset({0}), frozenset({1}), frozenset({2}), frozenset({6}), frozenset({7}), frozenset({9}), frozenset({3}), frozenset({4})]
# ct2 = [frozenset({0, 1}), frozenset({0}), frozenset({1}), frozenset({2}), frozenset({6}), frozenset({7}), frozenset({9}), frozenset({3}), frozenset({4})]
# print(get_total_length(db, ct1))
# print(get_total_length(db, ct2))
print(max_description_length(db=db, k=4))
print(krimp_diversity(db=db, k=4))