import time

from tqdm import tqdm
from typing import List, Set, Dict
import itertools
import numpy as np


class CodingTable:
    def __init__(self, db: List[frozenset], ct: List[frozenset] = None, st: List[frozenset] = None):
        self.db = db

        # If no standard table is given, then it can be generated from the given database of items
        if st is None:
            self.standard_table = self.get_standard_table(db=db)
        else:
            self.standard_table = st

        # If no coding table is given at initialization,
        # then it will be automatically generated using the krimp algorithm
        if ct is None:
            self.coding_table = self.krimp(db=db)
        else:
            self.coding_table = ct

    def krimp(self, db: List[frozenset]) -> List[frozenset]:
        # The coding table initially starts off as the standard table (all singletons)
        ct = self.standard_table
        n_st = len(ct)
        begin = time.time()
        # all candidates to enter into the coding table are first generated
        candidates = self.generate_candidates(db=db)

        # These candidates are ordered using the standard candidate order.
        # The resulting array is an index-array.
        st_can_order = self.get_standard_candidate_order(db=db, candidates=candidates)

        new_count = 0
        old_length = self.get_total_length(db=db, ct=ct)
        # All candidates are iteratively added to the coding table to see if compression size decreases.
        for i, can_index in enumerate(st_can_order):
            candidate = candidates[can_index]
            # If the candidate is already in the coding table, it can be skipped.
            if candidate in ct:
                continue

            # the new coding table is constructed in such a way that it maintains the standard cover order.
            new_ct = ct[:new_count] + [candidate] + ct[-n_st:]
            new_ct = self.get_standard_cover_order(ct=new_ct, db=db)
            new_length = self.get_total_length(db=db, ct=new_ct)
            # If the total encoding length decreases, the new coding table replaces the old one.
            if new_length < old_length:
                old_length = new_length
                new_count += 1
                ct = new_ct

        return ct

    @staticmethod
    def supp(db: List[frozenset], x: frozenset) -> int:
        """
        Method to calculate the support of an item in a database.
        :param db: database, set of items
        :param x: item to calculate the support for
        :return: integer. Supp(db, x)
        """
        count = 0
        for t in db:
            if x.issubset(t):
                count += 1
        return count

    def get_standard_cover_order(self, ct: List[frozenset], db: List[frozenset] = None) -> List[frozenset]:
        if db is None:
            db = self.db
        # all indexes are sorted according to two subsequent 'rules': the item-length and the support of the item in
        # the database
        list_sorted = sorted(ct, key=lambda i: (len(i), self.supp(db=db, x=i)), reverse=True)
        # index_list_sorted = sorted(index_list, key=lambda i: (len(ct[i]), 1/self.supp(db=db, x=ct[i])), reverse=False)
        return list_sorted

    def get_standard_candidate_order(self, candidates: List[frozenset], db: List[frozenset] = None) -> List[int]:
        if db is None:
            db = self.db

        index_list = range(len(candidates))
        # all indexes are sorted according to two subsequent 'rules': the item-length and the support of the item in
        # the database
        # index_list_sorted = sorted(index_list, key=lambda i: (len(candidates[i])), self.supp(db=db, x=candidates[i])
        #                            reverse=True)
        index_list_sorted = sorted(index_list, key=lambda i: (self.supp(db=db, x=candidates[i]), len(candidates[i])),
                               reverse=True)
        return index_list_sorted

    @staticmethod
    def get_cover(t: frozenset, ct: List[frozenset]) -> List[frozenset]:
        """
        This method returns a list of items from the coding table that are used to fully construct the transaction item
        """
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
        """
        This method returns a dictionary, which contains pairs of all transaction and their cover from the coding table.
        """
        if db is None:
            db = self.db
        d = {}
        for t in db:
            d[t] = self.get_cover(t=t, ct=ct)
        return d

    def get_code_lengths(self, ct: List[frozenset], db: List[frozenset] = None) -> Dict[frozenset, float]:
        """
        This method returns a dictionary that contains pairs of all items in the coding table, together with their code
        length, that will be used during encoding of the database.
        """
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
        """
        Returns the encoding length for one item, using a certain coding table.
        """
        total_length = 0
        code_list = self.get_cover(t=t, ct=ct)
        for c in code_list:
            total_length += l[c]
        return total_length

    def get_db_encode_length(self, db: List[frozenset], ct: List[frozenset]) -> float:
        """
        Returns the encoding length for the entire database (excluding the coding table encoding length)
        """
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

    def get_ct_length(self, db: List[frozenset], ct: List[frozenset] = None) -> float:
        """
        Returns the length to encode the coding table itself.
        """
        if ct is None:
            ct = self.coding_table
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
        """
        Returns the total encoding length
        """

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
        # return total_length

    def generate_candidates(self, db: List[frozenset] = None) -> List[frozenset]:
        if db is None:
            db = self.db
        global_set = set()
        for s in db:
            for j in range(2, len(s)+1):
                comb_list = itertools.combinations(s, j)
                new_set = {frozenset(i) for i in comb_list}
                global_set = global_set.union(new_set)
        return list(global_set)


def get_standard_table(db: List[frozenset]) -> List[frozenset]:
    st: List[frozenset] = []
    for t in db:
        for i in t:
            if frozenset({i}) not in st:
                st += [frozenset({i})]
    return st


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