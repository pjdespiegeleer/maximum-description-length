import time

from tqdm import tqdm
from typing import List, Set, Dict
import itertools
import numpy as np

from coding_table import CodingTable


class CodingTableBoolean:
    def __init__(self, db: np.ndarray, ct: np.ndarray = None, st: np.ndarray = None):
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

    def krimp(self, db: np.ndarray) -> np.ndarray:
        # The coding table initially starts off as the standard table (all singletons)
        ct = self.standard_table
        n_st = ct.shape[0]
        begin = time.time()
        # all candidates to enter into the coding table are first generated
        candidates = self.generate_candidates(db=db)
        print("Candidate Generation = " + str(time.time()-begin))

        # These candidates are ordered using the standard candidate order.
        # The resulting array is an index-array.
        st_can_order = self.get_standard_candidate_order(db=db, candidates=candidates)

        new_count = 0
        old_length = self.get_total_length(db=db, ct=ct)
        # All candidates are iteratively added to the coding table to see if compression size decreases.
        for i, can_index in enumerate(st_can_order):
            candidate = candidates[can_index]
            # If the candidate is already in the coding table, it can be skipped.
            # if candidate in ct:
            #     continue

            # the new coding table is constructed in such a way that it maintains the standard cover order.
            # new_ct = ct[:new_count] + [candidate] + ct[-n_st:]
            new_ct = np.insert(ct, new_count, candidate, axis=0)
            new_length = self.get_total_length(db=db, ct=new_ct)
            # If the total encoding length decreases, the new coding table replaces the old one.
            # print("Candidate: " + str(candidate))
            # print("Old Length: " + str(old_length))
            # print("New Length: " + str(new_length))
            if new_length < old_length:
                old_length = new_length
                new_count += 1
                ct = new_ct

        return ct

    @staticmethod
    def supp(db: np.ndarray, x: np.ndarray) -> int:
        """
        Method to calculate the support of an item in a database.
        :param db: database, set of items
        :param x: item to calculate the support for
        :return: integer. Supp(db, x)
        """
        summed_db = np.sum(db * x, axis=0)
        count = np.sum(summed_db == np.sum(x))
        return count

    def get_standard_cover_order(self, ct: np.ndarray, db: np.ndarray = None) -> List[int]:
        if db is None:
            db = self.db

        index_list = range(len(db))
        # all indexes are sorted according to two subsequent 'rules': the item-length and the support of the item in
        # the database
        index_list_sorted = sorted(index_list, key=lambda i: (np.sum(ct[i]), self.supp(db=db, x=ct[i])), reverse=True)
        return index_list_sorted

    def get_standard_candidate_order(self, candidates: np.ndarray, db: np.ndarray = None) -> List[int]:
        if db is None:
            db = self.db

        index_list = range(len(candidates[:, 0]))
        # all indexes are sorted according to two subsequent 'rules': the item-length and the support of the item in
        # the database
        index_list_sorted = sorted(index_list, key=lambda i: (np.sum(candidates[i]), self.supp(db=db, x=candidates[i])),
                                   reverse=True)
        return index_list_sorted

    def get_cover(self, t: np.ndarray, ct: np.ndarray) -> np.ndarray:
        """
        This method returns a list of items from the coding table that are used to fully construct the transaction item
        """
        t_copy = t.copy()
        cover = []
        for x in ct:
            if np.sum(t_copy) == 0:
                return np.array(cover)

            if self.is_subset(t=t_copy, x=x):
                cover += [x, ]
                t_copy = t_copy - x

        return np.array(cover)

    @staticmethod
    def is_subset(t: np.ndarray, x: np.ndarray):
        """
        Is x a subset of t?
        """
        return np.sum(x) == np.sum(t * x)

    def get_db_cover(self, ct: np.ndarray, db: np.ndarray = None) -> Dict[np.ndarray, np.ndarray]:
        """
        This method returns a dictionary, which contains pairs of all transaction and their cover from the coding table.
        """
        if db is None:
            db = self.db
        d = {}
        for t in db:
            d[tuple(t)] = self.get_cover(t=t, ct=ct)
        return d

    def get_code_lengths(self, ct: np.ndarray, db: np.ndarray = None) -> Dict[bytes, float]:
        """
        This method returns a dictionary that contains pairs of all items in the coding table, together with their code
        length, that will be used during encoding of the database.
        """
        if db is None:
            db = self.db
        l: Dict[bytes, int] = {}

        total_codes = 0
        for t in db:
            code_list = self.get_cover(t=t, ct=ct)
            for c in code_list:
                c_key = c.tobytes()
                if c_key in l.keys():
                    l[c_key] += 1
                else:
                    l[c_key] = 1
            total_codes += len(code_list)

        final_len_dict: Dict[bytes, float] = {}
        for c in l.keys():
            length = l[c]
            new_length: float = length / total_codes
            final_len_dict[c] = -np.log2(new_length)

        return final_len_dict

    def get_item_encode_length(self, t: np.ndarray, ct: np.ndarray, l: Dict[bytes, float]) -> float:
        """
        Returns the encoding length for one item, using a certain coding table.
        """
        total_length = 0
        code_list = self.get_cover(t=t, ct=ct)
        for c in code_list:
            total_length += l[c.tobytes()]
        return total_length

    def get_db_encode_length(self, db: np.ndarray, ct: np.ndarray, l_ct=None) -> float:
        """
        Returns the encoding length for the entire database (excluding the coding table encoding length)
        """

        total_length: float = 0.0
        for t in db:
            total_length += self.get_item_encode_length(t=t, ct=ct, l=l_ct)
        return total_length

    def get_standard_table(self, db: np.ndarray = None) -> np.ndarray:
        if db is None:
            db = self.db
        n = len(db[0])
        return np.diag(np.full(n, 1))

    def get_ct_length(self, db: np.ndarray, ct: np.ndarray, l_ct=None, l_st=None) -> float:
        """
        Returns the length to encode the coding table itself.
        """
        st = self.standard_table
        # l_ct = self.l_ct
        # l_st = self.get_code_lengths(db=db, ct=st)
        singletons_db = np.diag(np.full(len(db[0]), 1))
        total_length = 0
        for c in ct:
            c_key = c.tobytes()
            if c_key in l_ct.keys():
                total_length += l_ct[c_key]
                for i in np.where(c)[0]:
                    total_length += l_st[singletons_db[i].tobytes()]
        return total_length

    def get_total_length(self, ct: np.ndarray = None, db: np.ndarray = None) -> float:
        """
        Returns the total encoding length
        """

        if db is None:
            db = self.db
        if ct is None:
            ct = self.coding_table
        st = self.standard_table
        l_st = self.get_code_lengths(ct=st, db=db)
        l_ct = self.get_code_lengths(ct=ct, db=db)
        # encoding length for database
        db_length = self.get_db_encode_length(db=db, ct=ct, l_ct=l_ct)

        # encoding length for code table
        ct_length = self.get_ct_length(db=db, ct=ct, l_st=l_st, l_ct=l_ct)

        total_length: float = db_length + ct_length
        return total_length

    def generate_candidates(self, db: np.ndarray = None) -> np.ndarray:
        if db is None:
            db = self.db
        global_set = set()
        for b in db:
            s = frozenset(np.where(b)[0])
            for j in range(2, len(s)+1):
                comb_list = itertools.combinations(s, j)
                new_set = {frozenset(i) for i in comb_list}
                global_set = global_set.union(new_set)
        candidates_bool = []
        for s in list(global_set):
            candidates_bool += [[x in s for x in range(len(db[0]))], ]
        return np.array(candidates_bool)


def get_standard_table(db: np.ndarray) -> np.ndarray:
    return np.diag(np.full(len(db[0]),1))


def generate_candidates(db: np.ndarray = None) -> np.ndarray:
    global_set = set()
    for b in db:
        s = frozenset(np.where(b)[0])
        for j in range(2, len(s)+1):
            comb_list = itertools.combinations(s, j)
            new_set = {frozenset(i) for i in comb_list}
            global_set = global_set.union(new_set)
    candidates_bool = []
    for s in list(global_set):
        candidates_bool += [[x in s for x in range(len(db[0]))], ]
    return np.array(candidates_bool)


if __name__ == "__main__":
    np.random.seed(0)
    db_test = np.random.rand(15, 15).round().astype(int)
    begin = time.time()
    ct = CodingTableBoolean(db=db_test)
    print(ct.coding_table)
    print(time.time()-begin)
    db_set = []
    for b in db_test:
        db_set += [frozenset(np.where(b)[0]), ]
    begin = time.time()
    ct2 = CodingTable(db=db_set)
    print(ct2.coding_table)
    print(time.time()-begin)