import numpy as np
from cpmpy import *
import pickle

def job_shop(n: int = 3, m: int = 2, t: int = 3):

    d_jobs = np.random.randint(0, 10, size=(n, t))
    max_t = np.sum(np.sum(d_jobs))
    start_times = intvar(0, max_t, shape=(n, t))
