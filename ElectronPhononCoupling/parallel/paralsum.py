
import multiprocessing
import numpy as np
import thread

"""
Code snipped from
stackoverflow.com/questions/9068478/how-to-parallelize-a-sum-calculation-in-python-numpy
"""

class Sum:
    def __init__(self):
        self.value = None
        self.lock = thread.allocate_lock()
        self.count = 0

    def add(self, value):
        self.count += 1
        self.lock.acquire()
        if self.value is None:
            self.value = np.zeros(value.shape, dtype=value.dtype)
        self.value += value
        self.lock.release()


def summer(f, args_list, ncpu=1):
    pool = multiprocessing.Pool(processes=ncpu)

    sumArr = Sum()
    for args in args_list:
        singlepoolresult = pool.apply_async(f,args,callback=sumArr.add)

    pool.close()
    pool.join() #waits for all the processes to finish
        
    return sumArr.value
    

