import concurrent.futures
import numpy as np
import torch
import multiprocessing as mp

class b:
    def __init__(self,):
        self.a = np.arange(12)

    def try_(self, ind):
        k = ind**2
        return ind, k, self.a[ind]

    def test(self,):

        # with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
        #     output = executor.map(self.try_, np.arange(10))
        pool = mp.Pool(processes=3)
        output_list = pool.map(self.try_, torch.arange(12))

        for i in output_list:
            print(i[0], i[1], i[2])
          
        
bb = b()
bb.test()
