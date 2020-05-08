import unittest

import sys
import os
sys.path.append(os.path.expanduser("/workspace/code/pose-prediction/src/functions"))
from bin_and_delta import *

class BinAndDeltaTestCase(unittest.TestCase):
    num_bins = [10,10,5,10,10]
    min_bin = [0,0,-2,-0.5,0]
    max_bin = [5,5,3,4.5,5]
    raw_val = [2.4,2.6,-0.3,2.6,-2.75]

    bin_index = [4,5,1,6,0]
    delta = [0.15,-0.15,0.2,-0.15,-3]

    def test_get_bin_idx_and_delta(self):
        for i in range(len(self.num_bins)):
            bin_index, delta = get_bin_idx_and_delta(self.num_bins[i],self.min_bin[i],self.max_bin[i],self.raw_val[i])
            self.assertEqual(bin_index, self.bin_index[i])
            self.assertEqual(delta, np.float32(self.delta[i]))

    def test_get_value_from_bin_idx_and_delta(self):
        for i in range(len(self.num_bins)):
            value = get_value_from_bin_idx_and_delta(self.num_bins[i],self.min_bin[i],self.max_bin[i],self.bin_index[i],self.delta[i])
            self.assertEqual(value, np.float32(self.raw_val[i]))

if __name__ == '__main__': 
    import timeit
    setup = "from __main__ import BinAndDeltaTestCase; tmp = BinAndDeltaTestCase()"
    print(timeit.timeit("tmp.test_get_bin_idx_and_delta()", setup=setup, number=100))
    print(timeit.timeit("tmp.test_get_value_from_bin_idx_and_delta()", setup=setup, number=100))

    unittest.main()
    