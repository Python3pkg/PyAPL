import unittest
from PyAPL import apl
from collections import namedtuple
from numpy import testing
import numpy as np

class TestAPLPrograms(unittest.TestCase):

    def test_interp(self):
        self.assertAlmostEqual(float(apl('(÷5-7)+÷15')), -0.43333, places=3)

    def test_vectors(self):
        testing.assert_array_equal(apl('2 3 4 + 1 2 1'), np.array([3.0, 5.0, 5.0]))
        testing.assert_array_equal(apl('1 2 3 4 × 4'), np.array([4.0, 8.0, 12.0, 16.0]))
        testing.assert_array_equal(apl('7 + 4 2 1 5 × ⍳4'), np.array([11.0, 11.0, 10.0, 27.0]))

    def test_complicated(self):
        testing.assert_array_equal(apl('((⍳4) × 4)<(7 + 4.2 1.4 1 5 × ⍳4)'), np.array([1, 1, 0, 1]))

    def test_length(self):
        with self.assertRaises(RuntimeError):
            apl('4 2 1 5 × 1 2 3')
        with self.assertRaises(RuntimeError):
            apl('(⍳6) = (⍳7)')

    def test_logic(self):
        testing.assert_array_equal(apl('~ 1 0 0 0 1'), np.array([0, 1, 1, 1, 0]))

    def test_residue(self):
        testing.assert_array_equal(apl('.1|2.5 3.64 2 ¯1.6'), np.array([0.0, 0.04, 0.0, 0.0]))


if __name__ == '__main__':
    unittest.main()