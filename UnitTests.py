import unittest
from src import apl, APLobj

class TestAPLPrograms(unittest.TestCase):

    def test_interp(self):
        self.assertAlmostEqual(apl('(÷5-7)+÷15').value, -0.43333, places=3)

    def test_vectors(self):
        self.assertEqual(apl('2 3 4 + 1 2 1'), APLobj(value=[3.0, 5.0, 5.0], shape=3))
        self.assertEqual(apl('1 2 3 4 × 4'), APLobj(value=[4.0, 8.0, 12.0, 16.0], shape=4))
        self.assertEqual(apl('7 + 4 2 1 5 × ⍳4'), APLobj(value=[11.0, 11.0, 10.0, 27.0], shape=4))

    def test_complicated(self):
        self.assertEqual(apl('((⍳4) × 4)<(7 + 4.2 1.4 1 5 × ⍳4)'), APLobj(value=[1, 1, 0, 1], shape=4))

    def test_length(self):
        with self.assertRaises(RuntimeError):
            apl('4 2 1 5 × 1 2 3')
        with self.assertRaises(RuntimeError):
            apl('(⍳6) = (⍳7)')

    def test_logic(self):
        self.assertEqual(apl('~ 1 0 0 0 1'), APLobj(value=[0, 1, 1, 1, 0], shape=5))

    def test_residue(self):
        self.assertEqual(apl('.1|2.5 3.64 2 ¯1.6'), APLobj(value=[0.0, 0.04, 0.0, 0.0], shape=4))



if __name__ == '__main__':
    unittest.main()