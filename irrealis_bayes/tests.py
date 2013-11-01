from irrealis_bayes import PMF

import unittest


class TestORM(unittest.TestCase):
  def exercise_pmf(self):
    self.pmf.normalize()
    # Should have six entries corresponding to 'a' through 'f'.
    self.assertEqual(6, len(self.pmf))
    # Sum should be one to within rounding error.
    self.assertTrue(0.999 < sum(self.pmf.itervalues()) < 1.001)

  def test_ints(self):
    self.pmf = PMF(dict.fromkeys('abcdef', 1))
    self.exercise_pmf()

  def test_floats(self):
    self.pmf = PMF(dict.fromkeys('abcdef', 1.))
    self.exercise_pmf()

  def test_zerosum(self):
    self.pmf = PMF(dict.fromkeys('abcdef', 0))
    self.pmf.normalize()
    total = sum(self.pmf.itervalues())
    # This is how we verify total is 'nan': only 'nan' is not equal to itself.
    self.assertNotEqual(total, total)


if __name__ == "__main__": unittest.main()
