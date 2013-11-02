from irrealis_bayes import PMF

import unittest


class TestORM(unittest.TestCase):
  def setUp(self):
    self.pmf = PMF(dict.fromkeys('abcde', 1))

  def exercise_pmf(self):
    self.pmf.normalize()
    # Should have six entries corresponding to 'a' through 'f'.
    self.assertEqual(5, len(self.pmf))
    # Sum should be one to within rounding error.
    self.assertTrue(0.999 < sum(self.pmf.itervalues()) < 1.001)

  def test_get_total(self):
    self.assertEqual(5, self.pmf.get_total())

  def test_get_normalizer(self):
    self.assertTrue(0.199 < self.pmf.get_normalizer() < 0.201)

  def test_ints(self):
    self.exercise_pmf()

  def test_floats(self):
    self.pmf = PMF.fromkeys('abcde', 1.)
    self.exercise_pmf()

  def test_zerosum(self):
    self.pmf = PMF.fromkeys('abcde', 0)
    self.pmf.normalize()
    total = sum(self.pmf.itervalues())
    # This is how we verify total is 'nan': only 'nan' is not equal to itself.
    self.assertNotEqual(total, total)


if __name__ == "__main__": unittest.main()
