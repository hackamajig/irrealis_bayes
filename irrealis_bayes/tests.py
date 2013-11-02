from irrealis_bayes import PMF

import unittest


class UnitTestPMF(unittest.TestCase):
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

  def test_copy(self):
    '''
    test_copy

    Verify that pmf.copy() copies data into a new, independent PMF instance.
    '''
    pmf2 = self.pmf.copy()
    pmf2.normalize()
    for key in self.pmf:
      self.assertEqual(1, self.pmf[key])
      self.assertTrue(0.199 < pmf2[key] < 0.201)


class FunctionalTestPMF(unittest.TestCase):
  def test_cookie_problem(self):
    '''
    test_cookie_problem

    Suppose there are two bowls of cookies. The first bowl contains 30 vanilla
    cookies and ten chocolate cookies. The second bowl contains twenty of each.
    Now suppose you choose one of the bowls at random and, without looking,
    select a cookie from bowl at random. The cookie is vanilla. What is the
    probability that it came from the first bowl?

    Prior to choosing the cookie, the probability P(bowl_1) of choosing the
    first bowl was 0.5 (since we were equally likely to choose either bowl).

    Assuming we had chosen the first bowl, the likelihood P(vanilla | bowl_1)
    of choosing a vanilla cookie was 0.75 (30 vanilla cookies out a total of
    forty cookies in the first bowl). On the other hand, assuming we had chosen
    the second bowl, the likelihood P(vanilla | bowl_2) of choosing a vanilla
    cookie was 0.5 (twenty vanilla cookies out of 40 cookies in the second
    bowl).

    Since our hypotheses (bowl one or bowl two) are exclusive and exhaustive,
    the law of total probability gives:
    
      P(bowl_1 | vanilla)
      = (P(bowl_1)*P(vanilla | bowl_1)) / (P(bowl_1)*P(vanilla | bowl_1) + P(bowl_2)*P(vanilla | bowl_2))
      = (0.5*0.75)/(0.5*0.75 + 0.5*0.5)
      = (0.75)/(0.75 + 0.5)
      = 0.6
    '''
    pmf = PMF(bowl_1 = 0.5, bowl_2 = 0.5)
    pmf['bowl_1'] *= 0.75
    pmf['bowl_2'] *= 0.5
    pmf.normalize()
    self.assertTrue(0.599 < pmf['bowl_1'] < 0.601)


if __name__ == "__main__": unittest.main()
