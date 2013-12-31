'''
Following code and probability problems are draw
from and based on the following text:

Downey, Allen B.
"Think Bayes: Bayesian Statistics Made Simple."
Version 1.0.1. Needham, Massachusetts:
Green Tea Press, 2012. Print and electronic
versions. Accessed 26 December 2013.
www.greenteapress.com/thinkbayes/thinkbayes.pdf.
'''

import irrealis_bayes
import unittest


class Brainstorm(unittest.TestCase):
  def test_cookie_problem(self):
    '''
    From Downey:
      Suppose there are two bowls of cookies.
      Bowl 1 contains 30 vanilla cookies and
      10 chocolate cookies. Bowl 2 contains
      20 of each.

      Now suppose you choose one of the bowls
      at random and, without looking, select
      a cookie at random. The cookie is
      vanilla. What is the probability that
      it came from Bowl 1?
    '''
    P_V_given_B1 = 30./(30.+10.) # Likelihood of a vanilla cookie from bowl 1.
    P_V_given_B2 = 20./(20.+20.) # Likelihood of a vanilla cookie from bowl 2.
    P_B1 = 1./2. # Prior probability of bowl 1.
    P_B2 = 1./2. # Prior probability of bowl 2.
    # Posterior probability of bowl 1 given a randomly-selected vanilla cookie:
    P_B1_given_V = P_B1 * P_V_given_B1 / (P_B1*P_V_given_B1 + P_B2*P_V_given_B2)
    self.assertTrue((3./5. - 0.001) < P_B1_given_V < (3./5. + 0.001))


if __name__ == "__main__": unittest.main()
