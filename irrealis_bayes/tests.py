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

  def test_m_and_m_problem(self):
    '''
    From Downey:
      M&Ms are small candy-coated chocolates
      that come in a variety of colors. Mars,
      Inc., which makes M&Ms, changes the
      mixture of colors from time to time.
      
      In 1995, they introduced blue M&Ms.
      Before then, the color mix in a bag of
      plain M&Ms was 30% Brown, 20% Yellow,
      20% Red, 10% Green, 10% Orange, 10% Tan.
      Afterward it was 24% Blue , 20% Green,
      16% Orange, 14% Yellow, 13% Red, 13%
      Brown.
      
      Suppose a friend of mine has two bags
      of M&Ms, and he tells me that one is
      from 1994 and one from 1996. He won't
      tell me which is which, but he gives me
      one M&M from each bag. One is yellow and
      one is green. What is the probability
      that the yellow one came from the 1994
      bag?

    Hypothesis A: first bag is from 1994, second from 1996.
    Hypothesis B: first from 1996, second from 1994.
    Datum: Yellow from first bag, green from second.

    Assuming hypothesis A, and assuming independence in
    drawing an M&M from each bag, the probability of
    drawing a yellow from the first bag is 0.2, and a
    green from the second is 0.2, so their conjuction has
    probability 0.2*0.2=0.04.

    Assuming hypothesis B, the probability of drawing a
    yellow from the first bag is 0.14, and a green from the
    second is 0.1, so their conjuction has probability
    0.14*0.1=0.014.
    '''
    P_D_given_A = 0.2*0.2
    P_D_given_B = 0.14*0.1
    P_A = 0.5
    P_B = 0.5
    P_A_given_D = P_A*P_D_given_A/(P_A*P_D_given_A + P_B*P_D_given_B)
    self.assertTrue(20./27. - 0.001 < P_A_given_D < 20./27. + 0.001)




if __name__ == "__main__": unittest.main()
