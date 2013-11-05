irrealis_bayes
==============

I'm studying Allen B. Downey's "Think Bayes: Bayesian Statistics Made Simple",
version 1.0.1. This repository is for tracking exploratory code I write as I
study.

Demo usage
==========

I've placed commented demos in the test cases defined in "irrealis_bayes/tests.py".

Three-step modeling strategy
============================
See test FunctionalTestPMF.test_dice_problem() in "irrealis_bayes/tests.py" for
discussion:
1. Choose a representation for the hypotheses.
2. Choose a representation for the data.
3. Write the likelihood function.

How to run tests
================

To run tests, type "python -m irrealis_bayes.tests".  If you have "nose" and
"coverage" installed, type "nosetests --with-coverage
--cover-package=irrealis_orm".
