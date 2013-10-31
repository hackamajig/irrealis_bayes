from setuptools import setup, find_packages
import sys, os

version = '0.1'

setup(name='irrealis_bayes',
      version=version,
      description="Bayesian inference study",
      long_description="""Studying Bayesian stats/inference/probability via Think Bayes""",
      classifiers=[], # Get strings from http://pypi.python.org/pypi?%3Aaction=list_classifiers
      keywords='Bayesian probability',
      author='Kaben Nanlohy',
      author_email='kaben.nanlohy@gmail.com',
      url='https://github.com/kaben/irrealis_bayes',
      license='MIT',
      packages=find_packages(exclude=['ez_setup', 'examples', 'tests']),
      include_package_data=True,
      zip_safe=False,
      install_requires=[
          # -*- Extra requirements: -*-
      ],
      entry_points="""
      # -*- Entry points: -*-
      """,
      )
