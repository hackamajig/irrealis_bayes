import bisect


class PMF(dict):
  '''Dictionary as probability mass function.'''
  def __init__(self, *al, **kw):
    super(PMF, self).__init__(*al, **kw)

  def copy(self):
    return self.__class__(self)

  def total(self):
    return sum(self.itervalues())

  def normalizer(self):
    total = self.total()
    return 1./total if total else float('inf')

  def expectation(self):
    try:
      return sum(hypo*prob for hypo, prob in self.iteritems())
    except TypeError as e:
      raise TypeError("Can't compute expectation of non-numeric hypotheses ({})".format(e))

  def scale(self, factor):
    '''Scale all measures by a common factor.'''
    for key in self:
      self[key] *= factor

  def normalize(self):
    '''Normalize all measures so they sum to one, making this a probability distribution.'''
    self.scale(self.normalizer())

  def uniform_dist(self, hypotheses):
    '''Assign equal probabilities to each of a list of hypotheses.'''
    self.clear()
    for hypothesis in hypotheses:
      self[hypothesis] = 1
    self.normalize()

  def power_law_dist(self, hypotheses, alpha=1.):
    '''Assign power law distribution to each of a list of quantitative hypotheses.'''
    self.clear()
    for hypothesis in hypotheses:
      self[hypothesis] = hypothesis**(-alpha)
    self.normalize()

  def update(self, data):
    '''Updates posterior probability distribution given new data.'''
    for hypothesis in self:
      self[hypothesis] *= self.likelihood(data, given_hypothesis = hypothesis)
    self.normalize()

  def likelihood(self, data, given_hypothesis):
    '''
    Returns likelihood of observed data given a hypothesis. Unimplemented.
    Should be implemented in subclasses.
    '''
    raise NotImplementedError


class CDF(object):
  def __init__(self, items=None, cmp=None, key=None, reverse=False):
    if not items:
      items = []
    if not key:
      key = lambda x: x[0]
    items = list(items)
    items.sort(cmp, key, reverse)
    self.hypotheses, probabilities = zip(*items)
    percentile = 0
    self.percentiles = []
    for probability in probabilities:
      percentile += probability
      self.percentiles.append(percentile)

  def hypothesis(self, percentile):
    index = bisect.bisect(self.percentiles, percentile)
    return self.hypotheses[(index-1) if percentile==self.percentiles[index-1] else index]
