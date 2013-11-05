import bisect


class PMF(dict):
  '''Dictionary as probability mass function.'''
  def __init__(self, *al, **kw):
    super(PMF, self).__init__(*al, **kw)

  def copy(self):
    '''Return a shallow copy of this distribution.'''
    return self.__class__(self)

  def total(self):
    '''Sum elements of this distribution.'''
    return sum(self.itervalues())

  def normalizer(self):
    '''Return normalizing constant to scale distribution so it sums to one.'''
    total = self.total()
    return 1./total if total else float('inf')

  def expectation(self):
    '''Compute the expectation, aka mean, of this distribution.'''
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


def first_element(l):
  return l[0]

def dict_items_from_data(data):
  return dict(data if data else []).items()

def sort_items(items, cmp, key, reverse):
  items.sort(cmp=cmp, key=key if key else first_element, reverse=reverse)

def running_sum(l):
  total = 0
  for x in l:
    total += x
    yield total


class CDF(object):
  def __init__(self, data=None, cmp=None, key=None, reverse=False):
    items = dict_items_from_data(data)
    sort_items(items, cmp, key, reverse)
    self.hypotheses, probabilities = zip(*items)
    self.cumulative_distribution = list(running_sum(probabilities))

  def _floor_index(self, index, probability):
    return index-1 if probability==self.cumulative_distribution[index-1] else index

  def percentile(self, probability):
    index = bisect.bisect(self.cumulative_distribution, probability)
    return self.hypotheses[self._floor_index(index, probability)]

  def percentiles(self, *probabilities):
    return tuple(self.percentile(probability) for probability in probabilities)
