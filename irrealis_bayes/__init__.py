'''
Probability mass function (PMF) and cumulative distribution function (CDF)
classes for use in study of Allen B. Downey's "Think Bayes: Bayesian Statistics
Made Simple", version 1.0.1.
'''
import bisect


class PMF(dict):
  'Dictionary as probability mass function.'
  def __init__(self, *al, **kw):
    super(PMF, self).__init__(*al, **kw)

  def copy(self):
    'Return a shallow copy of this distribution.'
    return self.__class__(self)

  def total(self):
    'Sum elements of this distribution.'
    return sum(self.itervalues())

  def normalizer(self):
    'Return normalizing constant to scale distribution so it sums to one.'
    total = self.total()
    return 1./total if total else float('inf')

  def expectation(self):
    'Compute the expectation, aka mean, of this distribution.'
    try:
      return sum(hypo*prob for hypo, prob in self.iteritems())
    except TypeError as e:
      raise TypeError("Can't compute expectation of non-numeric hypotheses ({})".format(e))

  def scale(self, factor):
    'Scale all measures by a common factor.'
    for key in self:
      self[key] *= factor

  def normalize(self):
    'Normalize all measures so they sum to one, making this a probability distribution.'
    self.scale(self.normalizer())

  def uniform_dist(self, hypotheses):
    'Assign equal probabilities to each of a list of hypotheses.'
    self.clear()
    for hypothesis in hypotheses:
      self[hypothesis] = 1
    self.normalize()

  def power_law_dist(self, hypotheses, alpha=1.):
    'Assign power law distribution to each of a list of quantitative hypotheses.'
    self.clear()
    for hypothesis in hypotheses:
      self[hypothesis] = hypothesis**(-alpha)
    self.normalize()

  def update(self, data):
    'Updates posterior probability distribution given new data.'
    for hypothesis in self:
      self[hypothesis] *= self.likelihood(data, given_hypothesis = hypothesis)
    self.normalize()

  def likelihood(self, data, given_hypothesis):
    '''
    Returns likelihood of observed data given a hypothesis. Unimplemented.
    Should be implemented in subclasses.
    '''
    raise NotImplementedError


def dict_items_from_data(data):
  'Convert data into a dict, then return its elements as key-value pairs.'
  return dict(data if data else []).items()

def first_element(l):
  'Return the first element of l.'
  return l[0]

def sort_items(items, cmp=None, key=None, reverse=False):
  '''
  Sort item list in-place.
  By default, expects list of dict key-value pairs, and sorts pairs on value.
  '''
  items.sort(cmp=cmp, key=key if key else first_element, reverse=reverse)

def running_sum(l):
  'Generator function to return an iterable running sum of l (which can itself be any iterable).'
  total = 0
  for x in l:
    total += x
    yield total


class CDF(object):
  'Discrete cumulative distribution function.'
  def __init__(self, data=None, cmp=None, key=None, reverse=False):
    items = dict_items_from_data(data)
    sort_items(items, cmp, key, reverse)
    self.hypotheses, probabilities = zip(*items)
    self.cumulative_distribution = tuple(running_sum(probabilities))

  def floor_index(self, probability):
    'Get index of last hypothesis at or below given percentile (specified as probability).'
    index = bisect.bisect(self.cumulative_distribution, probability)
    return index-1 if probability==self.cumulative_distribution[index-1] else index

  def percentile(self, probability):
    'Return hypothesis corresponding to percentile (specified as probability).'
    return self.hypotheses[self.floor_index(probability)]

  def percentiles(self, *probabilities):
    '''
    Return list of hypothesis corresponding list of percentiles (specified as probabilities).
    Use this to obtain a credible interval; for example, to get the 90%
    interval between the fifth and 95th percentiles, write:

    credible_interval = cdf.percentiles(0.05, 0.95)
    '''
    return tuple(self.percentile(probability) for probability in probabilities)
