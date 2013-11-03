
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

  def scale(self, factor):
    '''Scale all measures by a common factor.'''
    for key in self: self[key] *= factor

  def normalize(self):
    '''Normalize all measures so they sum to one, making this a probability distribution.'''
    self.scale(self.normalizer())


class BayesPMF(PMF):
  '''
  dict as probability mass function with update of posterior probabilities.

  The update() method calls the likelihood() method, which should be
  implemented in subclasses.
  '''
  def __init__(self, *al, **kw):
    super(BayesPMF, self).__init__(*al, **kw)

  def uniform_priors(self, hypotheses):
    '''Assign equal probabilities to each of a list of hypotheses.'''
    for hypothesis in hypotheses: self[hypothesis] = 1

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
