
class PMF(dict):
  '''Dictionary as probability mass function.'''
  def __init__(self, *al, **kw):
    super(PMF, self).__init__(*al, **kw)

  def copy(self):
    return self.__class__(self)

  def get_total(self):
    return sum(self.itervalues())

  def get_normalizer(self):
    total = self.get_total()
    return 1./total if total else float('inf')

  def scale(self, factor):
    for key in self: self[key] *= factor

  def normalize(self):
    '''Normalize all probabilities (so they sum to one).'''
    self.scale(self.get_normalizer())


class BayesPMF(PMF):
  '''
  dict as probability mass function with update of posterior probabilities.

  The update() method calls the likelihood() method, which should be
  implemented in subclasses.
  '''
  def __init__(self, *al, **kw):
    super(BayesPMF, self).__init__(*al, **kw)

  def update(self, data):
    '''Updates posterior  probability distribution given new data.'''
    for hypothesis in self:
      self[hypothesis] *= self.likelihood(data, given = hypothesis)
    self.normalize()

  def likelihood(self, data, given):
    '''
    Returns likelihood of observed data given a hypothesis. Unimplemented.
    Should be implemented in subclasses.
    '''
    raise NotImplementedError
