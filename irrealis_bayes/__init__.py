
class PMF(dict):
  '''Dictionary as probability mass function.'''
  def __init__(self, *al, **kw):
    super(self.__class__, self).__init__(*al, **kw)

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
