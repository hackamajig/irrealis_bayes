
class PMF(dict):
  '''Dictionary as probability mass function.'''
  def __init__(self, *al, **kw):
    super(PMF, self).__init__(*al, **kw)

  def normalize(self):
    '''Normalize all probabilities (so they sum to one).'''
    total = sum(self.itervalues())
    factor = 1./total if total else float('inf')
    for key in self: self[key] *= factor
