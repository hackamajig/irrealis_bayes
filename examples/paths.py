import os, sys

here = os.path.abspath(os.path.dirname(__file__))
libdir = os.path.abspath(os.path.join(here, '..'))
if not libdir in sys.path: sys.path.insert(0, libdir)
