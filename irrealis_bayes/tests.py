import irrealis_bayes
import unittest


class UnitTestHello(unittest.TestCase):
  def test_hello(self):
    self.assertTrue(True)

  def test_fail(self):
    self.assertTrue(False)


if __name__ == "__main__": unittest.main()
