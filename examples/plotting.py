import paths
import irrealis_bayes as ib
import matplotlib.pyplot as plt

class LocomotiveProblem(ib.PMF):
  def likelihood(self, data, given_hypothesis):
    return 0 if given_hypothesis < data else 1./given_hypothesis

def run_demo():
  probability_mass_function = LocomotiveProblem()
  for upper_bound in (500, 1000, 2000):
    print "Using upper bound of", upper_bound, "on estimated number of locomotives..."
    hypotheses = range(1, upper_bound+1)
    probability_mass_function.power_law_dist(hypotheses)
    for locomotive_number in (60, 30, 90):
      print "Observed locomotive number", locomotive_number
      probability_mass_function.update(locomotive_number)
    print "Expected (mean) number of locomotives:", probability_mass_function.expectation()
    cumulative_distribution_function = ib.CDF(probability_mass_function)
    print "90% credible_interval (from 5% to 95%):", cumulative_distribution_function.percentiles(0.05, 0.95)
    probabilities = [probability_mass_function[hypothesis] for hypothesis in hypotheses]
    plt.plot(hypotheses, probabilities)
    print
  plt.show()

if __name__ == "__main__": run_demo()
