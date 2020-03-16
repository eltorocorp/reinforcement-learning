package math

// Bellman applies a Bellman equation to recommend a new q-value for a state
// based on the supplied paramters.
// See https://en.wikipedia.org/wiki/Bellman_equation
func Bellman(oldValue, learningRate, reward, discountFactor, optimalFutureValue float64) float64 {
	return oldValue +
		learningRate*(reward+
			discountFactor*optimalFutureValue-
			oldValue)
}

// BayesianAverage returns a bayesian weighted average where:
//   c = A scalar constant, generally set to a value that represents the
//       minimum number of observations required before an observed parameter
//       begins to be more reliable than the estimated parameter.
//       If c < n, BayesianAverage will favor m.
//       If c > n, BayesianAverage will favor v.
//       If c = n, BayesianAverage will treat v and m with equal weight.
//   n = The number of times parameter of value as been observed.
//   m = An estimated parameter value (typically a mean).
//   v = An observed parameter value.
//   see https://en.wikipedia.org/wiki/Bayesian_average
//   see https://fulmicoton.com/posts/bayesian_rating/
func BayesianAverage(c, n, m, v float64) float64 {
	return (c*m + n*v) / (c * n)
}
