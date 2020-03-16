package qlearning

// ActionStats contains statistics about an action that has been applied to some
// state.
type ActionStats struct {
	// Calls represents the number of times this action has been called.
	Calls int
	// QValueRaw is the raw q-value associated with this action.
	QValueRaw float64
	// QValueWeighted is the q-value for this action that has been weighted
	// according to the agent's weighting rules.
	QValueWeighted float64
}
