package qlearning

import "github.com/eltorocorp/reinforcement-learning/pkg/qlearning/iface"

// ActionStats contains statistics about an action that has been applied to some
// state.
type ActionStats struct {
	calls int
	// QValueRaw is the raw q-value associated with this action.
	qValueRaw float64
	// QValueWeighted is the q-value for this action that has been weighted
	// according to the agent's weighting rules.
	qValueWeighted float64
}

// Calls returns the number of times this action has been called.
func (as *ActionStats) Calls() int {
	return as.calls
}

// SetCalls sets the number of times this action has been called.
func (as *ActionStats) SetCalls(n int) {
	as.calls = n
}

// QValueRaw returns the raw q-value for this action.
func (as *ActionStats) QValueRaw() float64 {
	return as.qValueRaw
}

// SetQValueRaw sets the raw q-value for this action.
func (as *ActionStats) SetQValueRaw(value float64) {
	as.qValueRaw = value
}

// QValueWeighted returns the weighted q-value for this action.
func (as *ActionStats) QValueWeighted() float64 {
	return as.qValueWeighted
}

// SetQValueWeighted sets the weighted q-value for this action.
func (as *ActionStats) SetQValueWeighted(value float64) {
	as.qValueWeighted = value
}

var _ iface.ActionStatter = (*ActionStats)(nil)
