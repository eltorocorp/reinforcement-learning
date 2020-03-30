package qlearning

import "github.com/eltorocorp/reinforcement-learning/pkg/qlearning/iface"

// ActionStats contains statistics about an action that has been applied to some
// state.
type ActionStats struct {
	CallCount int
	// QValueRaw is the raw q-value associated with this action.
	QRaw float64
	// QValueWeighted is the q-value for this action that has been weighted
	// according to the agent's weighting rules.
	QWeighted float64
}

// Calls returns the number of times this action has been called.
func (as *ActionStats) Calls() int {
	return as.CallCount
}

// SetCalls sets the number of times this action has been called.
func (as *ActionStats) SetCalls(n int) {
	as.CallCount = n
}

// QValueRaw returns the raw q-value for this action.
func (as *ActionStats) QValueRaw() float64 {
	return as.QRaw
}

// SetQValueRaw sets the raw q-value for this action.
func (as *ActionStats) SetQValueRaw(value float64) {
	as.QRaw = value
}

// QValueWeighted returns the weighted q-value for this action.
func (as *ActionStats) QValueWeighted() float64 {
	return as.QWeighted
}

// SetQValueWeighted sets the weighted q-value for this action.
func (as *ActionStats) SetQValueWeighted(value float64) {
	as.QWeighted = value
}

var _ iface.ActionStatter = (*ActionStats)(nil)
