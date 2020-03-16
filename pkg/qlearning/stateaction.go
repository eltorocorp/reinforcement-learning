package qlearning

import "github.com/eltorocorp/markov/pkg/qlearning/iface"

// StateAction is a grouping of an action to a state along with a q-value.
type StateAction struct {
	state  iface.Stater
	action iface.Actioner
	value  float64
}

// NewStateAction creates a new StateAction for a state and action.
func NewStateAction(state iface.Stater, action iface.Actioner, value float64) *StateAction {
	return &StateAction{
		state:  state,
		action: action,
		value:  value,
	}
}

// State returns the StateAction's state.
func (sa *StateAction) State() iface.Stater {
	return sa.state
}

// Action returns the StateAction's action.
func (sa *StateAction) Action() iface.Actioner {
	return sa.action
}

// Transition executes the stateaction's action against the stateaction's state,
// resulting in a transition to a new state.
func (sa *StateAction) Transition() iface.Stater {
	return sa.State().Apply(sa.Action())
}

var _ iface.StateActioner = (*StateAction)(nil)