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

// Value returns the Q-value associated with this StateAction.
func (sa *StateAction) Value() float64 {
	return sa.value
}

// SetValue sets the Q-value associated with this StateAction.
func (sa *StateAction) SetValue(value float64) {
	sa.value = value
}

var _ iface.StateActioner = (*StateAction)(nil)
