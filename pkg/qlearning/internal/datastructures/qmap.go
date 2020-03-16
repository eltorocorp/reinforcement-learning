package datastructures

import "github.com/eltorocorp/markov/pkg/qlearning/iface"

// QMap is a mapping of states to actions (and each action's q-value).
type QMap map[string]map[string]float64

// NewQMap returns a new QMap
func NewQMap() *QMap {
	qmap := (QMap)(map[string]map[string]float64{})
	return &qmap
}

// GetValue return the q-value for a given state and action.
func (qq *QMap) GetValue(state iface.Stater, action iface.Actioner) float64 {
	return qq.GetActionsForState(state)[action.String()]
}

// SetValue sets the value of a given state and action.
func (qq *QMap) SetValue(stateAction iface.StateActioner, value float64) {
	qq.GetActionsForState(stateAction.State())[stateAction.Action().String()] = value
}

// GetActionsForState returns the actions associated with a given state.
func (qq *QMap) GetActionsForState(state iface.Stater) map[string]float64 {
	if _, exists := (*qq)[state.String()]; !exists {
		(*qq)[state.String()] = make(map[string]float64)
	}
	return (*qq)[state.String()]
}
