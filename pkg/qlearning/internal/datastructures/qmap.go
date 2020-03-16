package datastructures

import "github.com/eltorocorp/markov/pkg/qlearning/iface"

// QMap is a mapping of states to actions (and each action's q-value).
type QMap map[string]map[string]iface.ActionStatter

// NewQMap returns a new QMap
func NewQMap() *QMap {
	qmap := (QMap)(map[string]map[string]iface.ActionStatter{})
	return &qmap
}

// GetStats returns the stats for a given state and action.
func (qq *QMap) GetStats(state iface.Stater, action iface.Actioner) iface.ActionStatter {
	return qq.GetActionsForState(state)[action.String()]
}

// UpdateStats updates the stats of a given state and action.
func (qq *QMap) UpdateStats(stateAction iface.StateActioner, stats iface.ActionStatter) {
	qq.GetActionsForState(stateAction.State())[stateAction.Action().String()] = stats
}

// GetActionsForState returns the actions associated with a given state.
func (qq *QMap) GetActionsForState(state iface.Stater) map[string]iface.ActionStatter {
	if _, exists := (*qq)[state.String()]; !exists {
		(*qq)[state.String()] = make(map[string]iface.ActionStatter)
	}
	return (*qq)[state.String()]
}
