package datastructures

import "github.com/eltorocorp/reinforcement-learning/pkg/qlearning/iface"

// QMap is a mapping of states to actions (and each action's q-value).
type QMap struct {
	data map[string]map[string]iface.ActionStatter
}

// NewQMap returns a new QMap
func NewQMap() *QMap {
	return &QMap{
		data: map[string]map[string]iface.ActionStatter{},
	}
}

// GetStats returns the stats for a given state and action.
// If a the specified action has not been recorded for the given state, the
// method will return nil, false.
func (qq *QMap) GetStats(state iface.Stater, action iface.Actioner) (stats iface.ActionStatter, found bool) {
	actions := qq.GetActionsForState(state)
	stats, found = actions[action.String()]
	return
}

// UpdateStats updates the stats of a given state and action.
func (qq *QMap) UpdateStats(stateAction iface.StateActioner, stats iface.ActionStatter) {
	qq.GetActionsForState(stateAction.State())[stateAction.Action().String()] = stats
}

// GetActionsForState returns the actions associated with a given state.
func (qq *QMap) GetActionsForState(state iface.Stater) map[string]iface.ActionStatter {
	if _, exists := qq.data[state.String()]; !exists {
		qq.data[state.String()] = make(map[string]iface.ActionStatter)
	}
	return qq.data[state.String()]
}
