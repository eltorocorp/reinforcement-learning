package datastructures

import "github.com/eltorocorp/reinforcement-learning/pkg/qlearning/iface"

// QMap is a mapping of states to actions (and each action's q-value).
type QMap struct {
	Data map[string]map[string]iface.ActionStatter
}

// NewQMap returns a new QMap
func NewQMap() *QMap {
	return &QMap{
		Data: map[string]map[string]iface.ActionStatter{},
	}
}

// GetStats returns the stats for a given state and action.
// If a the specified action has not been recorded for the given state, the
// method will return nil, false.
func (qq *QMap) GetStats(state iface.Stater, action iface.Actioner) (stats iface.ActionStatter, found bool) {
	actions := qq.GetActionsForState(state)
	stats, found = actions[action.ID()]
	return
}

// UpdateStats updates the stats of a given state and action.
func (qq *QMap) UpdateStats(state iface.Stater, action iface.Actioner, stats iface.ActionStatter) {
	qq.GetActionsForState(state)[action.ID()] = stats
}

// GetActionsForState returns the actions associated with a given state.
func (qq *QMap) GetActionsForState(state iface.Stater) map[string]iface.ActionStatter {
	if _, exists := qq.Data[state.ID()]; !exists {
		qq.Data[state.ID()] = make(map[string]iface.ActionStatter)
	}
	return qq.Data[state.ID()]
}
