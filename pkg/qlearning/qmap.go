package qlearning

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
	return qq.getActionsForState(state)[action.String()]
}

// GetBestValue returns the best possible q-value for a state.
func (qq *QMap) GetBestValue(state iface.Stater) (bestNewStateOutcome float64) {
	for _, v := range qq.getActionsForState(state) {
		if v > bestNewStateOutcome {
			bestNewStateOutcome = v
		}
	}
	return
}

// SetValue sets the value of a given state and action.
func (qq *QMap) SetValue(stateAction iface.StateActioner, value float64) {
	qq.getActionsForState(stateAction.State())[stateAction.Action().String()] = value
}

func (qq *QMap) getActionsForState(state iface.Stater) map[string]float64 {
	if _, exists := (*qq)[state.String()]; !exists {
		(*qq)[state.String()] = make(map[string]float64)
	}
	return (*qq)[state.String()]
}
