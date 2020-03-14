package qlearning

import (
	"math/rand"

	"github.com/eltorocorp/markov/pkg/qlearning/iface"
)

// Agent ...
type Agent struct {
}

// RecommendAction uses an Agent and State to find the highest scored Action.
//
// In the case of Q-value ties for a set of actions, a random value is selected.
func (a *Agent) RecommendAction(state iface.Stater) (result *StateAction) {
	bestActions := []iface.Actioner{}
	bestValue := 0.0

	panic("This portion is critical. Need to verify behavior of action recommendation, particularly when some actions have never been applied, but others have.")
	for _, action := range state.PossibleActions() {
		value := a.Value(state, action)
		if value > bestValue {
			bestActions = []iface.Actioner{action}
			bestValue = value
		} else if value == bestValue {
			bestActions = append(bestActions, action)
		}
	}

	if len(bestActions) == 1 {
		result = NewStateAction(state, bestActions[0], bestValue)
	} else {
		result = NewStateAction(
			state,
			bestActions[rand.Intn(len(bestActions))],
			bestValue,
		)
	}

	return
}

var _ iface.Agenter = (*Agent)(nil)
