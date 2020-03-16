package qlearning

import (
	"math/rand"

	"github.com/eltorocorp/markov/pkg/qlearning/iface"
	"github.com/eltorocorp/markov/pkg/qlearning/internal/datastructures"
	"github.com/eltorocorp/markov/pkg/qlearning/internal/math"
)

// Agent executes the qlearning process and maintains state of the learning
// process.
type Agent struct {
	qmap           *datastructures.QMap
	learningRate   float64
	discountFactor float64
}

// NewAgent returns a reference to a new Agent.
func NewAgent(learningRate, discountFactor float64) *Agent {
	return &Agent{
		qmap:           datastructures.NewQMap(),
		discountFactor: discountFactor,
		learningRate:   learningRate,
	}
}

// Learn executes the supplied stateAction, completing a transition from
// the current state (A) to a new state (B). The q-value for the action applied
// to state A is updated using the supplied rewarder and a standard Bellman
// equation.
// See https://en.wikipedia.org/wiki/Q-learning#Algorithm
func (a *Agent) Learn(stateAction iface.StateActioner, rewarder iface.Rewarder) {
	newState := stateAction.Transition()
	newValue := math.Bellman(
		a.qmap.GetValue(stateAction.State(), stateAction.Action()),
		a.learningRate,
		rewarder.Reward(stateAction),
		a.discountFactor,
		a.getBestValue(newState),
	)
	a.qmap.SetValue(stateAction, newValue)
}

// RecommendAction recommends an action for a given state based on behavior of
// the system that the agent has learned thus far.
// If the q-value for two or more actions are the same, the action is chosen at
// random.
func (a *Agent) RecommendAction(state iface.Stater) (result iface.StateActioner) {
	bestActions := []iface.Actioner{}
	bestValue := 0.0

	for _, action := range state.PossibleActions() {
		value := a.qmap.GetValue(state, action)
		if value > bestValue {
			bestActions = []iface.Actioner{action}
			bestValue = value
		} else if value == bestValue {
			bestActions = append(bestActions, action)
		}
	}

	if len(bestActions) == 1 {
		result = NewStateAction(state, bestActions[0])
	} else {
		result = NewStateAction(
			state,
			bestActions[rand.Intn(len(bestActions))],
		)
	}

	return
}

// getBestValue returns the best possible q-value for a state.
func (a *Agent) getBestValue(state iface.Stater) (bestValue float64) {
	for _, v := range a.qmap.GetActionsForState(state) {
		if v > bestValue {
			bestValue = v
		}
	}
	return
}

var _ iface.Agenter = (*Agent)(nil)
