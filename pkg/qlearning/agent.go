package qlearning

import (
	"math/rand"

	"github.com/eltorocorp/markov/pkg/qlearning/iface"
)

// Agent ...
type Agent struct {
	qmap           map[string]map[string]float64
	learningRate   float64
	discountFactor float64
}

// NewAgent returns a reference to a new Agent.
func NewAgent(learningRate, discountFactor float64) *Agent {
	return &Agent{
		qmap:           make(map[string]map[string]float64),
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
	newValue := Bellman(
		a.getActionValueForState(stateAction.State(), stateAction.Action()),
		a.learningRate,
		rewarder.Reward(stateAction),
		a.discountFactor,
		a.getBestOutcomeEstimateForState(newState),
	)
	a.updateValueForStateAction(stateAction, newValue)
}

// RecommendAction recommends an action for a given state based on behavior of
// the system that the agent has learned thus far.
// If the q-value for two or more actions are the same, the action is chosen at
// random.
func (a *Agent) RecommendAction(state iface.Stater) (result iface.StateActioner) {
	bestActions := []iface.Actioner{}
	bestValue := 0.0

	for _, action := range state.PossibleActions() {
		value := a.getActionValueForState(state, action)
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

// Bellman applies a Bellman equation to recommend a new q-value for a state
// based on the supplied paramters.
// See https://en.wikipedia.org/wiki/Bellman_equation
func Bellman(oldValue, learningRate, reward, discountFactor, optimalFutureValue float64) float64 {
	return oldValue + learningRate*(reward+discountFactor*optimalFutureValue-oldValue)
}

func (a *Agent) getActionValueMapForState(state iface.Stater) map[string]float64 {
	if _, exists := a.qmap[state.String()]; !exists {
		a.qmap[state.String()] = make(map[string]float64)
	}
	return a.qmap[state.String()]
}

func (a *Agent) getActionValueForState(state iface.Stater, action iface.Actioner) float64 {
	return a.getActionValueMapForState(state)[action.String()]
}

func (a *Agent) getBestOutcomeEstimateForState(state iface.Stater) (bestNewStateOutcome float64) {
	for _, v := range a.getActionValueMapForState(state) {
		if v > bestNewStateOutcome {
			bestNewStateOutcome = v
		}
	}
	return
}

func (a *Agent) updateValueForStateAction(stateAction iface.StateActioner, value float64) {
	a.getActionValueMapForState(stateAction.State())[stateAction.Action().String()] = value
}

var _ iface.Agenter = (*Agent)(nil)
