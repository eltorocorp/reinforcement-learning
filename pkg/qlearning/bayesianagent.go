package qlearning

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/eltorocorp/reinforcement-learning/pkg/qlearning/iface"
	"github.com/eltorocorp/reinforcement-learning/pkg/qlearning/internal/datastructures"
	qlmath "github.com/eltorocorp/reinforcement-learning/pkg/qlearning/internal/math"
)

// BayesianAgent provides facilities for 1) maintaining the learning state of an
// environment, 2) making recommendations for actions based on the previous,
// current, and predicted states of the system, and 3) executing actions based
// that have been recommended by the agent.
//
// The BayesianAgent is so named because of the way it handles initial
// conditions of the q-values associated with each of a state's actions.
// When the agent is asked to recommend an action for some state, the agent
// does so by choosing the action that has previously recorded a greater
// cumulative reward than other possible actions.
//
// This poses a dilema for initial conditions when no reward has been previously
// recorded for one or more of the potential actions. To overcome this, the
// BayesianAgent applies a Bayesian Average function to each potential action.
// In essense, when an action has been called few (or zero) times, it is assumed
// that the reward for calling that action might be similar to that of calling
// any other action. Thus the agent weights its potential reward closer to the
// mean of all other actions. However, as an action is called more times, the
// agent begins to evaluate the action on its observed cumulative reward moreso
// than the mean of all other actions.
type BayesianAgent struct {
	// TieBreakSeeder is a function that returns a number used to seed the
	// random number generator used to break ties between actions of equal value.
	// This property defaults to a function that returns time.Now().UnixNano(),
	// but is exposed so this behavior can be overridden for testing.
	TieBreakSeeder   func() int64
	qmap             *datastructures.QMap
	learningRate     float64
	discountFactor   float64
	primingThreshold int
}

// NewBayesianAgent returns a reference to a new BayesianAgent.
//
// primingthreshold:
//  The number of observations required of any action before the action's
//  raw q-value is trusted more than average q-value for all of a state's
//  actions.
//
// learningRate:
//  Typically a number between 0 and 1 (though it can exceed 1)
//  From wikipedia: Determins to what extent newly acquired information
//  overrides old information.
//  see: https://en.wikipedia.org/wiki/Q-learning#Learning_Rate
//
// discountFactor:
//  From wikipedia: The discount factor determines the importance of future
//  rewards.
//  see: https://en.wikipedia.org/wiki/Q-learning#Discount_factor
func NewBayesianAgent(primingThreshold int, learningRate, discountFactor float64) *BayesianAgent {
	return &BayesianAgent{
		TieBreakSeeder:   func() int64 { return time.Now().UnixNano() },
		qmap:             datastructures.NewQMap(),
		discountFactor:   discountFactor,
		learningRate:     learningRate,
		primingThreshold: primingThreshold,
	}
}

// Learn updates the reinforcement model according to a transition that has
// occured from a previous state through some action to a current state. The
// reward value represents the positive, negative, or neutral impact that the
// transition has had on the environment. If no action has been previously
// taken, or there is no previous state (the system is being bootstrapped),
// nil may be supplied for previousState or actionTaken. In either of these
// cases, Learn becomes a no-op. Learn will panic if currentState is nil.
// See https://en.wikipedia.org/wiki/Q-learning#Algorithm
func (a *BayesianAgent) Learn(previousState iface.Stater, actionTaken iface.Actioner, currentState iface.Stater, reward float64) {
	if previousState == nil || actionTaken == nil {
		return
	}

	if currentState == nil {
		panic("currentState must not be nil")
	}

	var stats iface.ActionStatter
	var found bool
	stats, found = a.qmap.GetStats(previousState, actionTaken)
	if !found {
		stats = new(ActionStats)
	}

	newValue := qlmath.Bellman(
		stats.QValueWeighted(),
		a.learningRate,
		reward,
		a.discountFactor,
		a.getBestValue(currentState),
	)
	stats.SetCalls(stats.Calls() + 1)
	stats.SetQValueRaw(newValue)
	a.qmap.UpdateStats(previousState, actionTaken, stats)
	a.applyActionWeights(previousState)
}

// Transition applies an action to a given state.
func (a *BayesianAgent) Transition(currentState iface.Stater, action iface.Actioner) error {
	if !currentState.ActionIsCompatible(action) {
		return fmt.Errorf("action %v is not compatible with state %v", currentState.ID(), action.ID())
	}
	return currentState.Apply(action)
}

// RecommendAction recommends an action for a given state based on behavior of
// the system that the agent has learned thus far.
// If the q-value for two or more actions are the same, the action is chosen at
// random. See BayesianAgent struct docs for more information.
func (a *BayesianAgent) RecommendAction(state iface.Stater) (iface.Actioner, error) {
	type actionValue struct {
		action string
		value  float64
	}

	bestActions := []actionValue{}
	bestValue := -1 * math.MaxFloat64

	a.applyActionWeights(state)
	for action, stats := range a.qmap.GetActionsForState(state) {
		av := actionValue{action, stats.QValueWeighted()}
		if av.value > bestValue {
			bestActions = []actionValue{av}
			bestValue = av.value
		} else if av.value == bestValue {
			bestActions = append(bestActions, av)
		}
	}

	if len(bestActions) == 0 {
		return nil, fmt.Errorf("state '%v' reports no possible actions", state.ID())
	}

	rand.Seed(a.TieBreakSeeder())
	tieBreaker := rand.Intn(len(bestActions))
	bestAction, err := state.GetAction(bestActions[tieBreaker].action)
	if err != nil {
		return nil, err
	}
	return bestAction, nil
}

func (a *BayesianAgent) applyActionWeights(state iface.Stater) {
	rawValueSum := 0.0
	existingActionCount := 0.0
	for _, action := range state.PossibleActions() {
		stats, found := a.qmap.GetStats(state, action)
		if !found {
			a.qmap.UpdateStats(state, action, new(ActionStats))
		} else {
			rawValueSum += nanToZero(stats.QValueRaw())
			existingActionCount++
		}
	}

	mean := qlmath.SafeDivide(rawValueSum, existingActionCount)
	for _, stats := range a.qmap.GetActionsForState(state) {
		weighedMean := qlmath.BayesianAverage(
			float64(a.primingThreshold),
			float64(stats.Calls()),
			nanToZero(mean),
			nanToZero(stats.QValueRaw()),
		)
		stats.SetQValueWeighted(weighedMean)
	}
}

func nanToZero(f float64) float64 {
	if math.IsNaN(f) {
		return 0
	}
	return f
}

// getBestValue returns the best possible q-value for a state.
func (a *BayesianAgent) getBestValue(state iface.Stater) (bestQValue float64) {
	for _, stat := range a.qmap.GetActionsForState(state) {
		q := nanToZero(stat.QValueWeighted())
		if q > bestQValue {
			bestQValue = q
		}
	}
	return
}

// MarshalJSON serializes agent's current model.
func (a *BayesianAgent) MarshalJSON() ([]byte, error) {
	return json.MarshalIndent(a.qmap.Data, "", "  ")
}

// UnmarshalJSON hydrates the agent using an existing model.
func (a *BayesianAgent) UnmarshalJSON(model []byte) error {
	return json.Unmarshal(model, &a.qmap.Data)
}

var _ iface.Agenter = (*BayesianAgent)(nil)
