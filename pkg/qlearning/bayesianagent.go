package qlearning

import (
	"math/rand"
	"time"

	"github.com/eltorocorp/markov/pkg/qlearning/iface"
	"github.com/eltorocorp/markov/pkg/qlearning/internal/datastructures"
	"github.com/eltorocorp/markov/pkg/qlearning/internal/math"
)

// BayesianAgent executes the qlearning process and maintains state of the learning
// process.
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
// primingthreshold: The number of observations required of any action before
//					 the action's raw q-value is trusted more than average
//					 q-value for all of a state's actions.
// learningRate:	 Typically a number between 0 and 1 (though it can exceed 1)
//					 From wikipedia: Determins to what extent newly acquired
//					 information overrides old information.
//					 see: https://en.wikipedia.org/wiki/Q-learning#Learning_Rate
// discountFactor:   From wikipedia: The discount factor determines the
//					 importance of future rewards.
//					 see: https://en.wikipedia.org/wiki/Q-learning#Discount_factor
func NewBayesianAgent(primingThreshold int, learningRate, discountFactor float64) *BayesianAgent {
	return &BayesianAgent{
		TieBreakSeeder:   func() int64 { return time.Now().UnixNano() },
		qmap:             datastructures.NewQMap(),
		discountFactor:   discountFactor,
		learningRate:     learningRate,
		primingThreshold: primingThreshold,
	}
}

// Learn executes the supplied stateAction, completing a transition from
// the current state (A) to a new state (B). The q-value for the action applied
// to state A is updated using the supplied rewarder and a standard Bellman
// equation.
// See https://en.wikipedia.org/wiki/Q-learning#Algorithm
func (a *BayesianAgent) Learn(stateAction iface.StateActioner, rewarder iface.Rewarder) error {
	newState, err := stateAction.Transition()
	if err != nil {
		return err
	}

	var stats iface.ActionStatter
	var found bool
	stats, found = a.qmap.GetStats(stateAction.State(), stateAction.Action())
	if !found {
		stats = new(ActionStats)
	}

	newValue := math.Bellman(
		stats.QValueWeighted(),
		a.learningRate,
		rewarder.Reward(stateAction),
		a.discountFactor,
		a.getBestValue(newState).QValueWeighted(),
	)
	stats.SetCalls(stats.Calls() + 1)
	stats.SetQValueRaw(newValue)
	a.qmap.UpdateStats(stateAction, stats)
	a.applyActionWeights(stateAction.State())

	return nil
}

// RecommendAction recommends an action for a given state based on behavior of
// the system that the agent has learned thus far.
// If the q-value for two or more actions are the same, the action is chosen at
// random.
func (a *BayesianAgent) RecommendAction(state iface.Stater) (iface.StateActioner, error) {
	type actionValue struct {
		action string
		value  float64
	}

	bestActions := []actionValue{}
	bestValue := 0.0

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

	rand.Seed(a.TieBreakSeeder())
	tieBreaker := rand.Intn(len(bestActions))
	bestAction, err := state.GetAction(bestActions[tieBreaker].action)
	if err != nil {
		return nil, err
	}
	return NewStateAction(state, bestAction), nil
}

func (a *BayesianAgent) applyActionWeights(state iface.Stater) {
	rawValueSum := 0.0
	existingActionCount := 0.0
	for _, action := range state.PossibleActions() {
		stats, found := a.qmap.GetStats(state, action)
		if !found {
			a.qmap.UpdateStats(NewStateAction(state, action), new(ActionStats))
		} else {
			rawValueSum += stats.QValueRaw()
			existingActionCount++
		}
	}

	mean := math.SafeDivide(rawValueSum, existingActionCount)
	for _, stats := range a.qmap.GetActionsForState(state) {
		weighedMean := math.BayesianAverage(
			float64(a.primingThreshold),
			float64(stats.Calls()),
			mean,
			stats.QValueRaw(),
		)
		stats.SetQValueWeighted(weighedMean)
	}
}

// getBestValue returns the best possible q-value for a state.
func (a *BayesianAgent) getBestValue(state iface.Stater) (bestStat iface.ActionStatter) {
	for _, stat := range a.qmap.GetActionsForState(state) {
		if stat.QValueWeighted() > bestStat.QValueWeighted() {
			bestStat = stat
		}
	}
	return
}

var _ iface.Agenter = (*BayesianAgent)(nil)