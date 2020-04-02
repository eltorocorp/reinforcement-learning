package qlearning_test

import (
	"encoding/json"
	"fmt"
	"testing"

	"github.com/eltorocorp/reinforcement-learning/mocks/agent"
	"github.com/eltorocorp/reinforcement-learning/pkg/qlearning"
	"github.com/eltorocorp/reinforcement-learning/pkg/qlearning/iface"
	"github.com/golang/mock/gomock"
	"github.com/stretchr/testify/assert"
)

func Test_BayesianAgentRecommendAction(t *testing.T) {
	const testStateID = "testStateID"

	testCases := []struct {
		name            string
		possibleActions []string
		tieBreakIndex   int
		expAction       string
		expError        error
	}{
		{
			name:            "Error if no actions",
			possibleActions: []string{},
			tieBreakIndex:   0,
			expAction:       "",
			expError:        fmt.Errorf("state '%v' reports no possible actions", testStateID),
		},
		{
			name:            "Action returned when bootstrapping",
			possibleActions: []string{"A"},
			tieBreakIndex:   0,
			expAction:       "A",
			expError:        nil,
		},
		{
			name:            "Action chosen when tied",
			possibleActions: []string{"A", "B"},
			tieBreakIndex:   1,
			expAction:       "B",
			expError:        nil,
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			mc := gomock.NewController(t)
			defer mc.Finish()

			state := agent.NewMockStater(mc)
			state.EXPECT().ID().AnyTimes().Return(testStateID)

			actions := make([]iface.Actioner, len(testCase.possibleActions))
			for i, id := range testCase.possibleActions {
				newAction := agent.NewMockActioner(mc)
				newAction.EXPECT().ID().AnyTimes().Return(id)
				actions[i] = newAction
			}
			state.EXPECT().PossibleActions().AnyTimes().Return(actions)

			expectedAction := agent.NewMockActioner(mc)
			expectedAction.EXPECT().ID().Return(testCase.expAction).AnyTimes()

			if testCase.expError == nil {
				state.EXPECT().GetAction(testCase.expAction).Return(expectedAction, nil).Times(1)
			}

			a := qlearning.NewBayesianAgent(1, .5, .5)
			a.TieBreaker = func(int) int { return testCase.tieBreakIndex }
			actAction, actError := a.RecommendAction(state)

			if testCase.expError == nil {
				if assert.NotNil(t, actAction) {
					assert.Equal(t, testCase.expAction, actAction.ID())
				}
			}
			assert.Equal(t, testCase.expError, actError)
		})
	}
}

func Test_BayesianAgentLearn(t *testing.T) {
	mc := gomock.NewController(t)
	defer mc.Finish()

	action1 := agent.NewMockActioner(mc)
	action1.EXPECT().ID().Return("X").AnyTimes()

	action2 := agent.NewMockActioner(mc)
	action2.EXPECT().ID().Return("Y").AnyTimes()

	action3 := agent.NewMockActioner(mc)
	action3.EXPECT().ID().Return("Z").AnyTimes()

	ba := qlearning.NewBayesianAgent(10, 1, 0)
	previousState := agent.NewMockStater(mc)
	previousState.EXPECT().ID().Return("A").AnyTimes()
	previousState.EXPECT().PossibleActions().Return(
		[]iface.Actioner{
			action1,
			action2,
			action3,
		}).AnyTimes()

	currentState := agent.NewMockStater(mc)
	currentState.EXPECT().ID().Return("B").AnyTimes()
	currentState.EXPECT().PossibleActions().Return(
		[]iface.Actioner{
			action1,
			action2,
			action3,
		}).AnyTimes()

	reward := 1.0
	ba.Learn(previousState, action1, currentState, reward)
	ba.Learn(previousState, action2, currentState, reward)

	result := ba.GetAgentContext()
	actualJSON, err := json.Marshal(result)

	expected := qlearning.AgentContext{
		LearningRate:     1,
		DiscountFactor:   0,
		PrimingThreshold: 10,
		QValues: map[string]map[string]iface.ActionStatter{
			"A": map[string]iface.ActionStatter{
				"X": &qlearning.ActionStats{CallCount: 1, QRaw: 1, QWeighted: 0.6969696969696969},
				"Y": &qlearning.ActionStats{CallCount: 1, QRaw: 1, QWeighted: 0.6969696969696969},
				"Z": &qlearning.ActionStats{CallCount: 0, QRaw: 0, QWeighted: 0.66666666666666666},
			},
			"B": map[string]iface.ActionStatter{
				"X": &qlearning.ActionStats{CallCount: 0, QRaw: 0, QWeighted: 0},
				"Y": &qlearning.ActionStats{CallCount: 0, QRaw: 0, QWeighted: 0},
				"Z": &qlearning.ActionStats{CallCount: 0, QRaw: 0, QWeighted: 0},
			},
		},
	}

	expectedJSON, err := json.Marshal(expected)
	if err != nil {
		t.Fatal(err)
	}

	assert.Equal(t, string(expectedJSON), string(actualJSON))

}

func Test_Transition(t *testing.T) {

	testCases := []struct {
		name               string
		actionIsCompatible bool
		expError           error
	}{
		{
			name:               "happy path",
			actionIsCompatible: true,
			expError:           nil,
		},
		{
			name:               "incompatible action returns error",
			actionIsCompatible: false,
			expError:           fmt.Errorf("action X is not compatible with state A"),
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			mc := gomock.NewController(t)
			defer mc.Finish()

			ba := qlearning.NewBayesianAgent(0, 0, 0)
			action := agent.NewMockActioner(mc)
			currentState := agent.NewMockStater(mc)
			currentState.EXPECT().
				ActionIsCompatible(action).
				Return(testCase.actionIsCompatible).
				Times(1)

			if testCase.expError == nil {
				currentState.EXPECT().Apply(gomock.Any()).Return(nil).Times(1)
				err := ba.Transition(currentState, action)
				assert.NoError(t, err)
			} else {
				currentState.EXPECT().ID().Return("A").Times(1)
				action.EXPECT().ID().Return("X").Times(1)
				err := ba.Transition(currentState, action)
				assert.Error(t, err, testCase.expError)
			}
		})
	}

}
