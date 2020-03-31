package qlearning_test

import (
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

	reward := 1.0
	ba.Learn(previousState, action1, currentState, reward)
	ba.Learn(previousState, action2, currentState, reward)

	result, err := ba.MarshalJSON()
	if err != nil {
		t.Fatal(err)
	}

	expected := "{\n  \"A\": {\n    \"X\": {\n      \"CallCount\": 1,\n      \"QRaw\": 1,\n      \"QWeighted\": 0.6969696969696969\n    },\n    \"Y\": {\n      \"CallCount\": 1,\n      \"QRaw\": 1,\n      \"QWeighted\": 0.6969696969696969\n    },\n    \"Z\": {\n      \"CallCount\": 0,\n      \"QRaw\": 0,\n      \"QWeighted\": 0.6666666666666666\n    }\n  },\n  \"B\": {}\n}"
	assert.Equal(t, expected, string(result))
}
