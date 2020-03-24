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

func TestBayesianAgentRecommendAction(t *testing.T) {
	const testStateID = "testStateID"

	testCases := []struct {
		name            string
		possibleActions []string
		expAction       string
		expError        error
	}{
		{
			name:            "Error if no actions",
			possibleActions: []string{},
			expAction:       "",
			expError:        fmt.Errorf("state '%v' reports no possible actions", testStateID),
		},
		{
			name:            "Action returned when bootstrapping",
			possibleActions: []string{"A"},
			expAction:       "A",
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

			if testCase.expError == nil {
				// Hacky, but since the current test cases only ever setup for
				// zero or one possible actions, if we are expecting an action
				// to be returned, we assume it is action[0].
				state.EXPECT().GetAction(testCase.expAction).Return(actions[0], nil).Times(1)
			}

			a := qlearning.NewBayesianAgent(1, .5, .5)
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
