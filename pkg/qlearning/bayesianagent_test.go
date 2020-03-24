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
	// when bootstrapping, an action should be returned
	const testStateID = "testStateID"

	testCases := []struct {
		name            string
		possibleActions []iface.Actioner
		expAction       iface.Actioner
		expError        error
	}{
		{
			name:            "Error if no actions",
			possibleActions: []iface.Actioner{},
			expAction:       nil,
			expError:        fmt.Errorf("state '%v' reports no possible actions", testStateID),
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			mc := gomock.NewController(t)
			defer mc.Finish()

			state := agent.NewMockStater(mc)
			state.EXPECT().ID().AnyTimes().Return(testStateID)
			state.EXPECT().PossibleActions().AnyTimes().Return(testCase.possibleActions)

			a := qlearning.NewBayesianAgent(1, 1, 1)
			actAction, actError := a.RecommendAction(state)

			assert.Equal(t, testCase.expAction, actAction)
			assert.Equal(t, testCase.expError, actError)
		})
	}
}
