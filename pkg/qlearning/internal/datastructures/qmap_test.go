package datastructures_test

import (
	"testing"

	"github.com/eltorocorp/reinforcement-learning/mocks/agent"
	"github.com/eltorocorp/reinforcement-learning/pkg/qlearning/internal/datastructures"
	"github.com/golang/mock/gomock"
	"github.com/stretchr/testify/assert"
)

func Test_GetStats_NoData(t *testing.T) {
	mc := gomock.NewController(t)
	defer mc.Finish()

	state := agent.NewMockStater(mc)
	state.EXPECT().ID().Return("A").AnyTimes()

	action := agent.NewMockActioner(mc)
	action.EXPECT().ID().Return("X").AnyTimes()

	qq := datastructures.NewQMap()
	stats, found := qq.GetStats(state, action)

	assert.Equal(t, false, found)
	assert.Equal(t, nil, stats)
}

func Test_GetStats_StateHasData(t *testing.T) {
	mc := gomock.NewController(t)
	defer mc.Finish()

	state := agent.NewMockStater(mc)
	state.EXPECT().ID().Return("A").AnyTimes()

	action := agent.NewMockActioner(mc)
	action.EXPECT().ID().Return("X").AnyTimes()

	stats := agent.NewMockActionStatter(mc)

	qq := datastructures.NewQMap()
	qq.UpdateStats(state, action, stats)
	actStats, found := qq.GetStats(state, action)

	assert.Equal(t, true, found)
	assert.Equal(t, stats, actStats)
}
