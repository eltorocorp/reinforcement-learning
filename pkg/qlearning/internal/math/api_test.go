package math_test

import (
	"testing"

	qmath "github.com/eltorocorp/reinforcement-learning/pkg/qlearning/internal/math"
	"github.com/stretchr/testify/assert"
)

func Test_Bellman(t *testing.T) {
	oldValue := .1
	learningRate := .2
	reward := .3
	discountFactor := .4
	optimalFutureValue := .5
	actResult := qmath.Bellman(oldValue, learningRate, reward, discountFactor, optimalFutureValue)
	expResult := 0.18000000000000002
	assert.Equal(t, expResult, actResult)
}
