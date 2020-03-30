package math_test

import (
	"strconv"
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

func Test_BayesianAverage(t *testing.T) {
	testCases := []struct {
		c   float64
		n   float64
		m   float64
		v   float64
		exp float64
	}{
		{10.0, 0.0, 100.0, 200.0, 100.0},
		{10.0, 10.0, 100.0, 200.0, 150.0},
		{10.0, 10E100, 100.0, 200.0, 200.0},
		{0.0, 0.0, 100.0, 200.0, 0},
	}
	for i, tc := range testCases {
		t.Run(strconv.Itoa(i), func(t *testing.T) {
			act := qmath.BayesianAverage(tc.c, tc.n, tc.m, tc.v)
			assert.Equal(t, tc.exp, act)
		})
	}
}

func Test_SafeDivide(t *testing.T) {
	testCases := []struct {
		dividend float64
		divisor  float64
		exp      float64
	}{
		{10.0, 2.0, 5.0},
		{0.0, 2.0, 0.0},
		{10.0, 0.0, 0.0},
	}

	for i, tc := range testCases {
		t.Run(strconv.Itoa(i), func(t *testing.T) {
			act := qmath.SafeDivide(tc.dividend, tc.divisor)
			assert.Equal(t, tc.exp, act)
		})
	}
}
