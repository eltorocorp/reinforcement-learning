package iface

// Stater is an interface wrapping the current state of the model.
type Stater interface {
	// Apply executes a specified action for the current state, resulting in a
	// new state.
	Apply(Actioner) Stater

	// PossibleActions provides a slice of Actions that are applicable to this
	// state.
	PossibleActions() []Actioner

	// String returns a string representation of the this state.
	// Implementers should take care to ensure this is a consistent hash for a
	// given state.
	String()
}

// Actioner is an interace wrapping an action that can be applied to the model's
// current state.
type Actioner interface {
	// String returns a string representation of the given action.
	// Implementers should take care to ensure this is a consistent hash for a
	// given state.
	String()
}

// Rewarder is an interace wrapping the ability to provide a reward for the
// execution of an action in a given state.
type Rewarder interface {
	// Reward calculates the reward value for a given action in a given state.
	Reward(stateAction StateActioner) float64
}

// Agenter is an interface for a model's agent; is able to recommend actions,
// learn from actions, and return the current Q-value of an action at a given
// state.
type Agenter interface {
	// Learn updates the model for a given state and action using the provided
	// Rewarder.
	Learn(StateActioner, Rewarder)

	// RecommendAction recommends an action given a state and model that the
	// agent has learned thus far.
	RecommendAction(Stater) StateActioner

	// String returns a string represetnation of the Agent.
	String() string

	// Value returns the current Q-value for a State and Action.
	Value(Stater, Actioner) float64
}

// StateActioner is the pairing of a state and an action along with a Q-value for
// the pair.
type StateActioner interface {
	State() Stater
	Action() Actioner
	Value() float64
	SetValue(float64)
}
