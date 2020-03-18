package iface

// Stater is an interface wrapping the current state of the model.
type Stater interface {
	// PossibleActions provides a slice of Actions that are applicable to this
	// state.
	PossibleActions() []Actioner

	// GetAction returns an action of a specified name, or an error if no action
	// exists of that name for this state.
	GetAction(string) (Actioner, error)

	// ID returns a string representation of the this state.
	// Implementers should take care to ensure this is a consistent hash for a
	// given state.
	ID() string

	// Apply executes an action against the State, resulting in a new state.
	Apply(Actioner) (Stater, error)
}

// Actioner is an interace wrapping an action that can be applied to the model's
// current state.
type Actioner interface {
	// ID returns a string representation of the given action.
	// Implementers should take care to ensure this is a consistent hash for a
	// given state.
	ID() string
}

// An Agenter is anything that is capable of recommending actions, applying
// actions to a given state, and learning based on the transition from one state
// to another.
type Agenter interface {
	// RecommendAction recommends an action given a state and the model that the
	// agent has learned thus far.
	RecommendAction(Stater) (Actioner, error)

	// Transition applies an action to a given state.
	// Implementors should take care to ensure that Transition returns an error
	// if the supplied Action is not applicable to the specified state.
	Transition(Stater, Actioner) error

	// Learn updates the model for a given state and action using the provided
	// Rewarder.
	Learn(previousState Stater, actionTaken Actioner, currentState Stater, reward float64)
}

// ActionStatter is something that can represent the stats associated with an
// action.
type ActionStatter interface {
	Calls() int
	SetCalls(int)
	QValueRaw() float64
	SetQValueRaw(float64)
	QValueWeighted() float64
	SetQValueWeighted(float64)
}
