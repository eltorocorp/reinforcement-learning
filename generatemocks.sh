#!/bin/bash

mockgen -destination=./mocks/agent/agent.go -package agent github.com/eltorocorp/reinforcement-learning/pkg/qlearning/iface Stater,Actioner