local: build buildmocks test

build:
	@echo Updating build tooling...
	@go mod tidy
	@go install -i github.com/eltorocorp/drygopher/drygopher
	@go build ./...
.PHONY: build

buildmocks:
	@echo Purging old mocks...
	@rm -drf mocks/*

	@echo Building mocks...

	@mkdir mocks/agent && \
		mockgen -destination=./mocks/agent/agent.go -package agent github.com/eltorocorp/reinforcement-learning/pkg/qlearning/iface Stater,Actioner
.PHONY: buildmocks

test:
	@echo Testing...
	@drygopher -d -s 80 -e "'mocks','iface','types'"
.PHONY: test