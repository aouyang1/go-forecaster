usage:
	@echo "make all       : Runs all tests, examples, and benchmarks"
	@echo "make test      : Runs test suite"
	@echo "make cover     : Runs coverage profile"
	@echo "make bench     : Runs benchmarks"
	@echo "make example   : Runs example"
	@echo "make cpu-pprof : Runs pprof on the cpu profile from make bench
	@echo "make mem-pprof : Runs pprof on the memory profile from make bench

all: test bench example

test:
	go test -race -coverprofile=coverage.txt -covermode=atomic -cover -run=Test ./...

cover:
	go tool cover -html=coverage.txt

bench:
	go test ./... -run=XX -bench=. -test.benchmem

example:
	go test ./... -run=Example
