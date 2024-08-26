package main

import "time"

type Results struct {
	T        []time.Time
	Forecast []float64
	Upper    []float64
	Lower    []float64
}
