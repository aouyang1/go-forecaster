package forecaster

import "time"

type Results struct {
	T        []time.Time `json:"time"`
	Forecast []float64   `json:"forecast"`
	Upper    []float64   `json:"upper"`
	Lower    []float64   `json:"lower"`
}
