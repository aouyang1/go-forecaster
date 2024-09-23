package forecaster

import "time"

// Results returns the input time points with their predicted forecast, upper, and lower values. Slices
// will be of the same length.
type Results struct {
	T        []time.Time `json:"time"`
	Forecast []float64   `json:"forecast"`
	Upper    []float64   `json:"upper"`
	Lower    []float64   `json:"lower"`
}
