package forecaster

import (
	"time"

	"github.com/aouyang1/go-forecaster/forecast"
)

// Results returns the input time points with their predicted forecast, upper, and lower values. Slices
// will be of the same length.
type Results struct {
	T        []time.Time `json:"time"`
	Forecast []float64   `json:"forecast"`
	Upper    []float64   `json:"upper"`
	Lower    []float64   `json:"lower"`

	SeriesComponents      forecast.Components `json:"series_components"`
	UncertaintyComponents forecast.Components `json:"uncertainty_components"`
}
