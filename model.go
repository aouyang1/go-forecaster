package forecaster

import "github.com/aouyang1/go-forecaster/forecast"

// Model is a serializeable representation of the forecaster's configurations and models for the
// forecast and uncertainty.
type Model struct {
	Options     *Options       `json:"options"`
	Series      forecast.Model `json:"series_model"`
	Uncertainty forecast.Model `json:"uncertainty_model"`
}
