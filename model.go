package forecaster

import "github.com/aouyang1/go-forecaster/forecast"

type Model struct {
	Options  *Options       `json:"options"`
	Series   forecast.Model `json:"series_model"`
	Residual forecast.Model `json:"residual_model"`
}
