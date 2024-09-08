package main

import "github.com/aouyang1/go-forecast/forecast"

type Model struct {
	Options  *Options       `json:"options"`
	Series   forecast.Model `json:"series_model"`
	Residual forecast.Model `json:"residual_model"`
}
