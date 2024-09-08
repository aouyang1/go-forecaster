package main

import "github.com/aouyang1/go-forecast/forecast"

type OutlierOptions struct {
	NumPasses       int     `json:"num_passes"`
	UpperPercentile float64 `json:"upper_percentile"`
	LowerPercentile float64 `json:"lower_percentile"`
	TukeyFactor     float64 `json:"tukey_factor"`
}

func NewOutlierOptions() *OutlierOptions {
	return &OutlierOptions{
		NumPasses:       3,
		UpperPercentile: 0.9,
		LowerPercentile: 0.1,
		TukeyFactor:     1.0,
	}
}

type Options struct {
	SeriesOptions   *forecast.Options `json:"-"`
	ResidualOptions *forecast.Options `json:"-"`

	OutlierOptions *OutlierOptions `json:"outlier_options"`
	ResidualWindow int             `json:"residual_window"`
	ResidualZscore float64         `json:"residual_zscore"`
}

func NewOptions() *Options {
	return &Options{
		SeriesOptions:   forecast.NewDefaultOptions(),
		ResidualOptions: forecast.NewDefaultOptions(),
		ResidualWindow:  100,
		ResidualZscore:  4.0,
	}
}
