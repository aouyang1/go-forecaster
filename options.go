package main

import "github.com/aouyang1/go-forecast/forecast"

type OutlierOptions struct {
	NumPasses       int
	UpperPercentile float64
	LowerPercentile float64
	TukeyFactor     float64
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
	SeriesOptions   *forecast.Options
	ResidualOptions *forecast.Options

	OutlierOptions *OutlierOptions
	ResidualWindow int
	ResidualZscore float64
}

func NewOptions() *Options {
	return &Options{
		SeriesOptions:   forecast.NewDefaultOptions(),
		ResidualOptions: forecast.NewDefaultOptions(),
		ResidualWindow:  100,
		ResidualZscore:  4.0,
	}
}
