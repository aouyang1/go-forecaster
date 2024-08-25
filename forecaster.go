package main

import (
	"fmt"

	"gonum.org/v1/gonum/stat"
)

type ForecasterOptions struct {
	SeriesOptions   *Options
	ResidualOptions *Options
}

type Forecaster struct {
	opt *ForecasterOptions

	seriesForecast   *Forecast
	residualForecast *Forecast
}

func NewForecaster() (*Forecaster, error) {
	return &Forecaster{}, nil
}

func (f *Forecaster) Fit(trainingData *TimeDataset) error {
	seriesForecast, err := NewForecast(f.opt.SeriesOptions)
	if err != nil {
		return fmt.Errorf("unable to initialize forecast series, %w", err)
	}
	f.seriesForecast = seriesForecast
	if err := f.seriesForecast.Fit(trainingData); err != nil {
		return fmt.Errorf("unable to forecast series, %w", err)
	}

	residual := f.seriesForecast.Residuals()

	window := 100
	zscore := 3.0

	stddevSeries := make([]float64, len(residual)-window)
	for i := 0; i < len(residual)-window; i++ {
		_, stddev := stat.MeanStdDev(residual[i:i+window], nil)
		stddevSeries[i] = zscore * stddev
	}
	residualData, err := NewUnivariateDataset(trainingData.t[window-1:], stddevSeries)
	if err != nil {
		return fmt.Errorf("unable to create univariate dataset for residual, %w", err)
	}

	residualForecast, err := NewForecast(f.opt.ResidualOptions)
	if err != nil {
		return fmt.Errorf("unable to initialize forecast residual, %w", err)
	}
	f.residualForecast = residualForecast
	if err := f.residualForecast.Fit(residualData); err != nil {
		return fmt.Errorf("unable to forecast residual, %w", err)
	}
	return nil
}
