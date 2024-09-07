package main

import (
	"errors"
	"fmt"
	"math"
	"time"

	"github.com/aouyang1/go-forecast/feature"
	"github.com/aouyang1/go-forecast/forecast"
	"github.com/aouyang1/go-forecast/stats"
	"github.com/aouyang1/go-forecast/timedataset"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/stat"
)

var ErrEmptyTimeDataset = errors.New("no timedataset or uninitialized")

type Forecaster struct {
	opt *Options

	seriesForecast   *forecast.Forecast
	residualForecast *forecast.Forecast
}

func New(opt *Options) (*Forecaster, error) {
	if opt == nil {
		opt = NewOptions()
	}

	f := &Forecaster{
		opt: opt,
	}

	seriesForecast, err := forecast.New(f.opt.SeriesOptions)
	if err != nil {
		return nil, fmt.Errorf("unable to initialize forecast series, %w", err)
	}
	f.seriesForecast = seriesForecast

	residualForecast, err := forecast.New(f.opt.ResidualOptions)
	if err != nil {
		return nil, fmt.Errorf("unable to initialize forecast residual, %w", err)
	}
	f.residualForecast = residualForecast
	return f, nil
}

func (f *Forecaster) Fit(trainingData *timedataset.TimeDataset) error {
	if trainingData == nil {
		return ErrEmptyTimeDataset
	}
	td, err := timedataset.NewUnivariateDataset(trainingData.T, trainingData.Y)
	if err != nil {
		return fmt.Errorf("unable to create copy of training dataset, %w", err)
	}

	// iterate to remove outliers
	numPasses := 0
	if f.opt.OutlierOptions != nil {
		numPasses = f.opt.OutlierOptions.NumPasses
	}

	var residual []float64
	for i := 0; i <= numPasses; i++ {
		if err := f.seriesForecast.Fit(td); err != nil {
			return fmt.Errorf("unable to forecast series, %w", err)
		}

		residual = f.seriesForecast.Residuals()

		outlierIdxs := stats.DetectOutliers(
			residual,
			f.opt.OutlierOptions.LowerPercentile,
			f.opt.OutlierOptions.UpperPercentile,
			f.opt.OutlierOptions.TukeyFactor,
		)
		outlierSet := make(map[int]struct{})
		for _, idx := range outlierIdxs {
			outlierSet[idx] = struct{}{}
		}

		// no more outliers detected with outlier options so break early
		if len(outlierIdxs) == 0 {
			break
		}

		for i := 0; i < len(td.T); i++ {
			if _, exists := outlierSet[i]; exists {
				td.Y[i] = math.NaN()
				continue
			}
		}
	}

	// compute rolling window standard deviation of residual foor uncertaninty bands
	stddevSeries := make([]float64, len(residual)-f.opt.ResidualWindow+1)
	numWindows := len(residual) - f.opt.ResidualWindow + 1

	for i := 0; i < numWindows; i++ {
		_, stddev := stat.MeanStdDev(residual[i:i+f.opt.ResidualWindow], nil)
		stddevSeries[i] = f.opt.ResidualZscore * stddev
	}

	start := f.opt.ResidualWindow/2 - 1
	end := len(td.T) - f.opt.ResidualWindow/2 - f.opt.ResidualWindow%2
	residualData, err := timedataset.NewUnivariateDataset(td.T[start:end], stddevSeries)
	if err != nil {
		return fmt.Errorf("unable to create univariate dataset for residual, %w", err)
	}

	if err := f.residualForecast.Fit(residualData); err != nil {
		return fmt.Errorf("unable to forecast residual, %w", err)
	}
	return nil
}

func (f *Forecaster) Predict(t []time.Time) (*Results, error) {
	seriesRes, err := f.seriesForecast.Predict(t)
	if err != nil {
		return nil, fmt.Errorf("unable to predict series forecasts, %w", err)
	}
	residualRes, err := f.residualForecast.Predict(t)
	if err != nil {
		return nil, fmt.Errorf("unable to predict residual forecasts, %w", err)
	}

	r := &Results{
		T:        t,
		Forecast: seriesRes,
	}
	upper := make([]float64, len(seriesRes))
	lower := make([]float64, len(seriesRes))

	copy(upper, seriesRes)
	copy(lower, seriesRes)

	floats.Add(upper, residualRes)
	floats.Sub(lower, residualRes)
	r.Upper = upper
	r.Lower = lower
	return r, nil
}

func (f *Forecaster) Residuals() []float64 {
	return f.seriesForecast.Residuals()
}

func (f *Forecaster) TrendComponent() []float64 {
	return f.seriesForecast.TrendComponent()
}

func (f *Forecaster) SeasonalityComponent() []float64 {
	return f.seriesForecast.SeasonalityComponent()
}

func (f *Forecaster) SeriesIntercept() float64 {
	return f.seriesForecast.Intercept()
}

func (f *Forecaster) SeriesCoefficients() (map[feature.Feature]float64, error) {
	return f.seriesForecast.Coefficients()
}

func (f *Forecaster) ResidualIntercept() float64 {
	return f.residualForecast.Intercept()
}

func (f *Forecaster) ResidualCoefficients() (map[feature.Feature]float64, error) {
	return f.residualForecast.Coefficients()
}
