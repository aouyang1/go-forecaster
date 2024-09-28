package forecaster

import (
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"time"

	"github.com/aouyang1/go-forecaster/forecast"
	"github.com/aouyang1/go-forecaster/stats"
	"github.com/aouyang1/go-forecaster/timedataset"
	"github.com/go-echarts/go-echarts/v2/components"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/stat"
)

var (
	ErrInsufficientResidual = errors.New("insufficient samples from residual after outlier removal")
	ErrEmptyTimeDataset     = errors.New("no timedataset or uninitialized")
	ErrNoOptionsInModel     = errors.New("no options set in model")
)

const (
	MinResidualWindow       = 2
	MinResidualSize         = 2
	MinResidualWindowFactor = 4
)

// Forecaster fits a forecast model and can be used to generate forecasts
type Forecaster struct {
	opt *Options

	seriesForecast   *forecast.Forecast
	residualForecast *forecast.Forecast

	fitTrainingData *timedataset.TimeDataset
	fitResults      *Results
}

// New creates a new instance of a Forecaster using thhe provided options. If no options are provided
// a default is used.
func New(opt *Options) (*Forecaster, error) {
	if opt == nil {
		opt = NewDefaultOptions()
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

// NewFromModel creates a new instance of Forecaster from a pre-existing model. This should be generated from
// from a previous forecaster call to Model().
func NewFromModel(model Model) (*Forecaster, error) {
	if model.Options == nil {
		return nil, ErrNoOptionsInModel
	}
	opt := model.Options
	opt.SeriesOptions = model.Series.Options
	opt.ResidualOptions = model.Residual.Options

	seriesForecast, err := forecast.NewFromModel(model.Series)
	if err != nil {
		return nil, fmt.Errorf("unable to load from series model, %w", err)
	}
	residualForecast, err := forecast.NewFromModel(model.Residual)
	if err != nil {
		return nil, fmt.Errorf("unable to load from residual model, %w", err)
	}
	f := &Forecaster{
		opt:              opt,
		seriesForecast:   seriesForecast,
		residualForecast: residualForecast,
	}
	return f, nil
}

// Fit uses the input time dataset and fits the forecast model
func (f *Forecaster) Fit(t []time.Time, y []float64) error {
	td, err := timedataset.NewUnivariateDataset(t, y)
	if err != nil {
		return fmt.Errorf("unable to create training dataset, %w", err)
	}
	f.fitTrainingData = td.Copy()

	residual, err := f.fitSeriesWithOutliers(td.T, td.Y)
	if err != nil {
		return err
	}

	if err := f.fitResidual(td.T, residual); err != nil {
		return err
	}

	f.fitResults, err = f.Predict(t)
	if err != nil {
		return fmt.Errorf("unable to get predicted values from training set, %w", err)
	}

	return nil
}

func (f *Forecaster) fitSeriesWithOutliers(t []time.Time, y []float64) ([]float64, error) {
	// iterate to remove outliers
	numPasses := 0
	if f.opt.OutlierOptions != nil {
		numPasses = f.opt.OutlierOptions.NumPasses
	}

	var residual []float64
	for i := 0; i <= numPasses; i++ {
		if err := f.seriesForecast.Fit(t, y); err != nil {
			return nil, fmt.Errorf("unable to forecast series, %w", err)
		}

		residual = f.seriesForecast.Residuals()

		// break out if no outlier options provided
		if f.opt.OutlierOptions == nil {
			break
		}

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

		for i := 0; i < len(t); i++ {
			if _, exists := outlierSet[i]; exists {
				y[i] = math.NaN()
				continue
			}
		}
	}
	return residual, nil
}

func (f *Forecaster) fitResidual(t []time.Time, residual []float64) error {
	if len(residual) < MinResidualSize {
		return ErrInsufficientResidual
	}
	// compute rolling window standard deviation of residual for uncertaninty bands
	// the window is not necessarily a block of continuous time but could jump across
	// outlier points

	// limit residual window to a quarter of the resulting residual output
	if len(residual)/MinResidualWindowFactor < f.opt.ResidualWindow {
		f.opt.ResidualWindow = len(residual) / MinResidualWindowFactor
	}
	if f.opt.ResidualWindow < MinResidualWindow {
		f.opt.ResidualWindow = MinResidualWindow
	}

	stddevSeries := make([]float64, len(residual)-f.opt.ResidualWindow+1)
	numWindows := len(residual) - f.opt.ResidualWindow + 1

	for i := 0; i < numWindows; i++ {
		_, stddev := stat.MeanStdDev(residual[i:i+f.opt.ResidualWindow], nil)
		stddevSeries[i] = f.opt.ResidualZscore * stddev
	}

	// shifting by half the residual window since computing the residual series is similar to a
	// finite impulse response filtering having a group delay of window/2.
	start := f.opt.ResidualWindow / 2
	end := len(t) - f.opt.ResidualWindow/2 - f.opt.ResidualWindow%2 + 1

	residualData, err := timedataset.NewUnivariateDataset(t[start:end], stddevSeries)
	if err != nil {
		return fmt.Errorf("unable to create univariate dataset for residual, %w", err)
	}

	if err := f.residualForecast.Fit(residualData.T, residualData.Y); err != nil {
		return fmt.Errorf("unable to forecast residual, %w", err)
	}

	return nil
}

// Predict takes in any set of time samples and generates a forecast, upper, lower values per time point
func (f *Forecaster) Predict(t []time.Time) (*Results, error) {
	seriesRes, err := f.seriesForecast.Predict(t)
	if err != nil {
		return nil, fmt.Errorf("unable to predict series forecasts, %w", err)
	}
	residualRes, err := f.residualForecast.Predict(t)
	if err != nil {
		return nil, fmt.Errorf("unable to predict residual forecasts, %w", err)
	}

	// cap residual predictions to be greater than or equal to 0
	for i := 0; i < len(residualRes); i++ {
		if residualRes[i] < 0.0 {
			residualRes[i] = 0.0
		}
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

// Residuals returns the difference between the final series fit against the training data
func (f *Forecaster) Residuals() []float64 {
	return f.seriesForecast.Residuals()
}

// TrendComponent returns the trend component created by changepoints after fitting
func (f *Forecaster) TrendComponent() []float64 {
	return f.seriesForecast.TrendComponent()
}

// SeasonalityComponent returns the seasonality component after fitting the fourier series
func (f *Forecaster) SeasonalityComponent() []float64 {
	return f.seriesForecast.SeasonalityComponent()
}

// SeriesIntercept returns the intercept of the series fit
func (f *Forecaster) SeriesIntercept() float64 {
	return f.seriesForecast.Intercept()
}

// SeriesCoefficients returns all coefficient weight associated with the component label string
func (f *Forecaster) SeriesCoefficients() (map[string]float64, error) {
	return f.seriesForecast.Coefficients()
}

// ResidualIntercept returns the intercept of the uncertainty fit
func (f *Forecaster) ResidualIntercept() float64 {
	return f.residualForecast.Intercept()
}

// ResidualCoefficients returns all uncertainty coefficient weights associated with the component label string
func (f *Forecaster) ResidualCoefficients() (map[string]float64, error) {
	return f.residualForecast.Coefficients()
}

// Model generates a serializeable representaioon of the fit options, series model, and uncertainty model. This
// can be used to initialize a new Forecaster for immediate predictions skipping the training step.
func (f *Forecaster) Model() Model {
	m := Model{
		Options:  f.opt,
		Series:   f.seriesForecast.Model(),
		Residual: f.residualForecast.Model(),
	}
	return m
}

// SeriesModelEq returns a string representation of the fit series model represented as
// y ~ b + m1x1 + m2x2 ...
func (f *Forecaster) SeriesModelEq() (string, error) {
	return f.seriesForecast.ModelEq()
}

// ResidualModelEq returns a string representation of the fit uncertainty model represented as
// y ~ b + m1x1 + m2x2 ...
func (f *Forecaster) ResidualModelEq() (string, error) {
	return f.residualForecast.ModelEq()
}

// TrainingData returns the training data used to fit the current forecaster model
func (f *Forecaster) TrainingData() *timedataset.TimeDataset {
	return f.fitTrainingData
}

// FitResults returns the results of the fit which includes the forecast, upper, and lower values
func (f *Forecaster) FitResults() *Results {
	return f.fitResults
}

// PlotFit uses the Apache Echarts library to generate an html file showing the resulting fit,
// model components, and fit residual
func (f *Forecaster) PlotFit(path string) error {
	td := f.TrainingData()
	page := components.NewPage()
	page.AddCharts(
		LineForecaster(td, f.fitResults),
		LineTSeries(
			"Forecast Components",
			[]string{"Trend", "Seasonality"},
			td.T,
			[][]float64{
				f.TrendComponent(),
				f.SeasonalityComponent(),
			},
		),
		LineTSeries(
			"Forecast Residual",
			[]string{"Residual"},
			td.T,
			[][]float64{f.Residuals()},
		),
	)
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	return page.Render(io.MultiWriter(file))
}
