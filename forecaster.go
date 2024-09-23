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

type Forecaster struct {
	opt *Options

	seriesForecast   *forecast.Forecast
	residualForecast *forecast.Forecast

	fitTrainingData *timedataset.TimeDataset
	fitResults      *Results
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

	start := f.opt.ResidualWindow / 2
	end := len(td.T) - f.opt.ResidualWindow/2 - f.opt.ResidualWindow%2 + 1

	residualData, err := timedataset.NewUnivariateDataset(td.T[start:end], stddevSeries)
	if err != nil {
		return fmt.Errorf("unable to create univariate dataset for residual, %w", err)
	}

	if err := f.residualForecast.Fit(residualData); err != nil {
		return fmt.Errorf("unable to forecast residual, %w", err)
	}

	f.fitTrainingData = trainingData
	f.fitResults, err = f.Predict(trainingData.T)
	if err != nil {
		return fmt.Errorf("unable to get predicted values from training set, %w", err)
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

func (f *Forecaster) SeriesCoefficients() (map[string]float64, error) {
	return f.seriesForecast.Coefficients()
}

func (f *Forecaster) ResidualIntercept() float64 {
	return f.residualForecast.Intercept()
}

func (f *Forecaster) ResidualCoefficients() (map[string]float64, error) {
	return f.residualForecast.Coefficients()
}

func (f *Forecaster) Model() Model {
	m := Model{
		Options:  f.opt,
		Series:   f.seriesForecast.Model(),
		Residual: f.residualForecast.Model(),
	}
	return m
}

func (f *Forecaster) SeriesModelEq() (string, error) {
	return f.seriesForecast.ModelEq()
}

func (f *Forecaster) ResidualModelEq() (string, error) {
	return f.residualForecast.ModelEq()
}

func (f *Forecaster) TrainingData() *timedataset.TimeDataset {
	return f.fitTrainingData
}

func (f *Forecaster) FitResults() *Results {
	return f.fitResults
}

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
