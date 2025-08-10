// Package forecaster is the high level forecasting package for fitting and predicting
// other time points given a single time series. This will generate a model for both the
// forecast values as well as the uncertainty value. Uncertainty here represents the lower
// and upper bounds to be expected of a time series. This has only univariate time series
// support.
package forecaster

import (
	"errors"
	"fmt"
	"io"
	"math"
	"time"

	"github.com/aouyang1/go-forecaster/forecast"
	"github.com/aouyang1/go-forecaster/models"
	"github.com/aouyang1/go-forecaster/stats"
	"github.com/aouyang1/go-forecaster/timedataset"
	"github.com/go-echarts/go-echarts/v2/components"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/stat"
)

var (
	ErrInsufficientResidual         = errors.New("insufficient samples from residual after outlier removal")
	ErrEmptyTimeDataset             = errors.New("no timedataset or uninitialized")
	ErrCannotInferInterval          = errors.New("cannot infer interval from training data time")
	ErrNoSeriesOrUncertaintyOptions = errors.New("no series or uncertainty options provided")
)

const (
	MinResidualWindow       = 2
	MinResidualSize         = 2
	MinResidualWindowFactor = 4
)

// Forecaster fits a forecast model and can be used to generate forecasts
type Forecaster struct {
	opt *Options

	seriesForecast      *forecast.Forecast
	uncertaintyForecast *forecast.Forecast

	fitTrainingData *timedataset.TimeDataset
	fitResults      *Results
	residual        []float64
	uncertainty     []float64
}

// New creates a new instance of a Forecaster using thhe provided options. If no options are provided
// a default is used.
func New(opt *Options) (*Forecaster, error) {
	if opt == nil {
		opt = NewDefaultOptions()
	}

	if opt.SeriesOptions == nil || opt.UncertaintyOptions == nil {
		return nil, ErrNoSeriesOrUncertaintyOptions
	}
	f := &Forecaster{
		opt: opt,
	}

	seriesForecast, err := forecast.New(f.opt.SeriesOptions.ForecastOptions)
	if err != nil {
		return nil, fmt.Errorf("unable to initialize series forecast, %w", err)
	}
	f.seriesForecast = seriesForecast

	uncertaintyForecast, err := forecast.New(f.opt.UncertaintyOptions.ForecastOptions)
	if err != nil {
		return nil, fmt.Errorf("unable to initialize uncertainty forecast, %w", err)
	}
	f.uncertaintyForecast = uncertaintyForecast
	return f, nil
}

// NewFromModel creates a new instance of Forecaster from a pre-existing model. This should be generated from
// from a previous forecaster call to Model().
func NewFromModel(model Model) (*Forecaster, error) {
	if model.Options == nil {
		return nil, forecast.ErrNoOptionsInModel
	}
	opt := model.Options
	opt.SeriesOptions.ForecastOptions = model.Series.Options
	opt.UncertaintyOptions.ForecastOptions = model.Uncertainty.Options

	seriesForecast, err := forecast.NewFromModel(model.Series)
	if err != nil {
		return nil, fmt.Errorf("unable to load from series model, %w", err)
	}
	uncertaintyForecast, err := forecast.NewFromModel(model.Uncertainty)
	if err != nil {
		return nil, fmt.Errorf("unable to load from uncertainty model, %w", err)
	}
	f := &Forecaster{
		opt:                 opt,
		seriesForecast:      seriesForecast,
		uncertaintyForecast: uncertaintyForecast,
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

	residual, err := f.fitSeriesWithOutliers(td.T, td.Y, f.seriesForecast)
	if err != nil {
		return err
	}

	// create residual to align with original time window since td.T may have changed
	// after outlier removal
	f.residual = make([]float64, len(t))
	var j int
	for i := range len(t) {
		if t[i].Equal(td.T[j]) {
			f.residual[i] = residual[j]
			j += 1
		} else {
			f.residual[i] = math.NaN()
		}
	}

	uncertaintySeries, err := f.generateUncertaintySeries(residual)
	if err != nil {
		return fmt.Errorf("unable to generate uncertainty series, %w", err)
	}

	// shifting time by half the residual window since computing the uncertainty series is similar to a
	// finite impulse response filtering having a group delay of window/2.
	start := f.opt.UncertaintyOptions.ResidualWindow / 2
	end := len(t) - f.opt.UncertaintyOptions.ResidualWindow/2 - f.opt.UncertaintyOptions.ResidualWindow%2 + 1

	// create uncertainty to align with original time window since td.T may have changed
	// after outlier removal
	f.uncertainty = make([]float64, len(t))
	var k int
	for i := range len(t) {
		if t[i].Equal(td.T[k+start]) && k < len(uncertaintySeries) {
			f.uncertainty[i] = uncertaintySeries[k]
			k += 1
		} else {
			f.uncertainty[i] = math.NaN()
		}
	}

	if err := f.fitUncertainty(td.T[start:end], uncertaintySeries, f.uncertaintyForecast); err != nil {
		return err
	}

	f.fitResults, err = f.Predict(t)
	if err != nil {
		return fmt.Errorf("unable to get predicted values from training set, %w", err)
	}

	return nil
}

func (f *Forecaster) fitSeriesWithOutliers(t []time.Time, y []float64, seriesForecast *forecast.Forecast) ([]float64, error) {
	outlierOpts := f.opt.SeriesOptions.OutlierOptions

	// iterate to remove outliers
	numPasses := 0
	if outlierOpts != nil {
		numPasses = outlierOpts.NumPasses
	}

	var residual []float64
	for i := 0; i <= numPasses; i++ {
		if err := seriesForecast.Fit(t, y); err != nil {
			return nil, fmt.Errorf("unable to forecast series, %w", err)
		}

		residual = seriesForecast.Residuals()

		// break out if no outlier options provided
		if f.opt.SeriesOptions.OutlierOptions == nil {
			break
		}

		outlierIdxs := stats.DetectOutliers(
			residual,
			outlierOpts.LowerPercentile,
			outlierOpts.UpperPercentile,
			outlierOpts.TukeyFactor,
		)
		outlierSet := make(map[int]struct{})
		for _, idx := range outlierIdxs {
			outlierSet[idx] = struct{}{}
		}

		// no more outliers detected with outlier options so break early
		if len(outlierIdxs) == 0 {
			break
		}

		for i := range len(t) {
			if _, exists := outlierSet[i]; exists {
				y[i] = math.NaN()
				continue
			}
		}
	}
	return residual, nil
}

// generateUncertaintySeries creates the uncertainty series by computing the rolling standard deviation
// of the residual scaled by the configured z-score.
func (f *Forecaster) generateUncertaintySeries(residual []float64) ([]float64, error) {
	if len(residual) < MinResidualSize {
		return nil, ErrInsufficientResidual
	}
	// compute rolling window standard deviation of residual for uncertaninty bands
	// the window is not necessarily a block of continuous time but could jump across
	// outlier points

	// limit residual window to some factor of the resulting residual output
	resWindow := f.opt.UncertaintyOptions.ResidualWindow
	resWindow = min(len(residual)/MinResidualWindowFactor, resWindow)
	resWindow = max(MinResidualWindow, resWindow)
	f.opt.UncertaintyOptions.ResidualWindow = resWindow

	stddevSeries := make([]float64, len(residual)-resWindow+1)
	numWindows := len(residual) - resWindow + 1

	for i := range numWindows {
		resWindow := residual[i : i+resWindow]

		// move all nans to the front so we only compute standard deviation off of non-nan values
		var ptr int
		for j := range len(resWindow) {
			if math.IsNaN(resWindow[j]) {
				val := resWindow[ptr]
				resWindow[ptr] = resWindow[j]
				resWindow[j] = val
				ptr += 1
			}
		}
		_, stddev := stat.MeanStdDev(resWindow[ptr:], nil)
		stddevSeries[i] = f.opt.UncertaintyOptions.ResidualZscore * stddev
	}
	return stddevSeries, nil
}

func (f *Forecaster) fitUncertainty(t []time.Time, uncertaintySeries []float64, uncertaintyForecast *forecast.Forecast) error {
	uncertaintyData, err := timedataset.NewUnivariateDataset(t, uncertaintySeries)
	if err != nil {
		return fmt.Errorf("unable to create univariate dataset for uncertainty, %w", err)
	}

	if err := uncertaintyForecast.Fit(uncertaintyData.T, uncertaintyData.Y); err != nil {
		return fmt.Errorf("unable to forecast uncertainty, %w", err)
	}

	return nil
}

// Predict takes in any set of time samples and generates a forecast, upper, lower values per time point
func (f *Forecaster) Predict(t []time.Time) (*Results, error) {
	seriesRes, seriesComp, err := f.seriesForecast.Predict(t)
	if err != nil {
		return nil, fmt.Errorf("unable to predict series forecasts, %w", err)
	}
	uncertaintyRes, uncertaintyComp, err := f.uncertaintyForecast.Predict(t)
	if err != nil {
		return nil, fmt.Errorf("unable to predict uncertainty forecasts, %w", err)
	}

	// cap uncertainty predictions to be greater than or equal to 0
	for i := range len(uncertaintyRes) {
		if uncertaintyRes[i] < 0.0 {
			uncertaintyRes[i] = 0.0
		}
	}

	r := &Results{
		T:                     t,
		Forecast:              seriesRes,
		SeriesComponents:      seriesComp,
		UncertaintyComponents: uncertaintyComp,
	}
	upper := make([]float64, len(seriesRes))
	lower := make([]float64, len(seriesRes))

	copy(upper, seriesRes)
	copy(lower, seriesRes)

	floats.Add(upper, uncertaintyRes)
	floats.Sub(lower, uncertaintyRes)

	// clip data if specified in options
	f.clip(r.Forecast)
	f.clip(upper)
	f.clip(lower)

	r.Upper = upper
	r.Lower = lower
	return r, nil
}

// Score computes the coefficient of determination of the prediction
func (f *Forecaster) Score(t []time.Time, y []float64) (float64, error) {
	if t == nil {
		return 0.0, fmt.Errorf("no time slice for inference, %w", models.ErrNoDesignMatrix)
	}
	if y == nil {
		return 0.0, fmt.Errorf("no expected values for inference, %w", models.ErrNoTargetMatrix)
	}

	m := len(t)

	ym := len(y)
	if m != ym {
		return 0.0, fmt.Errorf("design matrix has %d rows and target has %d rows, %w", m, ym, models.ErrTargetLenMismatch)
	}

	res, err := f.Predict(t)
	if err != nil {
		return 0.0, err
	}

	return stat.RSquaredFrom(res.Forecast, y, nil), nil
}

// Residuals returns the difference between the final series fit against the training data
func (f *Forecaster) Residuals() []float64 {
	return f.residual
}

// Uncertainty returns the uncertainty series used to forecast the upper lower bounds
func (f *Forecaster) Uncertainty() []float64 {
	return f.uncertainty
}

// TrendComponent returns the trend component created by changepoints after fitting
func (f *Forecaster) TrendComponent() []float64 {
	return f.seriesForecast.TrendComponent()
}

// SeasonalityComponent returns the seasonality component after fitting the fourier series
func (f *Forecaster) SeasonalityComponent() []float64 {
	return f.seriesForecast.SeasonalityComponent()
}

// EventComponent returns the event component after fitting the fourier series
func (f *Forecaster) EventComponent() []float64 {
	return f.seriesForecast.EventComponent()
}

// SeriesIntercept returns the intercept of the series fit
func (f *Forecaster) SeriesIntercept() (float64, error) {
	return f.seriesForecast.Intercept()
}

// SeriesCoefficients returns all coefficient weight associated with the component label string
func (f *Forecaster) SeriesCoefficients() (map[string]float64, error) {
	return f.seriesForecast.Coefficients()
}

// UncertaintyIntercept returns the intercept of the uncertainty fit
func (f *Forecaster) UncertaintyIntercept() (float64, error) {
	return f.uncertaintyForecast.Intercept()
}

// UncertaintyCoefficients returns all uncertainty coefficient weights associated with the component label string
func (f *Forecaster) UncertaintyCoefficients() (map[string]float64, error) {
	return f.uncertaintyForecast.Coefficients()
}

// Model generates a serializeable representaioon of the fit options, series model, and uncertainty model. This
// can be used to initialize a new Forecaster for immediate predictions skipping the training step.
func (f *Forecaster) Model() (Model, error) {
	seriesModel, err := f.seriesForecast.Model()
	if err != nil {
		return Model{}, fmt.Errorf("unable to fetch series model, %w", err)
	}
	uncertaintyModel, err := f.uncertaintyForecast.Model()
	if err != nil {
		return Model{}, fmt.Errorf("unable to fetch uncertainty moodel, %w", err)
	}
	m := Model{
		Options:     f.opt,
		Series:      seriesModel,
		Uncertainty: uncertaintyModel,
	}
	return m, nil
}

// SeriesModelEq returns a string representation of the fit series model represented as
// y ~ b + m1x1 + m2x2 ...
func (f *Forecaster) SeriesModelEq() (string, error) {
	return f.seriesForecast.ModelEq()
}

// UncertaintyModelEq returns a string representation of the fit uncertainty model represented as
// y ~ b + m1x1 + m2x2 ...
func (f *Forecaster) UncertaintyModelEq() (string, error) {
	return f.uncertaintyForecast.ModelEq()
}

// TrainingData returns the training data used to fit the current forecaster model
func (f *Forecaster) TrainingData() *timedataset.TimeDataset {
	return f.fitTrainingData
}

// FitResults returns the results of the fit which includes the forecast, upper, and lower values
func (f *Forecaster) FitResults() *Results {
	return f.fitResults
}

// MakeFuturePeriods generates a slice of time after the last point in the training data. By default
// a zero freq will be inferred from the training data.
func (f *Forecaster) MakeFuturePeriods(periods int, freq time.Duration) ([]time.Time, error) {
	td := f.TrainingData()
	t := timedataset.TimeSlice(td.T)
	lastTime := t.EndTime()

	if freq == 0 {
		var err error
		freq, err = t.EstimateFreq()
		if err != nil {
			return nil, err
		}
	}
	horizon := make([]time.Time, 0, periods)
	for i := range periods {
		nextT := lastTime.Add(time.Duration(i+1) * freq)
		horizon = append(horizon, nextT)
	}
	return horizon, nil
}

// PlotOpts sets the horizon to forecast out. By default will use 10% of the training size assuming
// even intervals between points and the first two points are used to infer the horizon interval.
type PlotOpts struct {
	HorizonCnt      int
	HorizonInterval time.Duration
}

// PlotFit uses the Apache Echarts library to generate an html file showing the resulting fit,
// model components, and fit residual
func (f *Forecaster) PlotFit(w io.Writer, opt *PlotOpts) error {
	td := f.TrainingData()

	horizonCnt := len(td.T) / 10
	var horizonInterval time.Duration
	if opt != nil {
		horizonCnt = opt.HorizonCnt
		horizonInterval = opt.HorizonInterval
	}
	if horizonCnt < 1 {
		horizonCnt = 1
	}
	horizon, err := f.MakeFuturePeriods(horizonCnt, horizonInterval)
	if err != nil {
		return err
	}

	t := make([]time.Time, len(td.T), len(td.T)+horizonCnt)
	copy(t, td.T)
	t = append(t, horizon...)

	zpad := make([]float64, 0, horizonCnt)
	for i := 0; i < horizonCnt; i++ {
		zpad = append(zpad, math.NaN())
	}

	forecastRes, err := f.Predict(horizon)
	if err != nil {
		return fmt.Errorf("unable to predict with horizon, %w", err)
	}

	residuals := f.Residuals()
	residuals = append(residuals, zpad...)

	uncertainty := f.Uncertainty()
	uncertainty = append(uncertainty, zpad...)

	trendComp := f.TrendComponent()
	trendComp = append(trendComp, forecastRes.SeriesComponents.Trend...)

	seasonComp := f.SeasonalityComponent()
	seasonComp = append(seasonComp, forecastRes.SeriesComponents.Seasonality...)

	eventComp := f.EventComponent()
	eventComp = append(eventComp, forecastRes.SeriesComponents.Event...)

	page := components.NewPage()
	page.AddCharts(
		LineForecaster(td, f.fitResults, forecastRes),
		LineTSeries(
			"Forecast Components",
			[]string{"Trend", "Seasonality", "Event"},
			t,
			[][]float64{
				trendComp,
				seasonComp,
				eventComp,
			},
			len(td.T),
		),
		LineTSeries(
			"Forecast Residual",
			[]string{"Residual", "Uncertainty"},
			t,
			[][]float64{
				residuals,
				uncertainty,
			},
			len(td.T),
		),
	)
	return page.Render(w)
}

func (f *Forecaster) clip(series []float64) {
	var clipMin, clipMax bool
	var minVal, maxVal float64
	if f.opt.MinValue != nil {
		clipMin = true
		minVal = *f.opt.MinValue
	}
	if f.opt.MaxValue != nil {
		clipMax = true
		maxVal = *f.opt.MaxValue
	}
	if !clipMin && !clipMax {
		return
	}

	for i := range len(series) {
		if clipMin && series[i] < minVal {
			series[i] = minVal
			continue
		}
		if clipMax && series[i] > maxVal {
			series[i] = maxVal
			continue
		}
	}
}
