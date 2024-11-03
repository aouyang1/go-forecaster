package forecaster

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"runtime/debug"
	"testing"
	"time"

	"gonum.org/v1/gonum/mat"

	"github.com/aouyang1/go-forecaster/changepoint"
	"github.com/aouyang1/go-forecaster/feature"
	"github.com/aouyang1/go-forecaster/forecast"
	"github.com/aouyang1/go-forecaster/timedataset"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gonum.org/v1/gonum/floats"
)

func generateT(n int, interval time.Duration) []time.Time {
	t := make([]time.Time, 0, n)
	ct := time.Now()
	for i := 0; i < n; i++ {
		t = append(t, ct.Add(interval*time.Duration(i)))
	}
	return t
}

type series []float64

func (s series) add(src series) series {
	floats.Add(s, src)
	return s
}

func (s series) setConst(t []time.Time, val float64, start, end time.Time) series {
	n := len(s)
	for i := 0; i < n; i++ {
		if (t[i].After(start) || t[i].Equal(start)) && t[i].Before(end) {
			s[i] = val
		}
	}
	return s
}

func generateConstY(n int, val float64) series {
	y := make([]float64, 0, n)
	for i := 0; i < n; i++ {
		y = append(y, val)
	}
	return series(y)
}

func generateWaveY(t []time.Time, amp, periodSec, order, timeOffset float64) series {
	n := len(t)
	y := make([]float64, 0, n)
	for i := 0; i < n; i++ {
		val := amp * math.Sin(2.0*math.Pi*order/periodSec*(float64(t[i].Unix())+timeOffset))
		y = append(y, val)
	}
	return series(y)
}

func generateNoise(t []time.Time, noiseScale, amp, periodSec, order, timeOffset float64) series {
	n := len(t)
	y := make([]float64, 0, n)
	for i := 0; i < n; i++ {
		scale := (noiseScale + amp*math.Sin(2.0*math.Pi*order/periodSec*(float64(t[i].Unix())+timeOffset)))
		y = append(y, rand.NormFloat64()*scale)
	}
	return series(y)
}

func generateChange(t []time.Time, chpt time.Time, bias, slope float64) series {
	n := len(t)
	y := make([]float64, n)
	for i := 0; i < n; i++ {
		if t[i].After(chpt) || t[i].Equal(chpt) {
			jump := bias + slope*t[i].Sub(chpt).Minutes()
			y[i] = jump
		}
	}
	return series(y)
}

func compareScores(t *testing.T, expected, actual *forecast.Scores, msg string) {
	assert.InDelta(t, expected.R2, actual.R2, 0.05, msg+" scores:r2")
	assert.InDelta(t, expected.MAPE, actual.MAPE, 0.20, msg+" scores:mape")

	mse := actual.MSE
	if expected.MSE > 0 {
		mse = math.Abs((actual.MSE - expected.MSE) / expected.MSE)
	}
	assert.LessOrEqual(t, mse, 0.20, msg+" scores:mse")
}

func compareCoef(t *testing.T, expected, actual []forecast.FeatureWeight, tol float64, msg string) {
	var significantFeatures []forecast.FeatureWeight
	for _, fw := range actual {
		if fw.Value >= tol {
			significantFeatures = append(significantFeatures, fw)
		}
	}
	require.Equal(t, len(expected), len(significantFeatures), msg+" number of significant series coefficients")
	for i := 0; i < len(significantFeatures); i++ {
		assert.Equal(t, expected[i].Type, significantFeatures[i].Type, msg+" feature weight type")
		assert.Equal(t, expected[i].Labels, significantFeatures[i].Labels, msg+" feature weight labels")
		expectedVal := expected[i].Value
		actualVal := significantFeatures[i].Value
		percDiff := actualVal
		if expectedVal > 0 {
			percDiff = math.Abs((actualVal - expectedVal) / expectedVal)
		}
		assert.LessOrEqual(t, percDiff, 0.05, fmt.Sprintf("%s feature weight value, %.3f, %+v", msg, actualVal, expected[i].Labels))
	}
}

func TestForecaster(t *testing.T) {
	testData := map[string]struct {
		expectedErr   error
		opt           *Options
		t             []time.Time
		y             []float64
		expectedModel Model
		tol           float64
	}{
		"no data": {
			t: nil, y: nil,
			expectedErr: timedataset.ErrNoTrainingData,
		},
		"all nan": {
			t:           generateT(10, time.Minute),
			y:           generateConstY(10, math.NaN()),
			expectedErr: forecast.ErrInsufficientTrainingData,
		},
		"all constant": {
			t:   generateT(10, time.Minute),
			y:   generateConstY(10, 3.0),
			tol: 1e-5,
			expectedModel: Model{
				Series: forecast.Model{
					Scores: &forecast.Scores{
						MAPE: 0.0,
						MSE:  0.0,
						R2:   1.0,
					},
					Weights: forecast.Weights{
						Intercept: 3.0,
						Coef:      []forecast.FeatureWeight{},
					},
				},
				Uncertainty: forecast.Model{
					Scores: &forecast.Scores{
						MAPE: 0.0,
						MSE:  0.0,
						R2:   1.0,
					},
					Weights: forecast.Weights{
						Intercept: 0.0,
						Coef:      []forecast.FeatureWeight{},
					},
				},
			},
		},
		"daily wave with bias": {
			t: generateT(4*24*60, time.Minute),
			y: generateConstY(4*24*60, 3.0).
				add(generateWaveY(generateT(4*24*60, time.Minute), 7.2, 86400.0, 1.0, 0.0)),
			tol: 1e-5,
			opt: &Options{
				SeriesOptions: &SeriesOptions{
					ForecastOptions: &forecast.Options{
						DailyOrders: 2,
					},
					OutlierOptions: NewOutlierOptions(),
				},
				UncertaintyOptions: &UncertaintyOptions{
					ResidualWindow: 100,
					ResidualZscore: 4.0,
				},
			},
			expectedModel: Model{
				Series: forecast.Model{
					Scores: &forecast.Scores{
						MAPE: 0.0,
						MSE:  0.0,
						R2:   1.0,
					},
					Weights: forecast.Weights{
						Intercept: 3.0,
						Coef: []forecast.FeatureWeight{
							{
								Labels: map[string]string{
									"name":              "hod",
									"order":             "1",
									"fourier_component": "sin",
								},
								Type:  feature.FeatureTypeSeasonality,
								Value: 7.2,
							},
						},
					},
				},
				Uncertainty: forecast.Model{
					Scores: &forecast.Scores{
						MAPE: 0.03426,
						MSE:  0.0,
						R2:   0.99,
					},
					Weights: forecast.Weights{
						Intercept: 0.0,
						Coef:      []forecast.FeatureWeight{},
					},
				},
			},
		},
		"daily and weekly wave with bias": {
			t: generateT(14*24*60, time.Minute),
			y: generateConstY(14*24*60, 3.0).
				add(generateWaveY(generateT(14*24*60, time.Minute), 7.2, 24*60*60, 1.0, 0.0)).
				add(generateWaveY(generateT(14*24*60, time.Minute), 4.6, 7*24*60*60, 1.0, 0.0)),
			tol: 1e-5,
			opt: &Options{
				SeriesOptions: &SeriesOptions{
					ForecastOptions: &forecast.Options{
						DailyOrders:  2,
						WeeklyOrders: 2,
					},
					OutlierOptions: NewOutlierOptions(),
				},
				UncertaintyOptions: &UncertaintyOptions{
					ResidualWindow: 100,
					ResidualZscore: 4.0,
				},
			},
			expectedModel: Model{
				Series: forecast.Model{
					Scores: &forecast.Scores{
						MAPE: 0.0,
						MSE:  0.0,
						R2:   1.0,
					},
					Weights: forecast.Weights{
						Intercept: 3.0,
						Coef: []forecast.FeatureWeight{
							{
								Labels: map[string]string{
									"name":              "dow",
									"order":             "1",
									"fourier_component": "sin",
								},
								Type:  feature.FeatureTypeSeasonality,
								Value: 4.6,
							},
							{
								Labels: map[string]string{
									"name":              "hod",
									"order":             "1",
									"fourier_component": "sin",
								},
								Type:  feature.FeatureTypeSeasonality,
								Value: 7.2,
							},
						},
					},
				},
				Uncertainty: forecast.Model{
					Scores: &forecast.Scores{
						MAPE: 0.04704,
						MSE:  0.0,
						R2:   0.99,
					},
					Weights: forecast.Weights{
						Intercept: 0.0,
						Coef:      []forecast.FeatureWeight{},
					},
				},
			},
		},
		"daily and weekly wave with bias with noise": {
			t: generateT(14*24*60, time.Minute),
			y: generateConstY(14*24*60, 98.3).
				add(generateWaveY(generateT(14*24*60, time.Minute), 10.5, 24*60*60, 1.0, 0.0)).
				add(generateWaveY(generateT(14*24*60, time.Minute), 7.6, 24*60*60, 3.0, 0.0)).
				add(generateWaveY(generateT(14*24*60, time.Minute), 4.6, 7*24*60*60, 1.0, 0.0)).
				add(generateNoise(generateT(14*24*60, time.Minute), 3.2, 3.2, 24*60*60, 5.0, 0.0)),
			tol: 1.0,
			opt: &Options{
				SeriesOptions: &SeriesOptions{
					ForecastOptions: &forecast.Options{
						DailyOrders:  4,
						WeeklyOrders: 2,
					},
					OutlierOptions: NewOutlierOptions(),
				},
				UncertaintyOptions: &UncertaintyOptions{
					ForecastOptions: &forecast.Options{
						DailyOrders: 6,
					},
					ResidualWindow: 100,
					ResidualZscore: 1.0,
				},
			},
			expectedModel: Model{
				Series: forecast.Model{
					Scores: &forecast.Scores{
						MAPE: 0.02529,
						MSE:  13.5823,
						R2:   0.8739,
					},
					Weights: forecast.Weights{
						Intercept: 98.3,
						Coef: []forecast.FeatureWeight{
							{
								Labels: map[string]string{
									"name":              "dow",
									"order":             "1",
									"fourier_component": "sin",
								},
								Type:  feature.FeatureTypeSeasonality,
								Value: 4.6,
							},
							{
								Labels: map[string]string{
									"name":              "hod",
									"order":             "1",
									"fourier_component": "sin",
								},
								Type:  feature.FeatureTypeSeasonality,
								Value: 10.5,
							},
							{
								Labels: map[string]string{
									"name":              "hod",
									"order":             "3",
									"fourier_component": "sin",
								},
								Type:  feature.FeatureTypeSeasonality,
								Value: 7.6,
							},
						},
					},
				},
				Uncertainty: forecast.Model{
					Scores: &forecast.Scores{
						MAPE: 0.04704,
						MSE:  0.0,
						R2:   0.99,
					},
					Weights: forecast.Weights{
						Intercept: 3.3,
						Coef: []forecast.FeatureWeight{
							{
								Labels: map[string]string{
									"name":              "hod",
									"order":             "5",
									"fourier_component": "sin",
								},
								Type:  feature.FeatureTypeSeasonality,
								Value: 2.182,
							},
						},
					},
				},
			},
		},
	}

	for name, td := range testData {
		t.Run(name, func(t *testing.T) {
			defer func() {
				if r := recover(); r != nil {
					t.Logf("panic: %v\n", r)
					debug.PrintStack()
				}
			}()

			f, err := New(td.opt)
			require.Nil(t, err)

			err = f.Fit(td.t, td.y)
			if td.expectedErr != nil {
				require.ErrorAs(t, err, &td.expectedErr)
				return
			}
			require.Nil(t, err)

			m, err := f.Model()
			require.Nil(t, err)

			compareScores(t, td.expectedModel.Series.Scores, m.Series.Scores, "series")
			actualInt := m.Series.Weights.Intercept
			expectedInt := td.expectedModel.Series.Weights.Intercept
			percDiff := actualInt
			if expectedInt != 0 {
				percDiff = math.Abs((actualInt - expectedInt) / expectedInt)
			}
			assert.LessOrEqual(t, percDiff, 0.05, fmt.Sprintf("series intercept, %.3f", actualInt))
			compareCoef(t, td.expectedModel.Series.Weights.Coef, m.Series.Weights.Coef, td.tol, "series")

			actualInt = m.Uncertainty.Weights.Intercept
			expectedInt = td.expectedModel.Uncertainty.Weights.Intercept

			percDiff = actualInt
			if expectedInt != 0 {
				percDiff = math.Abs((actualInt - expectedInt) / expectedInt)
			}
			assert.LessOrEqual(t, percDiff, 0.05, fmt.Sprintf("uncertainty intercept, %.3f", actualInt))

			compareScores(t, td.expectedModel.Uncertainty.Scores, m.Uncertainty.Scores, "uncertainty")
			compareCoef(t, td.expectedModel.Uncertainty.Weights.Coef, m.Uncertainty.Weights.Coef, td.tol, "uncertainty")
		})
	}
}

func generateExampleSeries() ([]time.Time, []float64) {
	// create a daily sine wave at minutely with one week
	minutes := 28 * 24 * 60
	t := generateT(minutes, time.Minute)
	y := make(series, minutes)

	period := 86400.0
	y.add(generateConstY(minutes, 98.3)).
		add(generateWaveY(t, 10.5, period, 1.0, 2*60*60)).
		add(generateWaveY(t, 10.5, period, 3.0, 2.0*60*60+period/2.0/2.0/3.0)).
		add(generateNoise(t, 3.2, 3.2, period, 5.0, 0.0)).
		add(generateChange(t, t[minutes/2], 10.0, 0.0)).
		add(generateChange(t, t[minutes*2/3], 61.4, 0.0)).             // anomaly start
		add(generateChange(t, t[minutes*2/3+minutes/40], -61.4, 0.0)). // anomaly end
		add(generateChange(t, t[minutes*17/20], -70.0, 0.0)).
		setConst(t, 2.7, t[minutes/3], t[minutes/3+minutes/20])

	return t, y
}

func ExampleForecaster() {
	t, y := generateExampleSeries()

	changepoints := []changepoint.Changepoint{
		changepoint.New("anomaly1", t[len(t)/2]),
		changepoint.New("anomaly2", t[len(t)*17/20]),
	}

	opt := &Options{
		SeriesOptions: &SeriesOptions{
			ForecastOptions: &forecast.Options{
				DailyOrders:  12,
				WeeklyOrders: 12,
				ChangepointOptions: forecast.ChangepointOptions{
					Changepoints: changepoints,
				},
			},
			OutlierOptions: NewOutlierOptions(),
		},
		UncertaintyOptions: &UncertaintyOptions{
			ForecastOptions: &forecast.Options{
				DailyOrders:  12,
				WeeklyOrders: 12,
				ChangepointOptions: forecast.ChangepointOptions{
					Changepoints: nil,
				},
			},
			ResidualWindow: 100,
			ResidualZscore: 4.0,
		},
	}

	defer func() {
		if r := recover(); r != nil {
			fmt.Printf("panic: %v\n", r)
			debug.PrintStack()
		}
	}()

	f, err := New(opt)
	if err != nil {
		panic(err)
	}
	if err := f.Fit(t, y); err != nil {
		panic(err)
	}

	m, err := f.Model()
	if err != nil {
		panic(err)
	}
	if err := m.TablePrint(os.Stderr); err != nil {
		panic(err)
	}

	if err := f.PlotFit("examples/forecaster.html", nil); err != nil {
		panic(err)
	}
	// Output:
}

func generateExampleSeriesWithTrend() ([]time.Time, []float64) {
	// create a daily sine wave at minutely with one week
	minutes := 4 * 24 * 60
	t := generateT(minutes, time.Minute)
	y := make(series, minutes)

	period := 86400.0
	y.add(generateConstY(minutes, 98.3)).
		add(generateWaveY(t, 10.5, period, 1.0, 2*60*60)).
		add(generateWaveY(t, 10.5, period, 3.0, 2.0*60*60+period/2.0/2.0/3.0)).
		add(generateNoise(t, 3.2, 3.2, period, 5.0, 0.0)).
		add(generateChange(t, t[minutes/2], 0.0, 40.0/(t[minutes*17/20].Sub(t[minutes/2]).Minutes()))).
		add(generateChange(t, t[minutes*17/20], -40.0, -40.0/(t[minutes*17/20].Sub(t[minutes/2]).Minutes())))

	return t, y
}

func ExampleForecasterAutoChangepoint() {
	t, y := generateExampleSeries()

	opt := &Options{
		SeriesOptions: &SeriesOptions{
			ForecastOptions: &forecast.Options{
				Regularization: 200.0,
				DailyOrders:    12,
				WeeklyOrders:   12,
				ChangepointOptions: forecast.ChangepointOptions{
					Auto:                true,
					AutoNumChangepoints: 100,
					EnableGrowth:        true,
				},
			},
			OutlierOptions: NewOutlierOptions(),
		},
		UncertaintyOptions: &UncertaintyOptions{
			ForecastOptions: &forecast.Options{
				DailyOrders:  12,
				WeeklyOrders: 12,
				ChangepointOptions: forecast.ChangepointOptions{
					Auto:         false,
					Changepoints: []changepoint.Changepoint{},
				},
			},
			ResidualWindow: 100,
			ResidualZscore: 4.0,
		},
	}

	defer func() {
		if r := recover(); r != nil {
			fmt.Printf("panic: %v\n", r)
			debug.PrintStack()
		}
	}()

	f, err := New(opt)
	if err != nil {
		panic(err)
	}
	if err := f.Fit(t, y); err != nil {
		panic(err)
	}

	m, err := f.Model()
	if err != nil {
		panic(err)
	}
	if err := m.TablePrint(os.Stderr); err != nil {
		panic(err)
	}

	if err := f.PlotFit("examples/forecaster_auto_changepoint.html", nil); err != nil {
		panic(err)
	}
	// Output:
}

func ExampleForecasterWithTrend() {
	t, y := generateExampleSeriesWithTrend()

	changepoints := []changepoint.Changepoint{
		changepoint.New("trendstart", t[len(t)/2]),
		changepoint.New("rebaseline", t[len(t)*17/20]),
	}

	opt := &Options{
		SeriesOptions: &SeriesOptions{
			ForecastOptions: &forecast.Options{
				DailyOrders:  12,
				WeeklyOrders: 12,
				ChangepointOptions: forecast.ChangepointOptions{
					Changepoints: changepoints,
					EnableGrowth: true,
				},
			},
			OutlierOptions: NewOutlierOptions(),
		},
		UncertaintyOptions: &UncertaintyOptions{
			ForecastOptions: &forecast.Options{
				DailyOrders:  12,
				WeeklyOrders: 12,
				ChangepointOptions: forecast.ChangepointOptions{
					Changepoints: nil,
				},
			},
			ResidualWindow: 100,
			ResidualZscore: 4.0,
		},
	}

	defer func() {
		if r := recover(); r != nil {
			fmt.Printf("panic: %v\n", r)
			debug.PrintStack()
		}
	}()

	f, err := New(opt)
	if err != nil {
		panic(err)
	}
	if err := f.Fit(t, y); err != nil {
		panic(err)
	}

	m, err := f.Model()
	if err != nil {
		panic(err)
	}
	if err := m.TablePrint(os.Stderr); err != nil {
		panic(err)
	}

	if err := f.PlotFit("examples/forecaster_with_trend.html", nil); err != nil {
		panic(err)
	}
	// Output:
}

func TestMatrixMulWithNaN(t *testing.T) {
	// Initialize two matrices, a and b.
	a := mat.NewDense(1, 2, []float64{
		4, 3,
	})
	b := mat.NewDense(2, 4, []float64{
		4, 0, 0, math.NaN(),
		0, 0, 4, math.NaN(),
	})

	// Take the matrix product of a and b and place the result in c.
	var c mat.Dense
	c.Mul(a, b)

	// Print the result using the formatter.
	fc := mat.Formatted(&c, mat.Prefix("    "), mat.Squeeze())
	fmt.Printf("c = %v", fc)
	// Output: c = [16  0  12  NaN]
}
