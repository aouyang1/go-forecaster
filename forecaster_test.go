package forecaster

import (
	"fmt"
	"math"
	"os"
	"runtime/debug"
	"testing"
	"time"

	"github.com/goccy/go-json"
	"gonum.org/v1/gonum/mat"

	"github.com/aouyang1/go-forecaster/feature"
	"github.com/aouyang1/go-forecaster/forecast"
	"github.com/aouyang1/go-forecaster/forecast/options"
	"github.com/aouyang1/go-forecaster/forecast/util"
	"github.com/aouyang1/go-forecaster/timedataset"
	"github.com/pkg/profile"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func compareScores(t *testing.T, expected, actual *forecast.Scores, msg string) {
	if actual.R2 >= 0 && expected.R2 >= 0 {
		assert.InDelta(t, expected.R2, actual.R2, 0.05, msg+" scores:r2")
	}
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
		if math.Abs(fw.Value) >= tol {
			significantFeatures = append(significantFeatures, fw)
		}
	}
	require.Equal(t, len(expected), len(significantFeatures), msg+" number of significant series coefficients, %v", significantFeatures)
	for i := 0; i < len(significantFeatures); i++ {
		assert.Equal(t, expected[i].Type, significantFeatures[i].Type, msg+" feature weight type")
		assert.Equal(t, expected[i].Labels, significantFeatures[i].Labels, msg+" feature weight labels")
		expectedVal := expected[i].Value
		actualVal := significantFeatures[i].Value
		percDiff := actualVal
		if expectedVal > 0 {
			percDiff = math.Abs((actualVal - expectedVal) / expectedVal)
		}
		assert.LessOrEqual(t, percDiff, 0.05, fmt.Sprintf("%s feature weight value, %.5f, %+v", msg, actualVal, expected[i].Labels))
	}
}

func TestForecaster(t *testing.T) {
	nowFunc := func() time.Time {
		return time.Date(1970, 2, 1, 0, 0, 0, 0, time.UTC)
	}

	testDailySeasonalityOptions := options.SeasonalityOptions{
		SeasonalityConfigs: []options.SeasonalityConfig{
			options.NewDailySeasonalityConfig(2),
		},
	}
	testDailyWeeklySeasonalityOptions := options.SeasonalityOptions{
		SeasonalityConfigs: []options.SeasonalityConfig{
			options.NewDailySeasonalityConfig(2),
			options.NewWeeklySeasonalityConfig(2),
		},
	}
	perfectScores := &forecast.Scores{
		MAPE: 0.0,
		MSE:  0.0,
		R2:   1.0,
	}

	allConstantOptions := &Options{
		SeriesOptions: &SeriesOptions{
			ForecastOptions: &options.Options{
				Regularization: []float64{0.0},
			},
		},
		UncertaintyOptions: &UncertaintyOptions{
			ForecastOptions: &options.Options{
				Regularization: []float64{0.0},
			},
			ResidualWindow: 50,
			ResidualZscore: 8.0,
		},
	}

	allConstantOptionsWithLog := &Options{
		SeriesOptions: &SeriesOptions{
			ForecastOptions: &options.Options{
				SeasonalityOptions: options.SeasonalityOptions{},
				UseLog:             true,
				Regularization:     []float64{0.0},
			},
		},
		UncertaintyOptions: &UncertaintyOptions{
			ForecastOptions: &options.Options{
				SeasonalityOptions: options.SeasonalityOptions{},
				Regularization:     []float64{0.0},
			},
			ResidualWindow: 50,
			ResidualZscore: 8.0,
		},
	}

	dailyOptions := &Options{
		SeriesOptions: &SeriesOptions{
			ForecastOptions: &options.Options{
				SeasonalityOptions: testDailySeasonalityOptions,
			},
			OutlierOptions: NewOutlierOptions(),
		},
		UncertaintyOptions: &UncertaintyOptions{
			ForecastOptions: &options.Options{
				SeasonalityOptions: testDailySeasonalityOptions,
				Regularization:     []float64{1.0},
			},
			ResidualWindow: 50,
			ResidualZscore: 8.0,
		},
	}

	dailyOptionsWithLog := &Options{
		SeriesOptions: &SeriesOptions{
			ForecastOptions: &options.Options{
				UseLog:             true,
				SeasonalityOptions: testDailySeasonalityOptions,
			},
			OutlierOptions: NewOutlierOptions(),
		},
		UncertaintyOptions: &UncertaintyOptions{
			ForecastOptions: &options.Options{
				SeasonalityOptions: testDailySeasonalityOptions,
				Regularization:     []float64{1.0},
			},
			ResidualWindow: 50,
			ResidualZscore: 8.0,
		},
	}

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
			t:           timedataset.GenerateT(10, time.Minute, nowFunc),
			y:           timedataset.GenerateConstY(10, math.NaN()),
			expectedErr: forecast.ErrInsufficientTrainingData,
		},
		"all constant": {
			t:   timedataset.GenerateT(10, time.Minute, nowFunc),
			y:   timedataset.GenerateConstY(10, 3.0),
			tol: 0.0,
			opt: allConstantOptions,
			expectedModel: Model{
				Series: forecast.Model{
					Scores: perfectScores,
					Weights: forecast.Weights{
						Coef: []forecast.FeatureWeight{
							{
								Labels: map[string]string{
									"name": "intercept",
								},
								Type:  feature.FeatureTypeGrowth,
								Value: 3.0,
							},
						},
					},
				},
				Uncertainty: forecast.Model{
					Scores: perfectScores,
					Weights: forecast.Weights{
						Coef: []forecast.FeatureWeight{
							{
								Labels: map[string]string{
									"name": "intercept",
								},
								Type:  feature.FeatureTypeGrowth,
								Value: 0.0,
							},
						},
					},
				},
			},
		},
		"all constant use log": {
			t:   timedataset.GenerateT(10, time.Minute, nowFunc),
			y:   timedataset.GenerateConstY(10, 3.0),
			tol: 0.0,
			opt: allConstantOptionsWithLog,
			expectedModel: Model{
				Series: forecast.Model{
					Scores: perfectScores,
					Weights: forecast.Weights{
						Coef: []forecast.FeatureWeight{
							{
								Labels: map[string]string{
									"name": "intercept",
								},
								Type:  feature.FeatureTypeGrowth,
								Value: math.Log1p(3.0),
							},
						},
					},
				},
				Uncertainty: forecast.Model{
					Scores: perfectScores,
					Weights: forecast.Weights{
						Coef: []forecast.FeatureWeight{
							{
								Labels: map[string]string{
									"name": "intercept",
								},
								Type:  feature.FeatureTypeGrowth,
								Value: 0.0,
							},
						},
					},
				},
			},
		},
		"daily wave with bias": {
			t: timedataset.GenerateT(4*24*60, time.Minute, nowFunc),
			y: timedataset.GenerateConstY(4*24*60, 3.3).
				Add(timedataset.GenerateWaveY(
					timedataset.GenerateT(4*24*60, time.Minute, nowFunc),
					7.2, 86400.0, 1.0, 0.0)),
			tol: 0.0,
			opt: dailyOptions,
			expectedModel: Model{
				Series: forecast.Model{
					Scores: perfectScores,
					Weights: forecast.Weights{
						Coef: []forecast.FeatureWeight{
							{
								Labels: map[string]string{
									"name": "intercept",
								},
								Type:  feature.FeatureTypeGrowth,
								Value: 3.3,
							},

							{
								Labels: map[string]string{
									"name":              "epoch_daily",
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
						MAPE: 1.0,
						MSE:  0.0,
						R2:   -4.54,
					},
					Weights: forecast.Weights{
						Coef: []forecast.FeatureWeight{
							{
								Labels: map[string]string{
									"name": "intercept",
								},
								Type:  feature.FeatureTypeGrowth,
								Value: 0.0,
							},
						},
					},
				},
			},
		},
		"daily wave with bias with log": {
			t: timedataset.GenerateT(4*24*60, time.Minute, nowFunc),
			y: timedataset.GenerateConstY(4*24*60, 14.3).
				Add(timedataset.GenerateWaveY(
					timedataset.GenerateT(4*24*60, time.Minute, nowFunc),
					7.2, 86400.0, 1.0, 0.0)),
			tol: 1e-3,
			opt: dailyOptionsWithLog,
			expectedModel: Model{
				Series: forecast.Model{
					Scores: perfectScores,
					Weights: forecast.Weights{
						Coef: []forecast.FeatureWeight{
							{
								Labels: map[string]string{
									"name": "intercept",
								},
								Type:  feature.FeatureTypeGrowth,
								Value: math.Log1p(14.3),
							},
							{
								Labels: map[string]string{
									"name":              "epoch_daily",
									"order":             "1",
									"fourier_component": "sin",
								},
								Type:  feature.FeatureTypeSeasonality,
								Value: 0.500,
							},
							{
								Labels: map[string]string{
									"name":              "epoch_daily",
									"order":             "2",
									"fourier_component": "cos",
								},
								Type:  feature.FeatureTypeSeasonality,
								Value: 0.062,
							},
						},
					},
				},
				Uncertainty: forecast.Model{
					Scores: &forecast.Scores{
						MAPE: 1.0,
						MSE:  0.0,
						R2:   -4.54,
					},
					Weights: forecast.Weights{
						Coef: []forecast.FeatureWeight{
							{
								Labels: map[string]string{
									"name": "intercept",
								},
								Type:  feature.FeatureTypeGrowth,
								Value: 0.1467,
							},
							{
								Labels: map[string]string{
									"name":              "epoch_daily",
									"order":             "1",
									"fourier_component": "cos",
								},
								Type:  feature.FeatureTypeSeasonality,
								Value: -0.001,
							},
							{
								Labels: map[string]string{
									"name":              "epoch_daily",
									"order":             "1",
									"fourier_component": "sin",
								},
								Type:  feature.FeatureTypeSeasonality,
								Value: 0.033,
							},
						},
					},
				},
			},
		},
		"daily wave with bias with log and negative": {
			t: timedataset.GenerateT(4*24*60, time.Minute, nowFunc),
			y: timedataset.GenerateConstY(4*24*60, 1.0).
				Add(timedataset.GenerateWaveY(
					timedataset.GenerateT(4*24*60, time.Minute, nowFunc),
					7.2, 86400.0, 1.0, 0.0)),
			expectedErr: util.ErrNegativeDataWithLog,
			opt:         dailyOptionsWithLog,
		},
		"daily and weekly wave with bias": {
			t: timedataset.GenerateT(14*24*60, time.Minute, nowFunc),
			y: timedataset.GenerateConstY(14*24*60, 3.0).
				Add(timedataset.GenerateWaveY(timedataset.GenerateT(14*24*60, time.Minute, nowFunc), 7.2, 24*60*60, 1.0, 0.0)).
				Add(timedataset.GenerateWaveY(timedataset.GenerateT(14*24*60, time.Minute, nowFunc), 4.6, 7*24*60*60, 1.0, 0.0)),
			tol: 0.0,
			opt: &Options{
				SeriesOptions: &SeriesOptions{
					ForecastOptions: &options.Options{
						SeasonalityOptions: testDailyWeeklySeasonalityOptions,
					},
					OutlierOptions: NewOutlierOptions(),
				},
				UncertaintyOptions: &UncertaintyOptions{
					ForecastOptions: &options.Options{
						SeasonalityOptions: testDailyWeeklySeasonalityOptions,
						Regularization:     []float64{1.0},
					},
					ResidualWindow: 100,
					ResidualZscore: 4.0,
				},
			},
			expectedModel: Model{
				Series: forecast.Model{
					Scores: perfectScores,
					Weights: forecast.Weights{
						Coef: []forecast.FeatureWeight{
							{
								Labels: map[string]string{
									"name": "intercept",
								},
								Type:  feature.FeatureTypeGrowth,
								Value: 3.0,
							},
							{
								Labels: map[string]string{
									"name":              "epoch_daily",
									"order":             "1",
									"fourier_component": "sin",
								},
								Type:  feature.FeatureTypeSeasonality,
								Value: 7.2,
							},
							{
								Labels: map[string]string{
									"name":              "epoch_weekly",
									"order":             "1",
									"fourier_component": "sin",
								},
								Type:  feature.FeatureTypeSeasonality,
								Value: 4.6,
							},
						},
					},
				},
				Uncertainty: forecast.Model{
					Scores: &forecast.Scores{
						MAPE: 1.0,
						MSE:  0.0,
						R2:   -4.28,
					},
					Weights: forecast.Weights{
						Coef: []forecast.FeatureWeight{
							{
								Labels: map[string]string{
									"name": "intercept",
								},
								Type:  feature.FeatureTypeGrowth,
								Value: 0.0,
							},
						},
					},
				},
			},
		},
		"daily and weekly wave with bias with noise": {
			t: timedataset.GenerateT(14*24*60, time.Minute, nowFunc),
			y: timedataset.GenerateConstY(14*24*60, 98.3).
				Add(timedataset.GenerateWaveY(timedataset.GenerateT(14*24*60, time.Minute, nowFunc), 10.5, 24*60*60, 1.0, 0.0)).
				Add(timedataset.GenerateWaveY(timedataset.GenerateT(14*24*60, time.Minute, nowFunc), 7.6, 24*60*60, 3.0, 0.0)).
				Add(timedataset.GenerateWaveY(timedataset.GenerateT(14*24*60, time.Minute, nowFunc), 4.6, 7*24*60*60, 1.0, 0.0)).
				Add(timedataset.GenerateNoise(timedataset.GenerateT(14*24*60, time.Minute, nowFunc), 3.2, 3.2, 24*60*60, 5.0, 0.0)),
			tol: 1.0,
			opt: &Options{
				SeriesOptions: &SeriesOptions{
					ForecastOptions: &options.Options{
						SeasonalityOptions: options.SeasonalityOptions{
							SeasonalityConfigs: []options.SeasonalityConfig{
								options.NewDailySeasonalityConfig(4),
								options.NewWeeklySeasonalityConfig(2),
							},
						},
					},
					OutlierOptions: NewOutlierOptions(),
				},
				UncertaintyOptions: &UncertaintyOptions{
					ForecastOptions: &options.Options{
						SeasonalityOptions: options.SeasonalityOptions{
							SeasonalityConfigs: []options.SeasonalityConfig{
								options.NewDailySeasonalityConfig(6),
							},
						},
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
						Coef: []forecast.FeatureWeight{
							{
								Labels: map[string]string{
									"name": "intercept",
								},
								Type:  feature.FeatureTypeGrowth,
								Value: 98.3,
							},
							{
								Labels: map[string]string{
									"name":              "epoch_daily",
									"order":             "1",
									"fourier_component": "sin",
								},
								Type:  feature.FeatureTypeSeasonality,
								Value: 10.5,
							},
							{
								Labels: map[string]string{
									"name":              "epoch_daily",
									"order":             "3",
									"fourier_component": "sin",
								},
								Type:  feature.FeatureTypeSeasonality,
								Value: 7.6,
							},
							{
								Labels: map[string]string{
									"name":              "epoch_weekly",
									"order":             "1",
									"fourier_component": "sin",
								},
								Type:  feature.FeatureTypeSeasonality,
								Value: 4.6,
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
						Coef: []forecast.FeatureWeight{
							{
								Labels: map[string]string{
									"name": "intercept",
								},
								Type:  feature.FeatureTypeGrowth,
								Value: 3.3,
							},
							{
								Labels: map[string]string{
									"name":              "epoch_daily",
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
		"weekend events with daily seasonality": {
			t: timedataset.GenerateT(7*24*60, time.Minute, nowFunc),
			y: timedataset.GenerateConstY(7*24*60, 50.0).
				Add(timedataset.GenerateWaveY(timedataset.GenerateT(7*24*60, time.Minute, nowFunc), 15.0, 24*60*60, 1.0, 0.0)).
				Add(timedataset.GenerateConstY(7*24*60, -20.0).
					MaskWithWeekend(timedataset.GenerateT(7*24*60, time.Minute, nowFunc)),
				),
			tol: 0,
			opt: &Options{
				SeriesOptions: &SeriesOptions{
					ForecastOptions: &options.Options{
						SeasonalityOptions: options.SeasonalityOptions{
							SeasonalityConfigs: []options.SeasonalityConfig{
								options.NewDailySeasonalityConfig(2),
							},
						},
						WeekendOptions: options.WeekendOptions{
							Enabled: true,
						},
					},
				},
				UncertaintyOptions: &UncertaintyOptions{
					ForecastOptions: &options.Options{
						SeasonalityOptions: options.SeasonalityOptions{
							SeasonalityConfigs: []options.SeasonalityConfig{
								options.NewDailySeasonalityConfig(2),
							},
						},
						WeekendOptions: options.WeekendOptions{
							Enabled: true,
						},
					},
					ResidualWindow: 50,
					ResidualZscore: 8.0,
				},
			},
			expectedModel: Model{
				Series: forecast.Model{
					Scores: &forecast.Scores{
						MAPE: 0.056,
						MSE:  0.0,
						R2:   1.0,
					},
					Weights: forecast.Weights{
						Coef: []forecast.FeatureWeight{
							{
								Labels: map[string]string{
									"name": "intercept",
								},
								Type:  feature.FeatureTypeGrowth,
								Value: 50.0,
							},
							{
								Labels: map[string]string{
									"name": "weekend",
								},
								Type:  feature.FeatureTypeEvent,
								Value: -20,
							},
							{
								Labels: map[string]string{
									"name":              "epoch_daily",
									"order":             "1",
									"fourier_component": "sin",
								},
								Type:  feature.FeatureTypeSeasonality,
								Value: 15.0,
							},
						},
					},
				},
				Uncertainty: forecast.Model{
					Scores: &forecast.Scores{
						MAPE: 1.0,
						MSE:  0.0,
						R2:   1.0,
					},
					Weights: forecast.Weights{
						Coef: []forecast.FeatureWeight{
							{
								Labels: map[string]string{
									"name": "intercept",
								},
								Type:  feature.FeatureTypeGrowth,
								Value: 0.0,
							},
						},
					},
				},
			},
		},
	}

	for name, td := range testData {
		t.Run(name, func(t *testing.T) {
			defer recoverForecastPanic(t)

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
			compareCoef(t, td.expectedModel.Series.Weights.Coef, m.Series.Weights.Coef, td.tol, "series")

			compareScores(t, td.expectedModel.Uncertainty.Scores, m.Uncertainty.Scores, "uncertainty")
			compareCoef(t, td.expectedModel.Uncertainty.Weights.Coef, m.Uncertainty.Weights.Coef, td.tol, "uncertainty")
		})
	}
}

func generateExampleSeries() ([]time.Time, []float64) {
	// create a daily sine wave at minutely with one week
	minutes := 28 * 24 * 60
	t := timedataset.GenerateT(minutes, time.Minute, time.Now)
	y := make(timedataset.Series, minutes)

	period := 86400.0
	y.Add(timedataset.GenerateConstY(minutes, 98.3)).
		Add(timedataset.GenerateWaveY(t, 10.5, period, 1.0, 2*60*60)).
		Add(timedataset.GenerateWaveY(t, 10.5, period, 3.0, 2.0*60*60+period/2.0/2.0/3.0)).
		Add(timedataset.GenerateWaveY(t, 23.4, period, 7.0, 6.0*60*60+period/2.0/2.0/3.0).MaskWithTimeRange(t[minutes*4/16], t[minutes*5/16], t)).
		Add(timedataset.GenerateWaveY(t, -7.3, period, 3.0, 2*60*60+period/2.0/2.0/3.0).MaskWithWeekend(t)).
		Add(timedataset.GenerateNoise(t, 3.2, 3.2, period, 5.0, 0.0)).
		Add(timedataset.GenerateChange(t, t[minutes/2], 10.0, 0.0)).
		Add(timedataset.GenerateChange(t, t[minutes*2/3], 61.4, 0.0)).             // anomaly start
		Add(timedataset.GenerateChange(t, t[minutes*2/3+minutes/40], -61.4, 0.0)). // anomaly end
		Add(timedataset.GenerateChange(t, t[minutes*17/20], -70.0, 0.0)).
		SetConst(t, 2.7, t[minutes/3], t[minutes/3+minutes/20])

	return t, y
}

func runForecastExample(opt *Options, t []time.Time, y []float64, filename string) error {
	f, err := New(opt)
	if err != nil {
		return err
	}
	if err := f.Fit(t, y); err != nil {
		return err
	}

	m, err := f.Model()
	if err != nil {
		return err
	}
	if err := m.TablePrint(os.Stderr); err != nil {
		return err
	}

	file, err := os.Create(filename)
	if err != nil {
		return err
	}

	return f.PlotFit(file, nil)
}

func recoverForecastPanic(t *testing.T) {
	if r := recover(); r != nil {
		if t != nil {
			t.Errorf("panic: %v\n", r)
		} else {
			fmt.Printf("panic: %v\n", r)
		}
		debug.PrintStack()
	}
}

func setupWithOutliers() ([]time.Time, []float64, *Options) {
	t, y := generateExampleSeries()

	changepoints := []options.Changepoint{
		options.NewChangepoint("anomaly1", t[len(t)/2]),
		options.NewChangepoint("anomaly2", t[len(t)*17/20]),
	}
	events := []options.Event{
		options.NewEvent("custom_event", t[len(t)*4/16], t[len(t)*5/16]),
	}

	regularization := []float64{0.0, 1.0, 10.0, 100.0, 1000.0, 10000.0}
	opt := &Options{
		SeriesOptions: &SeriesOptions{
			ForecastOptions: &options.Options{
				Regularization: regularization,
				SeasonalityOptions: options.SeasonalityOptions{
					SeasonalityConfigs: []options.SeasonalityConfig{
						options.NewDailySeasonalityConfig(12),
						options.NewWeeklySeasonalityConfig(12),
					},
				},
				Iterations: 500,
				Tolerance:  1e-3,
				ChangepointOptions: options.ChangepointOptions{
					Changepoints: changepoints,
				},
				WeekendOptions: options.WeekendOptions{
					Enabled: true,
				},
				EventOptions: options.EventOptions{
					Events: events,
				},
			},
			OutlierOptions: NewOutlierOptions(),
		},
		UncertaintyOptions: &UncertaintyOptions{
			ForecastOptions: &options.Options{
				Regularization: regularization,
				SeasonalityOptions: options.SeasonalityOptions{
					SeasonalityConfigs: []options.SeasonalityConfig{
						options.NewDailySeasonalityConfig(12),
						options.NewWeeklySeasonalityConfig(12),
					},
				},
				Iterations: 250,
				Tolerance:  1e-2,
				ChangepointOptions: options.ChangepointOptions{
					Changepoints: nil,
				},
				WeekendOptions: options.WeekendOptions{
					Enabled: true,
				},
				EventOptions: options.EventOptions{
					Events: events,
				},
			},
			ResidualWindow: 100,
			ResidualZscore: 4.0,
		},
		MinUncertaintyValue: 13.0,
	}

	return t, y, opt
}

func Example_forecasterWithOutliers() {
	t, y, opt := setupWithOutliers()

	defer recoverForecastPanic(nil)

	if err := runForecastExample(opt, t, y, "examples/forecaster.html"); err != nil {
		panic(err)
	}
	// Output:
}

func generateExampleSeriesWithTrend() ([]time.Time, []float64) {
	// create a daily sine wave at minutely with one week
	minutes := 4 * 24 * 60
	t := timedataset.GenerateT(minutes, time.Minute, time.Now)
	y := make(timedataset.Series, minutes)

	period := 86400.0
	y.Add(timedataset.GenerateConstY(minutes, 98.3)).
		Add(timedataset.GenerateWaveY(t, 10.5, period, 1.0, 2*60*60)).
		Add(timedataset.GenerateWaveY(t, 10.5, period, 3.0, 2.0*60*60+period/2.0/2.0/3.0)).
		Add(timedataset.GenerateNoise(t, 3.2, 3.2, period, 5.0, 0.0)).
		Add(timedataset.GenerateChange(t, t[minutes/2], 0.0, 40.0/(t[minutes*17/20].Sub(t[minutes/2]).Minutes()))).
		Add(timedataset.GenerateChange(t, t[minutes*17/20], -40.0, -40.0/(t[minutes*17/20].Sub(t[minutes/2]).Minutes())))

	return t, y
}

func Example_forecasterAutoChangepoint() {
	t, y := generateExampleSeries()

	regularization := []float64{0.0, 1.0, 10.0, 100.0, 1000.0, 10000.0}
	opt := &Options{
		SeriesOptions: &SeriesOptions{
			ForecastOptions: &options.Options{
				Regularization: regularization,
				SeasonalityOptions: options.SeasonalityOptions{
					SeasonalityConfigs: []options.SeasonalityConfig{
						options.NewDailySeasonalityConfig(12),
						options.NewWeeklySeasonalityConfig(12),
					},
				},
				Iterations: 500,
				Tolerance:  1e-3,
				ChangepointOptions: options.ChangepointOptions{
					Auto:                true,
					AutoNumChangepoints: 100,
					EnableGrowth:        false,
				},
			},
			OutlierOptions: NewOutlierOptions(),
		},
		UncertaintyOptions: &UncertaintyOptions{
			ForecastOptions: &options.Options{
				Regularization: regularization,
				SeasonalityOptions: options.SeasonalityOptions{
					SeasonalityConfigs: []options.SeasonalityConfig{
						options.NewDailySeasonalityConfig(12),
						options.NewWeeklySeasonalityConfig(12),
					},
				},
				Iterations: 250,
				Tolerance:  1e-2,
				ChangepointOptions: options.ChangepointOptions{
					Auto:         false,
					Changepoints: []options.Changepoint{},
				},
			},
			ResidualWindow: 100,
			ResidualZscore: 4.0,
		},
	}
	opt.SetMinValue(0.0)
	opt.SetMaxValue(170.0)

	defer recoverForecastPanic(nil)

	if err := runForecastExample(opt, t, y, "examples/forecaster_auto_changepoint.html"); err != nil {
		panic(err)
	}
	// Output:
}

func Example_forecasterWithTrend() {
	t, y := generateExampleSeriesWithTrend()

	changepoints := []options.Changepoint{
		options.NewChangepoint("trendstart", t[len(t)/2]),
		options.NewChangepoint("rebaseline", t[len(t)*17/20]),
	}

	opt := &Options{
		SeriesOptions: &SeriesOptions{
			ForecastOptions: &options.Options{
				SeasonalityOptions: options.SeasonalityOptions{
					SeasonalityConfigs: []options.SeasonalityConfig{
						options.NewDailySeasonalityConfig(12),
						options.NewWeeklySeasonalityConfig(12),
					},
				},
				Iterations: 500,
				Tolerance:  1e-3,
				ChangepointOptions: options.ChangepointOptions{
					Changepoints: changepoints,
					EnableGrowth: true,
				},
			},
			OutlierOptions: NewOutlierOptions(),
		},
		UncertaintyOptions: &UncertaintyOptions{
			ForecastOptions: &options.Options{
				SeasonalityOptions: options.SeasonalityOptions{
					SeasonalityConfigs: []options.SeasonalityConfig{
						options.NewDailySeasonalityConfig(12),
						options.NewWeeklySeasonalityConfig(12),
					},
				},
				Iterations: 250,
				Tolerance:  1e-2,
				ChangepointOptions: options.ChangepointOptions{
					Changepoints: nil,
				},
			},
			ResidualWindow: 100,
			ResidualZscore: 4.0,
		},
	}

	defer recoverForecastPanic(nil)

	if err := runForecastExample(opt, t, y, "examples/forecaster_with_trend.html"); err != nil {
		panic(err)
	}
	// Output:
}

func generateExamplePulseSeries() ([]time.Time, []float64) {
	// create a daily sine wave at minutely with one week
	minutes := 28 * 24 * 60
	t := timedataset.GenerateT(minutes, time.Minute, time.Now)
	y := make(timedataset.Series, minutes)

	period := (3 * time.Hour).Seconds()
	y.Add(timedataset.GenerateConstY(len(t), 30)).
		Add(timedataset.GeneratePulseY(t, 100.0, period, 1.0, 0.0, 0.05)).
		Add(timedataset.GenerateNoise(t, 3.2, 0.0, period, 5.0, 0.0))

	return t, y
}

func setupWithPulses() ([]time.Time, []float64, *Options) {
	t, y := generateExamplePulseSeries()

	changepoints := []options.Changepoint{}
	events := []options.Event{}

	regularization := []float64{0.0, 1.0, 10.0, 100.0, 1000.0, 10000.0}
	opt := &Options{
		SeriesOptions: &SeriesOptions{
			ForecastOptions: &options.Options{
				Regularization: regularization,
				SeasonalityOptions: options.SeasonalityOptions{
					SeasonalityConfigs: []options.SeasonalityConfig{
						options.NewSeasonalityConfig("pulse", 3*time.Hour, 24),
					},
				},
				Iterations: 500,
				Tolerance:  1e-3,
				ChangepointOptions: options.ChangepointOptions{
					Changepoints: changepoints,
				},
				WeekendOptions: options.WeekendOptions{
					Enabled: false,
				},
				EventOptions: options.EventOptions{
					Events: events,
				},
			},
			OutlierOptions: NewOutlierOptions(),
		},
		UncertaintyOptions: &UncertaintyOptions{
			ForecastOptions: &options.Options{
				Regularization: regularization,
				SeasonalityOptions: options.SeasonalityOptions{
					SeasonalityConfigs: []options.SeasonalityConfig{
						options.NewDailySeasonalityConfig(12),
						options.NewWeeklySeasonalityConfig(12),
					},
				},
				Iterations: 250,
				Tolerance:  1e-2,
				ChangepointOptions: options.ChangepointOptions{
					Changepoints: nil,
				},
				WeekendOptions: options.WeekendOptions{
					Enabled: false,
				},
				EventOptions: options.EventOptions{
					Events: events,
				},
			},
			ResidualWindow: 100,
			ResidualZscore: 4.0,
		},
	}

	return t, y, opt
}

func Example_forecasterWithPulses() {
	t, y, opt := setupWithPulses()

	defer recoverForecastPanic(nil)

	if err := runForecastExample(opt, t, y, "examples/forecaster_pulses.html"); err != nil {
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

var benchPredictRes *Results

func BenchmarkTrainToModel(b *testing.B) {
	t, y, opt := setupWithOutliers()

	var f *Forecaster
	var err error

	b.ResetTimer()
	for b.Loop() {
		f, err = New(opt)
		if err != nil {
			panic(err)
		}

		if err := f.Fit(t, y); err != nil {
			panic(err)
		}
	}

	m, err := f.Model()
	if err != nil {
		panic(err)
	}

	bytes, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		panic(err)
	}

	if err := os.WriteFile("benchmark_model.json", bytes, 0644); err != nil {
		panic(err)
	}
}

func BenchmarkPredictFromModel(b *testing.B) {
	bytes, err := os.ReadFile("benchmark_model.json")
	if err != nil {
		panic(err)
	}

	var model Model
	if err := json.Unmarshal(bytes, &model); err != nil {
		panic(err)
	}
	f, err := NewFromModel(model)
	if err != nil {
		panic(err)
	}

	input := make([]time.Time, 0, 2)
	ct := time.Now()
	for i := range cap(input) {
		input = append(input, ct.Add(time.Duration(i)*time.Minute))
	}
	b.ResetTimer()
	defer profile.Start(profile.CPUProfile, profile.ProfilePath(".")).Stop()
	for b.Loop() {
		benchPredictRes, err = f.Predict(input)
		if err != nil {
			panic(err)
		}
	}
}
