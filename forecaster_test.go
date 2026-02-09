package forecaster

import (
	"fmt"
	"math"
	"testing"
	"time"

	"gonum.org/v1/gonum/mat"

	"github.com/aouyang1/go-forecaster/feature"
	"github.com/aouyang1/go-forecaster/forecast"
	"github.com/aouyang1/go-forecaster/forecast/options"
	"github.com/aouyang1/go-forecaster/forecast/util"
	"github.com/aouyang1/go-forecaster/timedataset"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func compareScores(t *testing.T, expected, actual *forecast.Scores, significantFeatures []forecast.FeatureWeight, msg string) {
	if len(significantFeatures) == 0 {
		return
	}
	if actual.R2 >= 0 && expected.R2 >= 0 {
		assert.InDelta(t, expected.R2, actual.R2, 0.05, msg+" scores:r2")
	}
	assert.InDelta(t, expected.MAPE, actual.MAPE, 0.20, msg+" scores:mape")

	mse := actual.MSE
	if expected.MSE > 0 {
		mse = math.Abs((actual.MSE - expected.MSE) / expected.MSE)
	}
	assert.LessOrEqual(t, mse, 0.20, msg+" scores:mse")

	t.Logf("expected: %.3f, actual: %.3f\n", expected, actual)
}

func compareCoef(t *testing.T, expected, actual []forecast.FeatureWeight, tol float64, msg string) []forecast.FeatureWeight {
	var significantFeatures []forecast.FeatureWeight
	for _, fw := range actual {
		t.Logf("actual %+v weight: %.3f", fw.Labels, fw.Value)
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
	return significantFeatures
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
		"auto remove outliers with tukey method": {
			t: timedataset.GenerateT(4*24*60, time.Minute, nowFunc),
			y: timedataset.GenerateConstY(4*24*60, 50.0).
				Add(timedataset.GenerateWaveY(timedataset.GenerateT(4*24*60, time.Minute, nowFunc), 10.0, 24*60*60, 1.0, 0.0)).
				SetConst(timedataset.GenerateT(4*24*60, time.Minute, nowFunc), 75, nowFunc().Add(-(24*60+10)*time.Minute), nowFunc().Add((-24*60)*time.Minute)),
			tol: 1,
			opt: &Options{
				SeriesOptions: &SeriesOptions{
					ForecastOptions: &options.Options{
						SeasonalityOptions: testDailySeasonalityOptions,
						Regularization:     []float64{0.0},
					},
					OutlierOptions: &OutlierOptions{
						NumPasses:       3,
						UpperPercentile: 0.75,
						LowerPercentile: 0.25,
						TukeyFactor:     1.5,
					},
				},
				UncertaintyOptions: &UncertaintyOptions{
					ForecastOptions: &options.Options{
						SeasonalityOptions: testDailySeasonalityOptions,
						Regularization:     []float64{0.0},
					},
					ResidualWindow: 50,
					ResidualZscore: 8.0,
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
									"name":              "epoch_daily",
									"order":             "1",
									"fourier_component": "sin",
								},
								Type:  feature.FeatureTypeSeasonality,
								Value: 10.0,
							},
						},
					},
				},
				Uncertainty: forecast.Model{
					Scores: &forecast.Scores{
						MAPE: 0.00,
						MSE:  0.00,
						R2:   1.00,
					},
					Weights: forecast.Weights{
						Coef: []forecast.FeatureWeight{},
					},
				},
			},
		},
		"manual remove outliers with events": {
			t: timedataset.GenerateT(4*24*60, time.Minute, nowFunc),
			y: timedataset.GenerateConstY(4*24*60, 50.0).
				Add(timedataset.GenerateWaveY(timedataset.GenerateT(4*24*60, time.Minute, nowFunc), 10.0, 24*60*60, 1.0, 0.0)).
				SetConst(timedataset.GenerateT(4*24*60, time.Minute, nowFunc), 75.0, nowFunc().Add(-(24*60+10)*time.Minute), nowFunc().Add((-24*60)*time.Minute)),
			tol: 2,
			opt: &Options{
				SeriesOptions: &SeriesOptions{
					ForecastOptions: &options.Options{
						SeasonalityOptions: testDailySeasonalityOptions,
						Regularization:     []float64{0.0},
					},
					OutlierOptions: &OutlierOptions{
						NumPasses:       0,
						UpperPercentile: 0.75,
						LowerPercentile: 0.25,
						TukeyFactor:     1.5,
						Events: []options.Event{
							options.NewEvent("outlier_event",
								timedataset.GenerateT(4*24*60, time.Minute, nowFunc)[3*24*60-10],
								timedataset.GenerateT(4*24*60, time.Minute, nowFunc)[3*24*60],
							),
						},
					},
				},
				UncertaintyOptions: &UncertaintyOptions{
					ForecastOptions: &options.Options{
						SeasonalityOptions: testDailySeasonalityOptions,
						Regularization:     []float64{1.0},
					},
					ResidualWindow: 50,
					ResidualZscore: 8.0,
				},
			},
			expectedModel: Model{
				Series: forecast.Model{
					Scores: &forecast.Scores{
						MAPE: 0.00,
						MSE:  0.00,
						R2:   1.00,
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
									"name":              "epoch_daily",
									"order":             "1",
									"fourier_component": "sin",
								},
								Type:  feature.FeatureTypeSeasonality,
								Value: 10.0,
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
						Coef: []forecast.FeatureWeight{},
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

			sigFeats := compareCoef(t, td.expectedModel.Series.Weights.Coef, m.Series.Weights.Coef, td.tol, "series")
			compareScores(t, td.expectedModel.Series.Scores, m.Series.Scores, sigFeats, "series")

			sigFeatsUnc := compareCoef(t, td.expectedModel.Uncertainty.Weights.Coef, m.Uncertainty.Weights.Coef, td.tol, "uncertainty")
			compareScores(t, td.expectedModel.Uncertainty.Scores, m.Uncertainty.Scores, sigFeatsUnc, "uncertainty")
		})
	}
}

func TestFixedPercentageUncertainty(t *testing.T) {
	nowFunc := func() time.Time {
		return time.Date(1970, 2, 1, 0, 0, 0, 0, time.UTC)
	}

	testData := map[string]struct {
		percentage     float64
		seriesValues   []float64
		expectedUpper  []float64
		expectedLower  []float64
		minUncertainty float64
	}{
		"10% of constant series": {
			percentage:     0.1,
			seriesValues:   []float64{100.0, 100.0, 100.0, 100.0, 100.0},
			expectedUpper:  []float64{110.0, 110.0, 110.0, 110.0, 110.0},
			expectedLower:  []float64{90.0, 90.0, 90.0, 90.0, 90.0},
			minUncertainty: 0.0,
		},
		"5% with min uncertainty": {
			percentage:     0.05,
			seriesValues:   []float64{100.0, 100.0, 100.0, 100.0, 100.0},
			expectedUpper:  []float64{106.0, 106.0, 106.0, 106.0, 106.0},
			expectedLower:  []float64{94.0, 94.0, 94.0, 94.0, 94.0},
			minUncertainty: 6.0,
		},
	}

	for name, td := range testData {
		t.Run(name, func(t *testing.T) {
			opt := &Options{
				SeriesOptions: &SeriesOptions{
					ForecastOptions: &options.Options{
						Regularization: []float64{0.0},
					},
				},
				UncertaintyOptions: &UncertaintyOptions{
					Percentage: td.percentage,
				},
				MinUncertaintyValue: td.minUncertainty,
			}

			f, err := New(opt)
			require.Nil(t, err)

			timestamps := timedataset.GenerateT(len(td.seriesValues), time.Minute, nowFunc)
			err = f.Fit(timestamps, td.seriesValues)
			require.Nil(t, err)

			// Test prediction to verify percentage uncertainty
			futureNow := func() time.Time {
				return nowFunc().Add(time.Duration(len(td.seriesValues)) * time.Minute)
			}
			futureTimes := timedataset.GenerateT(5, time.Minute, futureNow)
			results, err := f.Predict(futureTimes)
			require.Nil(t, err)

			assert.InDeltaSlice(t, td.expectedLower, results.Lower, 1e-5, "lower")
			assert.InDeltaSlice(t, td.expectedUpper, results.Upper, 1e-5, "upper")
		})
	}
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
