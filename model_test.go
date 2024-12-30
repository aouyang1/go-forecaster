package forecaster

import (
	"bytes"
	"testing"
	"time"

	"github.com/aouyang1/go-forecaster/feature"
	"github.com/aouyang1/go-forecaster/forecast"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestModelTablePrint(t *testing.T) {
	testData := map[string]struct {
		m        Model
		prefix   string
		indent   string
		expected string
	}{
		"no input": {
			expected: `Series:
  Forecast:
    Training End Time: 0001-01-01 00:00:00 +0000 UTC
  Weights:
          Type Labels Value
     Intercept        0.000

Uncertainty:
  Forecast:
    Training End Time: 0001-01-01 00:00:00 +0000 UTC
  Weights:
          Type Labels Value
     Intercept        0.000

`,
		},
		"basic input": {
			m: Model{
				Series: forecast.Model{
					TrainEndTime: time.Date(1970, 1, 1, 0, 0, 0, 0, time.UTC),
					Scores: &forecast.Scores{
						MAPE: 0.1234,
						MSE:  1.2345,
						R2:   0.0123,
					},
				},
				Uncertainty: forecast.Model{
					TrainEndTime: time.Date(1970, 1, 1, 0, 0, 0, 0, time.UTC),
					Scores: &forecast.Scores{
						MAPE: 0.2234,
						MSE:  1.3345,
						R2:   0.4123,
					},
				},
			},
			expected: `Series:
  Forecast:
    Training End Time: 1970-01-01 00:00:00 +0000 UTC
  Scores:
    MAPE: 0.123    MSE: 1.234    R2: 0.012
  Weights:
          Type Labels Value
     Intercept        0.000

Uncertainty:
  Forecast:
    Training End Time: 1970-01-01 00:00:00 +0000 UTC
  Scores:
    MAPE: 0.223    MSE: 1.335    R2: 0.412
  Weights:
          Type Labels Value
     Intercept        0.000

`,
		},
		"with options": {
			m: Model{
				Options: &Options{
					SeriesOptions: &SeriesOptions{
						OutlierOptions: &OutlierOptions{
							NumPasses:       3,
							TukeyFactor:     1.0,
							LowerPercentile: 0.1,
							UpperPercentile: 0.9,
						},
					},
					UncertaintyOptions: &UncertaintyOptions{
						ResidualWindow: 100,
						ResidualZscore: 4.0,
					},
				},
				Series: forecast.Model{
					TrainEndTime: time.Date(1970, 1, 3, 0, 0, 0, 0, time.UTC),
					Options: &forecast.Options{
						Regularization: []float64{0.0, 1.0},
						ChangepointOptions: forecast.ChangepointOptions{
							Changepoints: []forecast.Changepoint{
								{Name: "c0", T: time.Date(1970, 1, 2, 0, 0, 0, 0, time.UTC)},
							},
						},
						SeasonalityOptions: forecast.SeasonalityOptions{
							SeasonalityConfigs: []forecast.SeasonalityConfig{
								{Name: "s0", Period: 12 * time.Hour, Orders: 1},
							},
						},
						EventOptions: forecast.EventOptions{
							Events: []forecast.Event{
								{Name: "e0", Start: time.Date(1970, 1, 1, 0, 0, 0, 0, time.UTC), End: time.Date(1970, 1, 2, 0, 0, 0, 0, time.UTC)},
							},
						},
						WeekendOptions: forecast.WeekendOptions{
							Enabled:   true,
							DurBefore: 1 * time.Hour,
							DurAfter:  2 * time.Hour,
						},
					},
					Scores: &forecast.Scores{
						MAPE: 0.1234,
						MSE:  1.2345,
						R2:   0.0123,
					},
					Weights: forecast.Weights{
						Intercept: 1.1,
						Coef: []forecast.FeatureWeight{
							{
								Labels: map[string]string{
									"changepoint_component": "bias",
									"name":                  "c0",
								},
								Type:  feature.FeatureTypeChangepoint,
								Value: 9.8,
							},
							{
								Labels: map[string]string{
									"fourier_component": "sin",
									"name":              "s0",
									"order":             "1",
								},
								Type:  feature.FeatureTypeSeasonality,
								Value: 8.7,
							},
							{
								Labels: map[string]string{
									"name": "e0",
								},
								Type:  feature.FeatureTypeEvent,
								Value: 7.6,
							},
						},
					},
				},
			},
			expected: `Series:
  Options:
    Outlier Options:
      Number of Passes: 3    Tukey Factor: 1.000    Lower Percentile: 10.00%    Upper Percentile: 90.00%
  Forecast:
    Training End Time: 1970-01-03 00:00:00 +0000 UTC
    Regularization: [0.000 1.000]
    Seasonality:
       Name  Period Orders
         s0 12h0m0s      1
    Changepoints:
       Name                      Datetime
         c0 1970-01-02 00:00:00 +0000 UTC
    Weekends:
      Before: -1h0m0s, After: 2h0m0s
    Events:
       Name                         Start                           End
         e0 1970-01-01 00:00:00 +0000 UTC 1970-01-02 00:00:00 +0000 UTC
  Scores:
    MAPE: 0.123    MSE: 1.234    R2: 0.012
  Weights:
            Type                                              Labels Value
       Intercept                                                     1.100
     changepoint        {"changepoint_component":"bias","name":"c0"} 9.800
     seasonality {"fourier_component":"sin","name":"s0","order":"1"} 8.700
           event                                       {"name":"e0"} 7.600

Uncertainty:
  Options:
    Residual Window: 100 samples    Residual Z-Score: 4.000
  Forecast:
    Training End Time: 0001-01-01 00:00:00 +0000 UTC
  Weights:
          Type Labels Value
     Intercept        0.000

`,
		},
		"with disabled options": {
			m: Model{
				Options: &Options{
					SeriesOptions: &SeriesOptions{},
				},
				Series: forecast.Model{
					TrainEndTime: time.Date(1970, 1, 3, 0, 0, 0, 0, time.UTC),
					Options: &forecast.Options{
						Regularization: []float64{0.0, 1.0},
					},
					Scores: &forecast.Scores{
						MAPE: 0.1234,
						MSE:  1.2345,
						R2:   0.0123,
					},
					Weights: forecast.Weights{
						Intercept: 1.1,
					},
				},
			},
			expected: `Series:
  Options:
    Outlier Options: None
  Forecast:
    Training End Time: 1970-01-03 00:00:00 +0000 UTC
    Regularization: [0.000 1.000]
    Seasonality: None
    Changepoints: None
    Weekends: None
    Events: None
  Scores:
    MAPE: 0.123    MSE: 1.234    R2: 0.012
  Weights:
          Type Labels Value
     Intercept        1.100

Uncertainty:
  Options:
  Forecast:
    Training End Time: 0001-01-01 00:00:00 +0000 UTC
  Weights:
          Type Labels Value
     Intercept        0.000

`,
		},
	}

	for name, td := range testData {
		t.Run(name, func(t *testing.T) {
			var buf bytes.Buffer
			err := td.m.TablePrint(&buf)
			require.NoError(t, err)
			assert.Equal(t, td.expected, buf.String())
		})
	}
}
