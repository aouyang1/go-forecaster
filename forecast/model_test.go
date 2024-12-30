package forecast

import (
	"bytes"
	"testing"
	"time"

	"github.com/aouyang1/go-forecaster/feature"
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
			expected: `Forecast:
Training End Time: 0001-01-01 00:00:00 +0000 UTC
Weights:
      Type Labels Value
 Intercept        0.000
`,
		},
		"basic input with prefix and indent": {
			m: Model{
				TrainEndTime: time.Date(1970, 1, 1, 0, 0, 0, 0, time.UTC),
				Scores: &Scores{
					MAPE: 0.1234,
					MSE:  1.2345,
					R2:   0.0123,
				},
			},
			prefix: "--",
			indent: "**",
			expected: `--Forecast:
--**Training End Time: 1970-01-01 00:00:00 +0000 UTC
--Scores:
--**MAPE: 0.123    MSE: 1.234    R2: 0.012
--Weights:
      --**Type Labels Value
 --**Intercept        0.000
`,
		},
		"with all options": {
			m: Model{
				TrainEndTime: time.Date(1970, 1, 3, 0, 0, 0, 0, time.UTC),
				Options: &Options{
					Regularization: []float64{0.0, 1.0},
					ChangepointOptions: ChangepointOptions{
						Changepoints: []Changepoint{
							{Name: "c0", T: time.Date(1970, 1, 2, 0, 0, 0, 0, time.UTC)},
						},
					},
					SeasonalityOptions: SeasonalityOptions{
						SeasonalityConfigs: []SeasonalityConfig{
							{Name: "s0", Period: 12 * time.Hour, Orders: 1},
						},
					},
					EventOptions: EventOptions{
						Events: []Event{
							{Name: "e0", Start: time.Date(1970, 1, 1, 0, 0, 0, 0, time.UTC), End: time.Date(1970, 1, 2, 0, 0, 0, 0, time.UTC)},
						},
					},
					WeekendOptions: WeekendOptions{
						Enabled:   true,
						DurBefore: 1 * time.Hour,
						DurAfter:  2 * time.Hour,
					},
				},
				Scores: &Scores{
					MAPE: 0.1234,
					MSE:  1.2345,
					R2:   0.0123,
				},
				Weights: Weights{
					Intercept: 1.1,
					Coef: []FeatureWeight{
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
			prefix: "  ",
			indent: "  ",
			expected: `  Forecast:
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
`,
		},
		"with disabled options": {
			m: Model{
				TrainEndTime: time.Date(1970, 1, 3, 0, 0, 0, 0, time.UTC),
				Options: &Options{
					Regularization: []float64{0.0, 1.0},
				},
				Scores: &Scores{
					MAPE: 0.1234,
					MSE:  1.2345,
					R2:   0.0123,
				},
				Weights: Weights{
					Intercept: 1.1,
				},
			},
			prefix: "  ",
			indent: "  ",
			expected: `  Forecast:
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
`,
		},
	}

	for name, td := range testData {
		t.Run(name, func(t *testing.T) {
			var buf bytes.Buffer
			err := td.m.TablePrint(&buf, td.prefix, td.indent)
			require.NoError(t, err)
			assert.Equal(t, td.expected, buf.String())
		})
	}
}
