package options

import (
	"testing"
	"time"

	"github.com/aouyang1/go-forecaster/feature"
	"github.com/aouyang1/go-forecaster/timedataset"
)

func TestGenerateChangepointFeatures(t *testing.T) {
	endTime := time.Date(1970, 1, 8, 0, 0, 0, 0, time.UTC)
	nowFunc := func() time.Time {
		return endTime
	}

	chpntBias8hr := []float64{
		0, 0, 0, 0, // Thursday
		0, 0, 0, 0, // Friday
		0, 0, 0, 0, // Saturday
		0, 0, 0, 0, // Sunday
		0, 0, 0, 0, // Monday
		1, 1, 1, 1, // Tuesday
		1, 1, 1, 1, // Wednesday
	}
	testData := map[string]struct {
		opt             *ChangepointOptions
		trainingEndTime time.Time
		expected        *feature.Set
	}{
		"no changepoints": {
			opt:             new(ChangepointOptions),
			trainingEndTime: endTime,
			expected:        feature.NewSet(),
		},
		"changepoint after training end": {
			opt: &ChangepointOptions{
				Changepoints: []Changepoint{
					{Name: "chpt1", T: endTime.Add(1 * time.Minute)},
				},
			},
			trainingEndTime: endTime,
			expected:        feature.NewSet(),
		},
		"valid single changepoint": {
			opt: &ChangepointOptions{
				Changepoints: []Changepoint{
					{Name: "chpt1", T: endTime.Add(-8 * 6 * time.Hour)},
				},
			},
			trainingEndTime: endTime,
			expected: feature.NewSet().Set(
				feature.NewChangepoint("chpt1", feature.ChangepointCompBias),
				chpntBias8hr,
			),
		},
		"valid single changepoint with growth": {
			opt: &ChangepointOptions{
				Changepoints: []Changepoint{
					{Name: "chpt_with_growth", T: endTime.Add(-8 * 6 * time.Hour)},
				},
				EnableGrowth: true,
			},
			trainingEndTime: endTime,
			expected: feature.NewSet().Set(
				feature.NewChangepoint("chpt_with_growth", feature.ChangepointCompBias),
				chpntBias8hr,
			).Set(
				feature.NewChangepoint("chpt_with_growth", feature.ChangepointCompSlope),
				[]float64{
					0, 0, 0, 0, // Thursday
					0, 0, 0, 0, // Friday
					0, 0, 0, 0, // Saturday
					0, 0, 0, 0, // Sunday
					0, 0, 0, 0, // Monday
					0.000, 0.125, 0.250, 0.375, // Tuesday
					0.500, 0.625, 0.750, 0.875, // Wednesday
				},
			),
		},
		"valid single changepoint with growth and future training date": {
			opt: &ChangepointOptions{
				Changepoints: []Changepoint{
					{Name: "chpt_with_growth_and_future_training", T: endTime.Add(-12 * 6 * time.Hour)},
				},
				EnableGrowth: true,
			},
			trainingEndTime: endTime.Add(8 * 6 * time.Hour),
			expected: feature.NewSet().Set(
				feature.NewChangepoint("chpt_with_growth_and_future_training", feature.ChangepointCompBias),
				[]float64{
					0, 0, 0, 0, // Thursday
					0, 0, 0, 0, // Friday
					0, 0, 0, 0, // Saturday
					0, 0, 0, 0, // Sunday
					1, 1, 1, 1, // Monday
					1, 1, 1, 1, // Tuesday
					1, 1, 1, 1, // Wednesday
				},
			).Set(
				feature.NewChangepoint("chpt_with_growth_and_future_training", feature.ChangepointCompSlope),
				[]float64{
					0, 0, 0, 0, // Thursday
					0, 0, 0, 0, // Friday
					0, 0, 0, 0, // Saturday
					0, 0, 0, 0, // Sunday
					0.00, 0.05, 0.10, 0.15, // Monday
					0.20, 0.25, 0.30, 0.35, // Tuesday
					0.40, 0.45, 0.50, 0.55, // Wednesday
				},
			),
		},
	}

	tSeries := timedataset.GenerateT(4*7, 6*time.Hour, nowFunc)
	for name, td := range testData {
		t.Run(name, func(t *testing.T) {
			res := td.opt.GenerateFeatures(tSeries, td.trainingEndTime)
			compareFeatureSet(t, td.expected, res, 1e-4)
		})
	}
}
