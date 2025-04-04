package options

import (
	"bytes"
	"testing"
	"time"

	"github.com/aouyang1/go-forecaster/feature"
	"github.com/aouyang1/go-forecaster/timedataset"
	"github.com/stretchr/testify/assert"
)

func TestChangepointTablePrint(t *testing.T) {
	testData := map[string]struct {
		opt          *ChangepointOptions
		prefix       string
		indent       string
		indentGrowth int
		expected     string
	}{
		"no configs": {
			opt: &ChangepointOptions{},
			expected: `Changepoints: None
`,
		},
		"no configs with prefix and indent": {
			opt:          &ChangepointOptions{},
			prefix:       "  ",
			indent:       "--",
			indentGrowth: 1,
			expected: `  --Changepoints: None
`,
		},
		"config with prefix and indent": {
			opt: &ChangepointOptions{
				Changepoints: []Changepoint{
					{Name: "c0", T: time.Date(1970, 1, 2, 0, 0, 0, 0, time.UTC)},
				},
			},
			prefix:       "  ",
			indent:       "  ",
			indentGrowth: 1,
			expected: `    Changepoints:
       Name                      Datetime
         c0 1970-01-02 00:00:00 +0000 UTC
`,
		},
	}

	for name, td := range testData {
		t.Run(name, func(t *testing.T) {
			var buf bytes.Buffer
			td.opt.TablePrint(&buf, td.prefix, td.indent, td.indentGrowth)
			assert.Equal(t, td.expected, buf.String())
		})
	}
}

func TestGenerateAutoChangepoints(t *testing.T) {
	DefaultAutoNumChangepoints = 3
	defer func() {
		DefaultAutoNumChangepoints = 100
	}()

	testData := map[string]struct {
		opt      *ChangepointOptions
		t        []time.Time
		expected *ChangepointOptions
	}{
		"disabled": {
			opt: &ChangepointOptions{},
			t: []time.Time{
				time.Date(1970, 1, 1, 0, 0, 0, 0, time.UTC),
				time.Date(1970, 1, 2, 0, 0, 0, 0, time.UTC),
				time.Date(1970, 1, 3, 0, 0, 0, 0, time.UTC),
				time.Date(1970, 1, 4, 0, 0, 0, 0, time.UTC),
				time.Date(1970, 1, 5, 0, 0, 0, 0, time.UTC),
				time.Date(1970, 1, 6, 0, 0, 0, 0, time.UTC),
				time.Date(1970, 1, 7, 0, 0, 0, 0, time.UTC),
			},
			expected: &ChangepointOptions{},
		},
		"default auto": {
			opt: &ChangepointOptions{
				Auto: true,
			},
			t: []time.Time{
				time.Date(1970, 1, 1, 0, 0, 0, 0, time.UTC),
				time.Date(1970, 1, 2, 0, 0, 0, 0, time.UTC),
				time.Date(1970, 1, 3, 0, 0, 0, 0, time.UTC),
				time.Date(1970, 1, 4, 0, 0, 0, 0, time.UTC),
				time.Date(1970, 1, 5, 0, 0, 0, 0, time.UTC),
				time.Date(1970, 1, 6, 0, 0, 0, 0, time.UTC),
				time.Date(1970, 1, 7, 0, 0, 0, 0, time.UTC),
			},
			expected: &ChangepointOptions{
				Auto: true,
				Changepoints: []Changepoint{
					{Name: "auto_0", T: time.Date(1970, 1, 1, 0, 0, 0, 0, time.UTC)},
					{Name: "auto_1", T: time.Date(1970, 1, 3, 0, 0, 0, 0, time.UTC)},
					{Name: "auto_2", T: time.Date(1970, 1, 5, 0, 0, 0, 0, time.UTC)},
				},
				AutoNumChangepoints: DefaultAutoNumChangepoints,
			},
		},
	}

	for name, td := range testData {
		t.Run(name, func(t *testing.T) {
			td.opt.GenerateAutoChangepoints(td.t)
			assert.Equal(t, td.expected, td.opt)
		})
	}
}

func TestGenerateFeatures(t *testing.T) {
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
