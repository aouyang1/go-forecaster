package options

import (
	"fmt"
	"testing"
	"time"

	"github.com/aouyang1/go-forecaster/feature"
	"github.com/aouyang1/go-forecaster/linearmodel"
	"github.com/aouyang1/go-forecaster/timedataset"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gonum.org/v1/gonum/floats"
)

func compareFeatureSet(t *testing.T, expected, res *feature.Set, tol float64) {
	assert.Equal(t, expected.Len(), res.Len())
	require.Equal(t, expected.Labels(), res.Labels())

	for _, f := range res.Labels() {
		expVals, exists := expected.Get(f)
		require.True(t, exists)
		gotVals, exists := res.Get(f)
		require.True(t, exists)
		require.Equal(t, len(expVals), len(gotVals))
		assert.InDeltaSlice(t, expVals, gotVals, tol, fmt.Sprintf("feature: %+v, values: %+v\n", f, gotVals))
	}
}

func TestNewLassoAutoAptions(t *testing.T) {
	testData := map[string]struct {
		opt      *Options
		expected *linearmodel.LassoAutoOptions
	}{
		"defaults": {
			opt: &Options{},
			expected: &linearmodel.LassoAutoOptions{
				Lambdas:         []float64{1.0},
				FitIntercept:    false,
				Iterations:      linearmodel.DefaultIterations,
				Tolerance:       linearmodel.DefaultTolerance,
				Parallelization: 0,
			},
		},
		"with overrides": {
			opt: &Options{
				Regularization:  []float64{0.0, 1.0},
				Iterations:      3,
				Tolerance:       1e-1,
				Parallelization: 2,
			},
			expected: &linearmodel.LassoAutoOptions{
				Lambdas:         []float64{0.0, 1.0},
				FitIntercept:    false,
				Iterations:      3,
				Tolerance:       1e-1,
				Parallelization: 2,
			},
		},
	}

	for name, td := range testData {
		t.Run(name, func(t *testing.T) {
			res := td.opt.NewLassoAutoOptions()
			assert.Equal(t, td.expected, res)
		})
	}
}

func TestGenerateTimeFeatures(t *testing.T) {
	ones7DaysAt6Hr := make([]float64, 28)
	floats.AddConst(1.0, ones7DaysAt6Hr)

	ones7DaysAt8Hr := make([]float64, 21)
	floats.AddConst(1.0, ones7DaysAt8Hr)

	epoch7DaysAt6Hr := []float64{
		0 * 3600.0, 6 * 3600.0, 12 * 3600.0, 18 * 3600.0, // Thursday
		24 * 3600, 30 * 3600.0, 36 * 3600.0, 42 * 3600.0, // Friday
		48 * 3600, 54 * 3600.0, 60 * 3600.0, 66 * 3600.0, // Saturday
		72 * 3600, 78 * 3600.0, 84 * 3600.0, 90 * 3600.0, // Sunday
		96 * 3600, 102 * 3600.0, 108 * 3600.0, 114 * 3600.0, // Monday
		120 * 3600, 126 * 3600.0, 132 * 3600.0, 138 * 3600.0, // Tuesday
		144 * 3600, 150 * 3600.0, 156 * 3600.0, 162 * 3600.0, // Wednesday
	}
	epoch7DaysAt8Hr := []float64{
		0 * 3600.0, 8 * 3600.0, 16 * 3600.0, // Thursday
		24 * 3600, 32 * 3600.0, 40 * 3600.0, // Friday
		48 * 3600, 56 * 3600.0, 64 * 3600.0, // Saturday
		72 * 3600, 80 * 3600.0, 88 * 3600.0, // Sunday
		96 * 3600, 104 * 3600.0, 112 * 3600.0, // Monday
		120 * 3600, 128 * 3600.0, 136 * 3600.0, // Tuesday
		144 * 3600, 152 * 3600.0, 160 * 3600.0, // Wednesday
	}

	nowFunc := func() time.Time {
		return time.Date(1970, 1, 8, 0, 0, 0, 0, time.UTC)
	}
	testData := map[string]struct {
		t        []time.Time
		opt      *Options
		expected *feature.Set
	}{
		"empty options": {
			t:   timedataset.GenerateT(4*7, 6*time.Hour, nowFunc),
			opt: &Options{},
			expected: feature.NewSet().Set(
				feature.Intercept(),
				ones7DaysAt6Hr,
			).Set(
				feature.NewTime("epoch"),
				epoch7DaysAt6Hr,
			),
		},
		"basic weekend": {
			t: timedataset.GenerateT(4*7, 6*time.Hour, nowFunc),
			opt: &Options{
				WeekendOptions: WeekendOptions{
					Enabled: true,
				},
			},
			expected: feature.NewSet().Set(
				feature.Intercept(),
				ones7DaysAt6Hr,
			).Set(
				feature.NewTime("epoch"),
				epoch7DaysAt6Hr,
			).Set(
				feature.NewEvent("weekend"),
				[]float64{
					0, 0, 0, 0, // Thursday
					0, 0, 0, 0, // Friday
					1, 1, 1, 1, // Saturday
					1, 1, 1, 1, // Sunday
					0, 0, 0, 0, // Monday
					0, 0, 0, 0, // Tuesday
					0, 0, 0, 0, // Wednesday
				},
			),
		},
		"weekend with buffers": {
			t: timedataset.GenerateT(4*7, 6*time.Hour, nowFunc),
			opt: &Options{
				WeekendOptions: WeekendOptions{
					Enabled:   true,
					DurBefore: 6 * time.Hour,
					DurAfter:  12 * time.Hour,
				},
			},
			expected: feature.NewSet().Set(
				feature.Intercept(),
				ones7DaysAt6Hr,
			).Set(
				feature.NewTime("epoch"),
				epoch7DaysAt6Hr,
			).Set(
				feature.NewEvent("weekend"),
				[]float64{
					0, 0, 0, 0, // Thursday
					0, 0, 0, 1, // Friday
					1, 1, 1, 1, // Saturday
					1, 1, 1, 1, // Sunday
					1, 1, 0, 0, // Monday
					0, 0, 0, 0, // Tuesday
					0, 0, 0, 0, // Wednesday
				},
			),
		},
		"weekend with max extension buffers": {
			t: timedataset.GenerateT(4*7, 6*time.Hour, nowFunc),
			opt: &Options{
				WeekendOptions: WeekendOptions{
					Enabled:   true,
					DurBefore: 30 * time.Hour,
					DurAfter:  30 * time.Hour,
				},
			},
			expected: feature.NewSet().Set(
				feature.Intercept(),
				ones7DaysAt6Hr,
			).Set(
				feature.NewTime("epoch"),
				epoch7DaysAt6Hr,
			).Set(
				feature.NewEvent("weekend"),
				[]float64{
					0, 0, 0, 0, // Thursday
					1, 1, 1, 1, // Friday
					1, 1, 1, 1, // Saturday
					1, 1, 1, 1, // Sunday
					1, 1, 1, 1, // Monday
					0, 0, 0, 0, // Tuesday
					0, 0, 0, 0, // Wednesday
				},
			),
		},
		"weekend with max shrink buffers": {
			t: timedataset.GenerateT(4*7, 6*time.Hour, nowFunc),
			opt: &Options{
				WeekendOptions: WeekendOptions{
					Enabled:   true,
					DurBefore: -30 * time.Hour,
					DurAfter:  -30 * time.Hour,
				},
			},
			expected: feature.NewSet().Set(
				feature.Intercept(),
				ones7DaysAt6Hr,
			).Set(
				feature.NewTime("epoch"),
				epoch7DaysAt6Hr,
			).Set(
				feature.NewEvent("weekend"),
				[]float64{
					0, 0, 0, 0, // Thursday
					0, 0, 0, 0, // Friday
					0, 0, 0, 0, // Saturday
					0, 0, 0, 0, // Sunday
					0, 0, 0, 0, // Monday
					0, 0, 0, 0, // Tuesday
					0, 0, 0, 0, // Wednesday
				},
			),
		},
		"weekend with shrink buffers": {
			t: timedataset.GenerateT(4*7, 6*time.Hour, nowFunc),
			opt: &Options{
				WeekendOptions: WeekendOptions{
					Enabled:   true,
					DurBefore: -6 * time.Hour,
					DurAfter:  -6 * time.Hour,
				},
			},
			expected: feature.NewSet().Set(
				feature.Intercept(),
				ones7DaysAt6Hr,
			).Set(
				feature.NewTime("epoch"),
				epoch7DaysAt6Hr,
			).Set(
				feature.NewEvent("weekend"),
				[]float64{
					0, 0, 0, 0, // Thursday
					0, 0, 0, 0, // Friday
					0, 1, 1, 1, // Saturday
					1, 1, 1, 0, // Sunday
					0, 0, 0, 0, // Monday
					0, 0, 0, 0, // Tuesday
					0, 0, 0, 0, // Wednesday
				},
			),
		},

		"weekend with tz override": {
			t: timedataset.GenerateT(3*7, 8*time.Hour, nowFunc),
			opt: &Options{
				WeekendOptions: WeekendOptions{
					Enabled:          true,
					TimezoneOverride: "America/Los_Angeles",
				},
			},
			expected: feature.NewSet().Set(
				feature.Intercept(),
				ones7DaysAt8Hr,
			).Set(
				feature.NewTime("epoch"),
				epoch7DaysAt8Hr,
			).Set(
				feature.NewEvent("weekend"),
				[]float64{
					0, 0, 0, // Thursday
					0, 0, 0, // Friday
					0, 1, 1, // Saturday
					1, 1, 1, // Sunday
					1, 0, 0, // Monday
					0, 0, 0, // Tuesday
					0, 0, 0, // Wednesday
				},
			),
		},
		"weekend with invalid tz override": {
			t: timedataset.GenerateT(3*7, 8*time.Hour, nowFunc),
			opt: &Options{
				WeekendOptions: WeekendOptions{
					Enabled:          true,
					TimezoneOverride: "bogus",
				},
			},
			expected: feature.NewSet().Set(
				feature.Intercept(),
				ones7DaysAt8Hr,
			).Set(
				feature.NewTime("epoch"),
				epoch7DaysAt8Hr,
			).Set(
				feature.NewEvent("weekend"),
				[]float64{
					0, 0, 0, // Thursday
					0, 0, 0, // Friday
					1, 1, 1, // Saturday
					1, 1, 1, // Sunday
					0, 0, 0, // Monday
					0, 0, 0, // Tuesday
					0, 0, 0, // Wednesday
				},
			),
		},
		"basic weekend with hamm window": {
			t: timedataset.GenerateT(4*7, 6*time.Hour, nowFunc),
			opt: &Options{
				MaskWindow: "hann",
				WeekendOptions: WeekendOptions{
					Enabled: true,
				},
			},
			expected: feature.NewSet().Set(
				feature.Intercept(),
				ones7DaysAt6Hr,
			).Set(
				feature.NewTime("epoch"),
				epoch7DaysAt6Hr,
			).Set(
				feature.NewEvent("weekend"),
				[]float64{
					0, 0, 0, 0, // Thursday
					0, 0, 0, 0, // Friday
					0, 0.1882, 0.6112, 0.9504, // Saturday
					0.9504, 0.6112, 0.1882, 0, // Sunday
					0, 0, 0, 0, // Monday
					0, 0, 0, 0, // Tuesday
					0, 0, 0, 0, // Wednesday
				},
			),
		},
		"weekend with hamm window and overlap ends": {
			t: timedataset.GenerateT(4*7, 6*time.Hour,
				func() time.Time {
					return time.Date(1970, 1, 11, 0, 0, 0, 0, time.UTC)
				},
			),
			opt: &Options{
				MaskWindow: "hann",
				WeekendOptions: WeekendOptions{
					Enabled: true,
				},
			},
			expected: feature.NewSet().Set(
				feature.Intercept(),
				ones7DaysAt6Hr,
			).Set(
				feature.NewTime("epoch"),
				[]float64{
					72 * 3600, 78 * 3600, 84 * 3600, 90 * 3600, // Sunday
					96 * 3600, 102 * 3600, 108 * 3600, 114 * 3600, // Monday
					120 * 3600, 126 * 3600, 132 * 3600, 138 * 3600, // Tuesday
					144 * 3600, 150 * 3600, 156 * 3600, 162 * 3600, // Wednesday
					168 * 3600, 174 * 3600, 180 * 3600, 186 * 3600, // Thursday
					192 * 3600, 198 * 3600, 204 * 3600, 210 * 3600, // Friday
					216 * 3600, 222 * 3600, 228 * 3600, 234 * 3600, // Saturday
				},
			).Set(
				feature.NewEvent("weekend"),
				[]float64{
					0.9504, 0.6112, 0.1882, 0, // Sunday
					0, 0, 0, 0, // Monday
					0, 0, 0, 0, // Tuesday
					0, 0, 0, 0, // Wednesday
					0, 0, 0, 0, // Thursday
					0, 0, 0, 0, // Friday
					0, 0.1882, 0.6112, 0.9504, // Saturday
				},
			),
		},
		"basic event": {
			t: timedataset.GenerateT(4*7, 6*time.Hour, nowFunc),
			opt: &Options{
				EventOptions: EventOptions{
					Events: []Event{
						{
							Name:  "myevent",
							Start: time.Date(1970, 1, 1, 6, 0, 0, 0, time.UTC),
							End:   time.Date(1970, 1, 1, 18, 0, 0, 0, time.UTC),
						},
						{
							Name:  "my other event",
							Start: time.Date(1970, 1, 7, 6, 0, 0, 0, time.UTC),
							End:   time.Date(1970, 1, 7, 18, 0, 0, 0, time.UTC),
						},
					},
				},
			},
			expected: feature.NewSet().Set(
				feature.Intercept(),
				ones7DaysAt6Hr,
			).Set(
				feature.NewTime("epoch"),
				epoch7DaysAt6Hr,
			).Set(
				feature.NewEvent("myevent"),
				[]float64{
					0, 1, 1, 0, // Thursday
					0, 0, 0, 0, // Friday
					0, 0, 0, 0, // Saturday
					0, 0, 0, 0, // Sunday
					0, 0, 0, 0, // Monday
					0, 0, 0, 0, // Tuesday
					0, 0, 0, 0, // Wednesday
				},
			).Set(
				feature.NewEvent("my_other_event"),
				[]float64{
					0, 0, 0, 0, // Thursday
					0, 0, 0, 0, // Friday
					0, 0, 0, 0, // Saturday
					0, 0, 0, 0, // Sunday
					0, 0, 0, 0, // Monday
					0, 0, 0, 0, // Tuesday
					0, 1, 1, 0, // Wednesday
				},
			),
		},
		"event extending out of bounds": {
			t: timedataset.GenerateT(4*7, 6*time.Hour, nowFunc),
			opt: &Options{
				EventOptions: EventOptions{
					Events: []Event{
						{
							Name:  "overlaps_start",
							Start: time.Date(1969, 12, 30, 0, 0, 0, 0, time.UTC),
							End:   time.Date(1970, 1, 1, 18, 0, 0, 0, time.UTC),
						},
						{
							Name:  "overlaps_end",
							Start: time.Date(1970, 1, 7, 6, 0, 0, 0, time.UTC),
							End:   time.Date(1970, 1, 9, 0, 0, 0, 0, time.UTC),
						},
					},
				},
			},
			expected: feature.NewSet().Set(
				feature.Intercept(),
				ones7DaysAt6Hr,
			).Set(
				feature.NewTime("epoch"),
				epoch7DaysAt6Hr,
			).Set(
				feature.NewEvent("overlaps_start"),
				[]float64{
					1, 1, 1, 0, // Thursday
					0, 0, 0, 0, // Friday
					0, 0, 0, 0, // Saturday
					0, 0, 0, 0, // Sunday
					0, 0, 0, 0, // Monday
					0, 0, 0, 0, // Tuesday
					0, 0, 0, 0, // Wednesday
				},
			).Set(
				feature.NewEvent("overlaps_end"),
				[]float64{
					0, 0, 0, 0, // Thursday
					0, 0, 0, 0, // Friday
					0, 0, 0, 0, // Saturday
					0, 0, 0, 0, // Sunday
					0, 0, 0, 0, // Monday
					0, 0, 0, 0, // Tuesday
					0, 1, 1, 1, // Wednesday
				},
			),
		},
		"duplicate event": {
			t: timedataset.GenerateT(4*7, 6*time.Hour, nowFunc),
			opt: &Options{
				EventOptions: EventOptions{
					Events: []Event{
						{
							Name:  "duplicate",
							Start: time.Date(1970, 1, 3, 6, 0, 0, 0, time.UTC),
							End:   time.Date(1970, 1, 3, 18, 0, 0, 0, time.UTC),
						},
						{
							Name:  "duplicate",
							Start: time.Date(1970, 1, 2, 6, 0, 0, 0, time.UTC),
							End:   time.Date(1970, 1, 2, 18, 0, 0, 0, time.UTC),
						},
					},
				},
			},
			expected: feature.NewSet().Set(
				feature.Intercept(),
				ones7DaysAt6Hr,
			).Set(
				feature.NewTime("epoch"),
				epoch7DaysAt6Hr,
			).Set(
				feature.NewEvent("duplicate"),
				[]float64{
					0, 0, 0, 0, // Thursday
					0, 0, 0, 0, // Friday
					0, 1, 1, 0, // Saturday
					0, 0, 0, 0, // Sunday
					0, 0, 0, 0, // Monday
					0, 0, 0, 0, // Tuesday
					0, 0, 0, 0, // Wednesday
				},
			),
		},
		"invalid events": {
			t: timedataset.GenerateT(4*7, 6*time.Hour, nowFunc),
			opt: &Options{
				EventOptions: EventOptions{
					Events: []Event{
						{
							Name: "start is zero",
							End:  time.Date(1970, 1, 7, 12, 0, 0, 0, time.UTC),
						},
						{
							Name:  "end is zero",
							Start: time.Date(1970, 1, 1, 12, 0, 0, 0, time.UTC),
						},
						{
							Start: time.Date(1970, 1, 1, 12, 0, 0, 0, time.UTC),
							End:   time.Date(1970, 1, 7, 12, 0, 0, 0, time.UTC),
						},
						{
							Name:  "start after end",
							Start: time.Date(1970, 1, 7, 12, 0, 0, 0, time.UTC),
							End:   time.Date(1970, 1, 1, 12, 0, 0, 0, time.UTC),
						},
					},
				},
			},
			expected: feature.NewSet().Set(
				feature.Intercept(),
				ones7DaysAt6Hr,
			).Set(
				feature.NewTime("epoch"),
				epoch7DaysAt6Hr,
			),
		},
		"basic event with hamm window": {
			t: timedataset.GenerateT(4*7, 6*time.Hour, nowFunc),
			opt: &Options{
				MaskWindow: "hann",
				EventOptions: EventOptions{
					Events: []Event{
						{
							Name:  "myevent",
							Start: time.Date(1970, 1, 2, 6, 0, 0, 0, time.UTC),
							End:   time.Date(1970, 1, 4, 6, 0, 0, 0, time.UTC),
						},
					},
				},
			},
			expected: feature.NewSet().Set(
				feature.Intercept(),
				ones7DaysAt6Hr,
			).Set(
				feature.NewTime("epoch"),
				epoch7DaysAt6Hr,
			).Set(
				feature.NewEvent("myevent"),
				[]float64{
					0, 0, 0, 0, // Thursday
					0.0000, 0.0000, 0.1882, 0.6112, // Friday
					0.9504, 0.9504, 0.6112, 0.1882, // Saturday
					0, 0, 0, 0, // Sunday
					0, 0, 0, 0, // Monday
					0, 0, 0, 0, // Tuesday
					0, 0, 0, 0, // Wednesday
				},
			),
		},
		"event with hamm window overlapping start and end": {
			t: timedataset.GenerateT(4*7, 6*time.Hour, nowFunc),
			opt: &Options{
				MaskWindow: "hann",
				EventOptions: EventOptions{
					Events: []Event{
						{
							Name:  "hamm_before",
							Start: time.Date(1969, 12, 31, 0, 0, 0, 0, time.UTC),
							End:   time.Date(1970, 1, 2, 0, 0, 0, 0, time.UTC),
						},
						{
							Name:  "hamm_after",
							Start: time.Date(1970, 1, 7, 0, 0, 0, 0, time.UTC),
							End:   time.Date(1970, 1, 9, 0, 0, 0, 0, time.UTC),
						},
					},
				},
			},
			expected: feature.NewSet().Set(
				feature.Intercept(),
				ones7DaysAt6Hr,
			).Set(
				feature.NewTime("epoch"),
				epoch7DaysAt6Hr,
			).Set(
				feature.NewEvent("hamm_before"),
				[]float64{
					0.9504, 0.6112, 0.1882, 0.0000, // Thursday
					0, 0, 0, 0, // Friday
					0, 0, 0, 0, // Saturday
					0, 0, 0, 0, // Sunday
					0, 0, 0, 0, // Monday
					0, 0, 0, 0, // Tuesday
					0, 0, 0, 0, // Wednesday
				},
			).Set(
				feature.NewEvent("hamm_after"),
				[]float64{
					0, 0, 0, 0, // Thursday
					0, 0, 0, 0, // Friday
					0, 0, 0, 0, // Saturday
					0, 0, 0, 0, // Sunday
					0, 0, 0, 0, // Monday
					0, 0, 0, 0, // Tuesday
					0.0000, 0.1882, 0.6112, 0.9504, // Wednesday
				},
			),
		},
		"default": {
			t:   timedataset.GenerateT(4*7, 6*time.Hour, nowFunc),
			opt: nil,
			expected: feature.NewSet().Set(
				feature.Intercept(),
				ones7DaysAt6Hr,
			).Set(
				feature.NewTime("epoch"),
				epoch7DaysAt6Hr,
			),
		},
	}

	for name, td := range testData {
		t.Run(name, func(t *testing.T) {
			startTrainTime := timedataset.TimeSlice(td.t).StartTime()
			endTrainTime := timedataset.TimeSlice(td.t).EndTime()
			features, _ := td.opt.GenerateTimeFeatures(td.t, startTrainTime, endTrainTime)
			compareFeatureSet(t, td.expected, features, 1e-4)
		})
	}
}

func TestGenerateFourierFeatures(t *testing.T) {
	nowFunc := func() time.Time {
		return time.Date(1970, 1, 8, 0, 0, 0, 0, time.UTC)
	}

	dailySin1 := []float64{
		0, 1, 0, -1, // Thursday
		0, 1, 0, -1, // Friday
		0, 1, 0, -1, // Saturday
		0, 1, 0, -1, // Sunday
		0, 1, 0, -1, // Monday
		0, 1, 0, -1, // Tuesday
		0, 1, 0, -1, // Wednesday
	}
	dailyCos1 := []float64{
		1, 0, -1, 0, // Thursday
		1, 0, -1, 0, // Friday
		1, 0, -1, 0, // Saturday
		1, 0, -1, 0, // Sunday
		1, 0, -1, 0, // Monday
		1, 0, -1, 0, // Tuesday
		1, 0, -1, 0, // Wednesday
	}
	dailySin2 := []float64{
		0, 0, 0, 0, // Thursday
		0, 0, 0, 0, // Friday
		0, 0, 0, 0, // Saturday
		0, 0, 0, 0, // Sunday
		0, 0, 0, 0, // Monday
		0, 0, 0, 0, // Tuesday
		0, 0, 0, 0, // Wednesday
	}
	dailyCos2 := []float64{
		1, -1, 1, -1, // Thursday
		1, -1, 1, -1, // Friday
		1, -1, 1, -1, // Saturday
		1, -1, 1, -1, // Sunday
		1, -1, 1, -1, // Monday
		1, -1, 1, -1, // Tuesday
		1, -1, 1, -1, // Wednesday
	}
	weeklySin1 := []float64{
		+0.0000, +0.2225, +0.4338, +0.6234, // Thursday
		+0.7818, +0.9009, +0.9749, +1.0000, // Friday
		+0.9749, +0.9009, +0.7818, +0.6234, // Saturday
		+0.4338, +0.2225, +0.0000, -0.2225, // Sunday
		-0.4338, -0.6234, -0.7818, -0.9009, // Monday
		-0.9749, -1.0000, -0.9749, -0.9009, // Tuesday
		-0.7818, -0.6234, -0.4338, -0.2225, // Wednesday
	}
	weeklyCos1 := []float64{
		+1.0000, +0.9749, +0.9009, +0.7818, // Thursday
		+0.6234, +0.4338, +0.2225, +0.0000, // Friday
		-0.2225, -0.4338, -0.6234, -0.7818, // Saturday
		-0.9009, -0.9749, -1.0000, -0.9749, // Sunday
		-0.9009, -0.7818, -0.6234, -0.4338, // Monday
		-0.2225, +0.0000, +0.2225, +0.4338, // Tuesday
		+0.6234, +0.7818, +0.9009, +0.9749, // Wednesday
	}
	weekendSin1 := []float64{
		0, 0, 0, 0, // Thursday
		0, 0, 0, 0, // Friday
		0, 1, 0, -1, // Saturday
		0, 1, 0, -1, // Sunday
		0, 0, 0, 0, // Monday
		0, 0, 0, 0, // Tuesday
		0, 0, 0, 0, // Wednesday
	}
	weekendCos1 := []float64{
		0, 0, 0, 0, // Thursday
		0, 0, 0, 0, // Friday
		1, 0, -1, 0, // Saturday
		1, 0, -1, 0, // Sunday
		0, 0, 0, 0, // Monday
		0, 0, 0, 0, // Tuesday
		0, 0, 0, 0, // Wednesday
	}

	testData := map[string]struct {
		opt      *Options
		trained  bool
		expected *feature.Set
		err      error
	}{
		"no seasonality": {
			opt:      &Options{},
			expected: feature.NewSet(),
			err:      nil,
		},
		"daily seasonality": {
			opt: &Options{
				SeasonalityOptions: SeasonalityOptions{
					SeasonalityConfigs: []SeasonalityConfig{
						NewDailySeasonalityConfig(2),
					},
				},
			},
			expected: feature.NewSet().Set(
				feature.NewSeasonality("epoch_daily", feature.FourierCompSin, 1),
				dailySin1,
			).Set(
				feature.NewSeasonality("epoch_daily", feature.FourierCompCos, 1),
				dailyCos1,
			).Set(
				feature.NewSeasonality("epoch_daily", feature.FourierCompSin, 2),
				dailySin2,
			).Set(
				feature.NewSeasonality("epoch_daily", feature.FourierCompCos, 2),
				dailyCos2,
			),
			err: nil,
		},
		"weekly seasonality": {
			opt: &Options{
				SeasonalityOptions: SeasonalityOptions{
					SeasonalityConfigs: []SeasonalityConfig{
						NewWeeklySeasonalityConfig(2),
					},
				},
			},
			expected: feature.NewSet().Set(
				feature.NewSeasonality("epoch_weekly", feature.FourierCompSin, 1),
				weeklySin1,
			).Set(
				feature.NewSeasonality("epoch_weekly", feature.FourierCompCos, 1),
				weeklyCos1,
			).Set(
				feature.NewSeasonality("epoch_weekly", feature.FourierCompSin, 2),
				[]float64{
					+0.0000, +0.4338, +0.7818, +0.9749, // Thursday
					+0.9749, +0.7818, +0.4338, +0.0000, // Friday
					-0.4338, -0.7818, -0.9749, -0.9749, // Saturday
					-0.7818, -0.4338, +0.0000, +0.4338, // Sunday
					+0.7818, +0.9749, +0.9749, +0.7818, // Monday
					+0.4338, +0.0000, -0.4338, -0.7818, // Tuesday
					-0.9749, -0.9749, -0.7818, -0.4338, // Wednesday
				},
			).Set(
				feature.NewSeasonality("epoch_weekly", feature.FourierCompCos, 2),
				[]float64{
					+1.0000, +0.9009, +0.6234, +0.2225, // Thursday
					-0.2225, -0.6234, -0.9009, -1.0000, // Friday
					-0.9009, -0.6234, -0.2225, +0.2225, // Saturday
					+0.6234, +0.9009, +1.000, +0.9009, // Sunday
					+0.6234, +0.2225, -0.2225, -0.6234, // Monday
					-0.9009, -1.0000, -0.9009, -0.6234, // Tuesday
					-0.2225, +0.2225, +0.6234, +0.9009, // Wednesday
				},
			),
			err: nil,
		},
		"duplicate seasonality": {
			opt: &Options{
				SeasonalityOptions: SeasonalityOptions{
					SeasonalityConfigs: []SeasonalityConfig{
						NewDailySeasonalityConfig(1),
						NewDailySeasonalityConfig(1),
					},
				},
			},
			expected: feature.NewSet().Set(
				feature.NewSeasonality("epoch_daily", feature.FourierCompSin, 1),
				dailySin1,
			).Set(
				feature.NewSeasonality("epoch_daily", feature.FourierCompCos, 1),
				dailyCos1,
			),
			err: nil,
		},

		"daily seasonality with weekend enabled": {
			opt: &Options{
				SeasonalityOptions: SeasonalityOptions{
					SeasonalityConfigs: []SeasonalityConfig{
						NewDailySeasonalityConfig(1),
					},
				},
				WeekendOptions: WeekendOptions{
					Enabled: true,
				},
			},
			expected: feature.NewSet().Set(
				feature.NewSeasonality("epoch_daily", feature.FourierCompSin, 1),
				dailySin1,
			).Set(
				feature.NewSeasonality("epoch_daily", feature.FourierCompCos, 1),
				dailyCos1,
			).Set(
				feature.NewSeasonality("weekend_daily", feature.FourierCompSin, 1),
				weekendSin1,
			).Set(
				feature.NewSeasonality("weekend_daily", feature.FourierCompCos, 1),
				weekendCos1,
			),
			err: nil,
		},
		"daily seasonality with event": {
			opt: &Options{
				SeasonalityOptions: SeasonalityOptions{
					SeasonalityConfigs: []SeasonalityConfig{
						NewDailySeasonalityConfig(1),
					},
				},
				EventOptions: EventOptions{
					Events: []Event{
						{
							Name:  "myevent",
							Start: time.Date(1970, 1, 3, 12, 0, 0, 0, time.UTC),
							End:   time.Date(1970, 1, 5, 12, 0, 0, 0, time.UTC),
						},
					},
				},
			},
			expected: feature.NewSet().Set(
				feature.NewSeasonality("epoch_daily", feature.FourierCompSin, 1),
				dailySin1,
			).Set(
				feature.NewSeasonality("epoch_daily", feature.FourierCompCos, 1),
				dailyCos1,
			).Set(
				feature.NewSeasonality("myevent_daily", feature.FourierCompSin, 1),
				[]float64{
					0, 0, 0, 0, // Thursday
					0, 0, 0, 0, // Friday
					0, 0, 0, -1, // Saturday
					0, 1, 0, -1, // Sunday
					0, 1, 0, 0, // Monday
					0, 0, 0, 0, // Tuesday
					0, 0, 0, 0, // Wednesday
				},
			).Set(
				feature.NewSeasonality("myevent_daily", feature.FourierCompCos, 1),
				[]float64{
					0, 0, 0, 0, // Thursday
					0, 0, 0, 0, // Friday
					0, 0, -1, 0, // Saturday
					1, 0, -1, 0, // Sunday
					1, 0, 0, 0, // Monday
					0, 0, 0, 0, // Tuesday
					0, 0, 0, 0, // Wednesday
				},
			),
			err: nil,
		},
		"weekly seasonality with event": {
			opt: &Options{
				SeasonalityOptions: SeasonalityOptions{
					SeasonalityConfigs: []SeasonalityConfig{
						NewWeeklySeasonalityConfig(1),
					},
				},
				EventOptions: EventOptions{
					Events: []Event{
						{
							Name:  "weekend",
							Start: time.Date(1970, 1, 3, 0, 0, 0, 0, time.UTC),
							End:   time.Date(1970, 1, 5, 0, 0, 0, 0, time.UTC),
						},
					},
				},
			},
			expected: feature.NewSet().Set(
				feature.NewSeasonality("epoch_weekly", feature.FourierCompSin, 1),
				weeklySin1,
			).Set(
				feature.NewSeasonality("epoch_weekly", feature.FourierCompCos, 1),
				weeklyCos1,
			).Set(
				feature.NewSeasonality("weekend_weekly", feature.FourierCompSin, 1),
				[]float64{
					+0.0000, +0.0000, +0.0000, +0.0000, // Thursday
					+0.0000, +0.0000, +0.0000, +0.0000, // Friday
					+0.9749, +0.9009, +0.7818, +0.6234, // Saturday
					+0.4338, +0.2225, +0.0000, -0.2225, // Sunday
					+0.0000, +0.0000, +0.0000, +0.0000, // Monday
					+0.0000, +0.0000, +0.0000, +0.0000, // Tuesday
					+0.0000, +0.0000, +0.0000, +0.0000, // Wednesday
				},
			).Set(
				feature.NewSeasonality("weekend_weekly", feature.FourierCompCos, 1),
				[]float64{
					+0.0000, +0.0000, +0.0000, +0.0000, // Thursday
					+0.0000, +0.0000, +0.0000, +0.0000, // Friday
					-0.2225, -0.4338, -0.6234, -0.7818, // Saturday
					-0.9009, -0.9749, -1.0000, -0.9749, // Sunday
					+0.0000, +0.0000, +0.0000, +0.0000, // Monday
					+0.0000, +0.0000, +0.0000, +0.0000, // Tuesday
					+0.0000, +0.0000, +0.0000, +0.0000, // Wednesday
				},
			),
			err: nil,
		},
		"daily seasonality with changepoint": {
			opt: &Options{
				SeasonalityOptions: SeasonalityOptions{
					SeasonalityConfigs: []SeasonalityConfig{
						NewDailySeasonalityConfig(1),
					},
				},
				ChangepointOptions: ChangepointOptions{
					Changepoints: []Changepoint{
						{
							Name: "mychangepoint",
							T:    time.Date(1970, 1, 3, 12, 0, 0, 0, time.UTC),
						},
					},
				},
			},
			expected: feature.NewSet().Set(
				feature.NewSeasonality("epoch_daily", feature.FourierCompSin, 1),
				dailySin1,
			).Set(
				feature.NewSeasonality("epoch_daily", feature.FourierCompCos, 1),
				dailyCos1,
			).Set(
				feature.NewSeasonality("mychangepoint_daily", feature.FourierCompSin, 1),
				[]float64{
					0, 0, 0, 0, // Thursday
					0, 0, 0, 0, // Friday
					0, 0, 0, -1, // Saturday
					0, 1, 0, -1, // Sunday
					0, 1, 0, -1, // Monday
					0, 1, 0, -1, // Tuesday
					0, 1, 0, -1, // Wednesday
				},
			).Set(
				feature.NewSeasonality("mychangepoint_daily", feature.FourierCompCos, 1),
				[]float64{
					0, 0, 0, 0, // Thursday
					0, 0, 0, 0, // Friday
					0, 0, -1, 0, // Saturday
					1, 0, -1, 0, // Sunday
					1, 0, -1, 0, // Monday
					1, 0, -1, 0, // Tuesday
					1, 0, -1, 0, // Wednesday
				},
			),
			err: nil,
		},
		"weekly seasonality with colinear": {
			opt: &Options{
				SeasonalityOptions: SeasonalityOptions{
					SeasonalityConfigs: []SeasonalityConfig{
						{
							Name:   "colinear",
							Period: 3.5 * 24 * time.Hour,
							Orders: 2,
						},
						NewWeeklySeasonalityConfig(4),
					},
				},
			},
			expected: feature.NewSet().Set(
				feature.NewSeasonality("epoch_colinear", feature.FourierCompSin, 1),
				[]float64{
					+0.0000, +0.4338, +0.7818, +0.9749, // Thursday
					+0.9749, +0.7818, +0.4338, +0.0000, // Friday
					-0.4338, -0.7818, -0.9749, -0.9749, // Saturday
					-0.7818, -0.4338, +0.0000, +0.4338, // Sunday
					+0.7818, +0.9749, +0.9749, +0.7818, // Monday
					+0.4338, +0.0000, -0.4338, -0.7818, // Tuesday
					-0.9749, -0.9749, -0.7818, -0.4338, // Wednesday
				},
			).Set(
				feature.NewSeasonality("epoch_colinear", feature.FourierCompCos, 1),
				[]float64{
					+1.0000, +0.9009, +0.6234, +0.2225, // Thursday
					-0.2225, -0.6234, -0.9009, -1.0000, // Friday
					-0.9009, -0.6234, -0.2225, +0.2225, // Saturday
					+0.6234, +0.9009, +1.0000, +0.9009, // Sunday
					+0.6234, +0.2225, -0.2225, -0.6234, // Monday
					-0.9009, -1.0000, -0.9009, -0.6234, // Tuesday
					-0.2225, +0.2225, +0.6234, +0.9009, // Wednesday
				},
			).Set(
				feature.NewSeasonality("epoch_colinear", feature.FourierCompSin, 2),
				[]float64{
					+0.0000, +0.7818, +0.9749, +0.4338, // Thursday
					-0.4338, -0.9749, -0.7818, +0.0000, // Friday
					+0.7818, +0.9749, +0.4338, -0.4338, // Saturday
					-0.9749, -0.7818, +0.0000, +0.7818, // Sunday
					+0.9749, +0.4338, -0.4338, -0.9749, // Monday
					-0.7818, +0.0000, +0.7818, +0.9749, // Tuesday
					+0.4338, -0.4338, -0.9749, -0.7818, // Wednesday
				},
			).Set(
				feature.NewSeasonality("epoch_colinear", feature.FourierCompCos, 2),
				[]float64{
					+1.0000, +0.6234, -0.2225, -0.9009, // Thursday
					-0.9009, -0.2225, +0.6234, +1.0000, // Friday
					+0.6234, -0.2225, -0.9009, -0.9009, // Saturday
					-0.2225, +0.6234, +1.0000, +0.6234, // Sunday
					-0.2225, -0.9009, -0.9009, -0.2225, // Monday
					+0.6234, +1.0000, +0.6234, -0.2225, // Tuesday
					-0.9009, -0.9009, -0.2225, +0.6234, // Wednesday
				},
			).Set(
				feature.NewSeasonality("epoch_weekly", feature.FourierCompSin, 1),
				weeklySin1,
			).Set(
				feature.NewSeasonality("epoch_weekly", feature.FourierCompCos, 1),
				weeklyCos1,
			).Set(
				feature.NewSeasonality("epoch_weekly", feature.FourierCompSin, 3),
				[]float64{
					+0.0000, +0.6234, +0.9749, +0.9009, // Thursday
					+0.4338, -0.2225, -0.7818, -1.0000, // Friday
					-0.7818, -0.2225, +0.4338, +0.9009, // Saturday
					+0.9749, +0.6234, +0.0000, -0.6234, // Sunday
					-0.9749, -0.9009, -0.4338, +0.2225, // Monday
					+0.7818, +1.0000, +0.7818, +0.2225, // Tuesday
					-0.4338, -0.9009, -0.9749, -0.6234, // Wednesday
				},
			).Set(
				feature.NewSeasonality("epoch_weekly", feature.FourierCompCos, 3),
				[]float64{
					+1.0000, +0.7818, +0.2225, -0.4338, // Thursday
					-0.9009, -0.9749, -0.6234, +0.0000, // Friday
					+0.6234, +0.9749, +0.9009, +0.4338, // Saturday
					-0.2225, -0.7818, -1.0000, -0.7818, // Sunday
					-0.2225, +0.4338, +0.9009, +0.9749, // Monday
					+0.6234, +0.0000, -0.6234, -0.9749, // Tuesday
					-0.9009, -0.4338, +0.2225, +0.7818, // Wednesday
				},
			),
			err: nil,
		},
	}

	tSeries := timedataset.GenerateT(4*7, 6*time.Hour, nowFunc)
	for name, td := range testData {
		t.Run(name, func(t *testing.T) {
			startTrainTime := timedataset.TimeSlice(tSeries).StartTime()
			endTrainTime := timedataset.TimeSlice(tSeries).EndTime()
			tFeat, _ := td.opt.GenerateTimeFeatures(tSeries, startTrainTime, endTrainTime)

			chptFeat := td.opt.ChangepointOptions.GenerateFeatures(tSeries, tSeries[len(tSeries)-1], td.trained)
			tFeat.Update(chptFeat)

			res, err := td.opt.GenerateFourierFeatures(tFeat)
			if td.err != nil {
				assert.EqualError(t, err, td.err.Error())
				return
			}
			compareFeatureSet(t, td.expected, res, 1e-4)
		})
	}
}

func TestGenerateEventSeasonality(t *testing.T) {
	// Helper function to create test seasonality features using proper generation
	createTestSeasonality := func(period string, orders []int) *feature.Set {
		sFeat := feature.NewSet()

		// Create time points for testing (10 points at 6-hour intervals)
		tPoints := make([]float64, 10)
		for i := 0; i < 10; i++ {
			tPoints[i] = float64(i * 6 * 3600) // 6 hours in seconds
		}

		// Determine period based on seasonality type
		var periodDur time.Duration
		switch period {
		case "daily":
			periodDur = 24 * time.Hour
		case "weekly":
			periodDur = 7 * 24 * time.Hour
		default:
			periodDur = 24 * time.Hour
		}

		// Generate seasonality features for each order
		for _, order := range orders {
			// Create sin component
			sinFeat := feature.NewSeasonality("test_"+period, feature.FourierCompSin, order)
			sinData := sinFeat.Generate(tPoints, order, periodDur.Seconds())

			// Create cos component
			cosFeat := feature.NewSeasonality("test_"+period, feature.FourierCompCos, order)
			cosData := cosFeat.Generate(tPoints, order, periodDur.Seconds())

			sFeat.Set(sinFeat, sinData)
			sFeat.Set(cosFeat, cosData)
		}
		return sFeat
	}

	testData := map[string]struct {
		feat          *feature.Set
		sFeat         *feature.Set
		eCol          string
		sLabel        string
		expected      *feature.Set
		expectedError string
	}{
		"valid event mask with seasonality": {
			feat: feature.NewSet().Set(
				feature.NewEvent("myevent"),
				[]float64{1, 0, 1, 0, 1, 0, 1, 0, 1, 0},
			),
			sFeat:  createTestSeasonality("daily", []int{1}),
			eCol:   "myevent",
			sLabel: "daily",
			expected: feature.NewSet().Set(
				feature.NewSeasonality("myevent_daily", feature.FourierCompSin, 1),
				[]float64{0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, // sin: [0,1,0,-1,0,1,0,-1,0,1] masked with [1,0,1,0,1,0,1,0,1,0] = [0,0,0,0,0,0,0,0,0,0]
			).Set(
				feature.NewSeasonality("myevent_daily", feature.FourierCompCos, 1),
				[]float64{1, 0, -1, 0, 1, 0, -1, 0, 1, 0}, // cos: [1,0,-1,0,1,0,-1,0,1,0] masked with [1,0,1,0,1,0,1,0,1,0] = [1,0,-1,0,1,0,-1,0,1,0]
			),
			expectedError: "",
		},
		"missing event mask": {
			feat: feature.NewSet().Set(
				feature.NewTime("epoch"),
				[]float64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
			),
			sFeat:         createTestSeasonality("daily", []int{1}),
			eCol:          "nonexistent",
			sLabel:        "daily",
			expected:      nil,
			expectedError: "feature event mask not found, skipping event feature name, nonexistent",
		},
		"empty seasonality features": {
			feat: feature.NewSet().Set(
				feature.NewEvent("myevent"),
				[]float64{1, 0, 1, 0, 1, 0, 1, 0, 1, 0},
			),
			sFeat:         feature.NewSet(),
			eCol:          "myevent",
			sLabel:        "daily",
			expected:      feature.NewSet(),
			expectedError: "",
		},

		"mask shorter than feature data": {
			feat: feature.NewSet().Set(
				feature.NewEvent("myevent"),
				[]float64{1, 0, 1, 0}, // 4 elements, shorter than 10
			),
			sFeat:  createTestSeasonality("daily", []int{1}),
			eCol:   "myevent",
			sLabel: "daily",
			expected: feature.NewSet().Set(
				feature.NewSeasonality("myevent_daily", feature.FourierCompSin, 1),
				[]float64{0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, // Extended mask: [1,0,1,0,0,0,0,0,0,0] * sin [0,1,0,-1,0,1,0,-1,0,1]
			).Set(
				feature.NewSeasonality("myevent_daily", feature.FourierCompCos, 1),
				[]float64{1, 0, -1, 0, 0, 0, 0, 0, 0, 0}, // Extended mask: [1,0,1,0,0,0,0,0,0,0] * cos [1,0,-1,0,1,0,-1,0,1,0]
			),
			expectedError: "",
		},
		"multiple seasonality orders": {
			feat: feature.NewSet().Set(
				feature.NewEvent("myevent"),
				[]float64{1, 0, 1, 0, 1, 0, 1, 0, 1, 0},
			),
			sFeat:  createTestSeasonality("daily", []int{1, 2}),
			eCol:   "myevent",
			sLabel: "daily",
			expected: feature.NewSet().Set(
				feature.NewSeasonality("myevent_daily", feature.FourierCompSin, 1),
				[]float64{0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, // Order 1 sin masked
			).Set(
				feature.NewSeasonality("myevent_daily", feature.FourierCompCos, 1),
				[]float64{1, 0, -1, -0, 1, 0, -1, -0, 1, 0}, // Order 1 cos masked
			).Set(
				feature.NewSeasonality("myevent_daily", feature.FourierCompSin, 2),
				[]float64{0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, // Order 2 sin masked
			).Set(
				feature.NewSeasonality("myevent_daily", feature.FourierCompCos, 2),
				[]float64{1, -0, 1, -0, 1, -0, 1, -0, 1, -0}, // Order 2 cos masked
			),
			expectedError: "",
		},
		"nil feature set": {
			feat:          feature.NewSet(),
			sFeat:         createTestSeasonality("daily", []int{1}),
			eCol:          "myevent",
			sLabel:        "daily",
			expected:      nil,
			expectedError: "feature event mask not found, skipping event feature name, myevent",
		},
		"nil seasonality features": {
			feat: feature.NewSet().Set(
				feature.NewEvent("myevent"),
				[]float64{1, 0, 1, 0, 1, 0, 1, 0, 1, 0},
			),
			sFeat:         nil,
			eCol:          "myevent",
			sLabel:        "daily",
			expected:      feature.NewSet(),
			expectedError: "",
		},
	}

	for name, td := range testData {
		t.Run(name, func(t *testing.T) {
			result, err := generateEventSeasonality(td.feat, td.sFeat, td.eCol, td.sLabel)

			if td.expectedError != "" {
				assert.Error(t, err)
				assert.EqualError(t, err, td.expectedError)
				assert.Nil(t, result)
			} else {
				assert.NoError(t, err)
				if td.expected == nil {
					assert.Nil(t, result)
				} else {
					require.NotNil(t, result)

					compareFeatureSet(t, td.expected, result, 1e-6)
				}
			}
		})
	}
}

func TestGenerateMaskedSeasonality(t *testing.T) {
	// Test the helper function directly
	testData := map[string]struct {
		sFeat    *feature.Set
		col      string
		mask     []float64
		sLabel   string
		expected *feature.Set
	}{
		"basic masking": {
			sFeat: feature.NewSet().Set(
				feature.NewSeasonality("test_daily", feature.FourierCompSin, 1),
				[]float64{1, 2, 3, 4, 5},
			).Set(
				feature.NewSeasonality("test_daily", feature.FourierCompCos, 1),
				[]float64{2, 4, 6, 8, 10},
			),
			col:    "myevent",
			mask:   []float64{1, 0, 1, 0, 1},
			sLabel: "daily",
			expected: feature.NewSet().Set(
				feature.NewSeasonality("myevent_daily", feature.FourierCompSin, 1),
				[]float64{1, 0, 3, 0, 5},
			).Set(
				feature.NewSeasonality("myevent_daily", feature.FourierCompCos, 1),
				[]float64{2, 0, 6, 0, 10},
			),
		},
		"mask longer than feature data": {
			sFeat: feature.NewSet().Set(
				feature.NewSeasonality("test_daily", feature.FourierCompSin, 1),
				[]float64{1, 2, 3},
			),
			col:    "myevent",
			mask:   []float64{1, 0, 1, 0, 1}, // 5 elements
			sLabel: "daily",
			expected: feature.NewSet().Set(
				feature.NewSeasonality("myevent_daily", feature.FourierCompSin, 1),
				[]float64{1, 0, 3}, // Truncated mask
			),
		},
		"mask shorter than feature data": {
			sFeat: feature.NewSet().Set(
				feature.NewSeasonality("test_daily", feature.FourierCompSin, 1),
				[]float64{1, 2, 3, 4, 5},
			),
			col:    "myevent",
			mask:   []float64{1, 0}, // 2 elements
			sLabel: "daily",
			expected: feature.NewSet().Set(
				feature.NewSeasonality("myevent_daily", feature.FourierCompSin, 1),
				[]float64{1, 0, 0, 0, 0}, // Extended with zeros
			),
		},
		"empty seasonality features": {
			sFeat:    feature.NewSet(),
			col:      "myevent",
			mask:     []float64{1, 0, 1},
			sLabel:   "daily",
			expected: feature.NewSet(),
		},
		"nil seasonality features": {
			sFeat:    nil,
			col:      "myevent",
			mask:     []float64{1, 0, 1},
			sLabel:   "daily",
			expected: feature.NewSet(),
		},
	}

	for name, td := range testData {
		t.Run(name, func(t *testing.T) {
			result := generateMaskedSeasonality(td.sFeat, td.col, td.mask, td.sLabel)
			compareFeatureSet(t, td.expected, result, 1e-6)
		})
	}
}

func TestGenerateGrowthFeatures(t *testing.T) {
	testData := map[string]struct {
		epoch          []float64
		trainStart     time.Time
		trainEnd       time.Time
		growthType     string
		initialFeature *feature.Set
		expected       *feature.Set
	}{
		"intercept only": {
			epoch:      []float64{0, 43200, 86400}, // 0, 12h, 24h
			trainStart: time.Date(1970, 1, 1, 0, 0, 0, 0, time.UTC),
			trainEnd:   time.Date(1970, 1, 2, 0, 0, 0, 0, time.UTC),
			growthType: "",
			expected: feature.NewSet().Set(
				feature.Intercept(),
				[]float64{1, 1, 1},
			),
		},
		"linear growth": {
			epoch:      []float64{0, 43200, 86400}, // 0, 12h, 24h
			trainStart: time.Date(1970, 1, 1, 0, 0, 0, 0, time.UTC),
			trainEnd:   time.Date(1970, 1, 2, 0, 0, 0, 0, time.UTC),
			growthType: feature.GrowthLinear,
			expected: feature.NewSet().Set(
				feature.Intercept(),
				[]float64{1, 1, 1},
			).Set(
				feature.Linear(),
				[]float64{0.0, 0.5, 1.0},
			),
		},
		"quadratic growth": {
			epoch:      []float64{0, 43200, 86400}, // 0, 12h, 24h
			trainStart: time.Date(1970, 1, 1, 0, 0, 0, 0, time.UTC),
			trainEnd:   time.Date(1970, 1, 2, 0, 0, 0, 0, time.UTC),
			growthType: feature.GrowthQuadratic,
			expected: feature.NewSet().Set(
				feature.Intercept(),
				[]float64{1, 1, 1},
			).Set(
				feature.Quadratic(),
				[]float64{0.0, 0.25, 1.0},
			),
		},
		"zero duration": {
			epoch:      []float64{43200, 86400}, // 12h, 24h
			trainStart: time.Date(1970, 1, 1, 0, 0, 0, 0, time.UTC),
			trainEnd:   time.Date(1970, 1, 1, 0, 0, 0, 0, time.UTC), // Same as start
			growthType: feature.GrowthLinear,
			expected:   feature.NewSet(),
		},
		"single time point": {
			epoch:      []float64{43200}, // 12h
			trainStart: time.Date(1970, 1, 1, 0, 0, 0, 0, time.UTC),
			trainEnd:   time.Date(1970, 1, 2, 0, 0, 0, 0, time.UTC),
			growthType: feature.GrowthLinear,
			expected: feature.NewSet().Set(
				feature.Intercept(),
				[]float64{1},
			).Set(
				feature.Linear(),
				[]float64{0.5}, // 12h / 24h = 0.5
			),
		},
		"empty epoch": {
			epoch:      []float64{},
			trainStart: time.Date(1970, 1, 1, 0, 0, 0, 0, time.UTC),
			trainEnd:   time.Date(1970, 1, 2, 0, 0, 0, 0, time.UTC),
			growthType: feature.GrowthLinear,
			expected: feature.NewSet().Set(
				feature.Intercept(),
				[]float64{},
			).Set(
				feature.Linear(),
				[]float64{},
			),
		},
		"time before training start": {
			epoch:      []float64{-86400, 0, 43200}, // -24h, 0, 12h
			trainStart: time.Date(1970, 1, 1, 0, 0, 0, 0, time.UTC),
			trainEnd:   time.Date(1970, 1, 2, 0, 0, 0, 0, time.UTC),
			growthType: feature.GrowthLinear,
			expected: feature.NewSet().Set(
				feature.Intercept(),
				[]float64{1, 1, 1},
			).Set(
				feature.Linear(),
				[]float64{-1.0, 0.0, 0.5}, // Can be negative for times before start
			),
		},
		"time after training end": {
			epoch:      []float64{0, 86400, 172800}, // 0, 24h, 48h
			trainStart: time.Date(1970, 1, 1, 0, 0, 0, 0, time.UTC),
			trainEnd:   time.Date(1970, 1, 2, 0, 0, 0, 0, time.UTC),
			growthType: feature.GrowthLinear,
			expected: feature.NewSet().Set(
				feature.Intercept(),
				[]float64{1, 1, 1},
			).Set(
				feature.Linear(),
				[]float64{0.0, 1.0, 2.0}, // Can be > 1.0 for times after end
			),
		},
		"invalid growth type": {
			epoch:      []float64{0, 43200, 86400},
			trainStart: time.Date(1970, 1, 1, 0, 0, 0, 0, time.UTC),
			trainEnd:   time.Date(1970, 1, 2, 0, 0, 0, 0, time.UTC),
			growthType: "invalid",
			expected: feature.NewSet().Set(
				feature.Intercept(),
				[]float64{1, 1, 1},
			), // Should only add intercept, ignore invalid type
		},
		"pre-existing features": {
			epoch:      []float64{0, 43200},
			trainStart: time.Date(1970, 1, 1, 0, 0, 0, 0, time.UTC),
			trainEnd:   time.Date(1970, 1, 2, 0, 0, 0, 0, time.UTC),
			growthType: feature.GrowthLinear,
			initialFeature: feature.NewSet().Set(
				feature.NewTime("existing"),
				[]float64{99, 88},
			),
			expected: feature.NewSet().Set(
				feature.NewTime("existing"),
				[]float64{99, 88},
			).Set(
				feature.Intercept(),
				[]float64{1, 1},
			).Set(
				feature.Linear(),
				[]float64{0.0, 0.5},
			),
		},
		"multiple time points quadratic": {
			epoch:      []float64{0, 21600, 43200, 64800, 86400}, // 0, 6h, 12h, 18h, 24h
			trainStart: time.Date(1970, 1, 1, 0, 0, 0, 0, time.UTC),
			trainEnd:   time.Date(1970, 1, 2, 0, 0, 0, 0, time.UTC),
			growthType: feature.GrowthQuadratic,
			expected: feature.NewSet().Set(
				feature.Intercept(),
				[]float64{1, 1, 1, 1, 1},
			).Set(
				feature.Quadratic(),
				[]float64{0.0, 0.0625, 0.25, 0.5625, 1.0}, // (0/24)^2, (6/24)^2, (12/24)^2, (18/24)^2, (24/24)^2
			),
		},
		"large time range linear": {
			epoch:      []float64{0, 86400, 2592000}, // 0, 1d, 30d
			trainStart: time.Date(1970, 1, 1, 0, 0, 0, 0, time.UTC),
			trainEnd:   time.Date(1970, 1, 31, 0, 0, 0, 0, time.UTC),
			growthType: feature.GrowthLinear,
			expected: feature.NewSet().Set(
				feature.Intercept(),
				[]float64{1, 1, 1},
			).Set(
				feature.Linear(),
				[]float64{0.0, 1.0 / 30.0, 1.0}, // 0, 1/30, 1
			),
		},
		"small time range linear": {
			epoch:      []float64{0, 60, 120, 180}, // 0, 1m, 2m, 3m
			trainStart: time.Date(1970, 1, 1, 0, 0, 0, 0, time.UTC),
			trainEnd:   time.Date(1970, 1, 1, 0, 5, 0, 0, time.UTC), // 5 minute range
			growthType: feature.GrowthLinear,
			expected: feature.NewSet().Set(
				feature.Intercept(),
				[]float64{1, 1, 1, 1},
			).Set(
				feature.Linear(),
				[]float64{0.0, 0.2, 0.4, 0.6}, // 0/300, 60/300, 120/300, 180/300
			),
		},
	}

	for name, td := range testData {
		t.Run(name, func(t *testing.T) {
			tFeat := td.initialFeature
			if tFeat == nil {
				tFeat = feature.NewSet()
			}

			// Create Options instance to access private generateGrowthFeatures method
			opt := &Options{
				GrowthType: td.growthType,
			}
			opt.generateGrowthFeatures(td.epoch, td.trainStart, td.trainEnd, tFeat)

			compareFeatureSet(t, td.expected, tFeat, 1e-6)
		})
	}
}
