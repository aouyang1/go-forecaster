package options

import (
	"fmt"
	"testing"
	"time"

	"github.com/aouyang1/go-forecaster/feature"
	"github.com/aouyang1/go-forecaster/models"
	"github.com/aouyang1/go-forecaster/timedataset"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gonum.org/v1/gonum/dsp/window"
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

func TestWindowFunc(t *testing.T) {
	testData := map[string]struct {
		name     string
		expected func([]float64) []float64
	}{
		"bartlett hann":    {WindowBartlettHann, window.BartlettHann},
		"blackman":         {WindowBlackman, window.Blackman},
		"blackman harris":  {WindowBlackmanHarris, window.BlackmanHarris},
		"blackman nuttall": {WindowBlackmanNuttall, window.BlackmanNuttall},
		"flat top":         {WindowFlatTop, window.FlatTop},
		"hamming":          {WindowHamming, window.Hamming},
		"hann":             {WindowHann, window.Hann},
		"lanczos":          {WindowLanczos, window.Lanczos},
		"nuttall":          {WindowNuttall, window.Nuttall},
		"rectangular":      {WindowRectangular, window.Rectangular},
		"sine":             {WindowSine, window.Sine},
		"triangular":       {WindowTriangular, window.Triangular},
		"unknown":          {"unknown", window.Rectangular},
	}

	numPnts := 10
	for name, td := range testData {
		t.Run(name, func(t *testing.T) {
			res := WindowFunc(td.name)

			seqRes := make([]float64, numPnts)
			floats.AddConst(1.0, seqRes)

			seqExp := make([]float64, numPnts)
			floats.AddConst(1.0, seqExp)

			assert.Equal(t, td.expected(seqExp), res(seqRes))
		})
	}
}

func TestNewLassoAutoAptions(t *testing.T) {
	testData := map[string]struct {
		opt      *Options
		expected *models.LassoAutoOptions
	}{
		"defaults": {
			opt: &Options{},
			expected: &models.LassoAutoOptions{
				Lambdas:         []float64{1.0},
				FitIntercept:    false,
				Iterations:      models.DefaultIterations,
				Tolerance:       models.DefaultTolerance,
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
			expected: &models.LassoAutoOptions{
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
				feature.NewGrowth(feature.GrowthIntercept),
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
				feature.NewGrowth(feature.GrowthIntercept),
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
				feature.NewGrowth(feature.GrowthIntercept),
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
				feature.NewGrowth(feature.GrowthIntercept),
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
				feature.NewGrowth(feature.GrowthIntercept),
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
				feature.NewGrowth(feature.GrowthIntercept),
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
				feature.NewGrowth(feature.GrowthIntercept),
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
				feature.NewGrowth(feature.GrowthIntercept),
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
				feature.NewGrowth(feature.GrowthIntercept),
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
				feature.NewGrowth(feature.GrowthIntercept),
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
				feature.NewGrowth(feature.GrowthIntercept),
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
				feature.NewGrowth(feature.GrowthIntercept),
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
				feature.NewGrowth(feature.GrowthIntercept),
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
				feature.NewGrowth(feature.GrowthIntercept),
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
				feature.NewGrowth(feature.GrowthIntercept),
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
				feature.NewGrowth(feature.GrowthIntercept),
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
				feature.NewGrowth(feature.GrowthIntercept),
				ones7DaysAt6Hr,
			).Set(
				feature.NewTime("epoch"),
				epoch7DaysAt6Hr,
			),
		},
	}

	for name, td := range testData {
		t.Run(name, func(t *testing.T) {
			features, _ := td.opt.GenerateTimeFeatures(td.t)
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
			tFeat, _ := td.opt.GenerateTimeFeatures(tSeries)
			res, err := td.opt.GenerateFourierFeatures(tFeat)
			if td.err != nil {
				assert.EqualError(t, err, td.err.Error())
				return
			}
			compareFeatureSet(t, td.expected, res, 1e-4)
		})
	}
}
